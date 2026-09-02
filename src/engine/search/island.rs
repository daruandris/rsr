use rand::RngExt;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::HashMap;
use std::mem;

use crate::Instruction;
use crate::engine::data::dataset::Dataset;
use crate::engine::eval::types::ValueType;
use crate::engine::search::individual::Individual;
use crate::engine::search::operators::crossover::crossover;
use crate::engine::search::operators::generator::generate_random_ast;
use crate::engine::search::operators::mutation::{
    constant_perturbation, point_mutation, subtree_mutation,
};
use crate::engine::search::operators::selection::{
    assign_rank_and_crowding_distance, tournament_selection_pareto,
};
use crate::engine::search::strategy::Strategy;

pub struct Island<S: Strategy> {
    pub individuals: Vec<Individual>,
    pub next_gen_buffer: Vec<Individual>,
    pub best_individual: Individual,
    pub rng: Xoshiro256PlusPlus,
    pub stagnation_counter: usize,
    pub local_hof: HashMap<usize, (f32, Individual)>,
    pub strategy: S,
    pub allowed_ops: Vec<Instruction>,
    pub variable_registry: Vec<(ValueType, u8)>,
}

impl<S: Strategy> Island<S> {
    pub fn new(
        seed: u64,
        variable_registry: Vec<(ValueType, u8)>,
        strategy: S,
        allowed_ops: Vec<Instruction>,
    ) -> Self {
        let size = strategy.island_size();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut individuals = Vec::with_capacity(size);
        for _ in 0..size {
            let (ast, constants) = generate_random_ast(
                ValueType::Float,
                5,
                &mut rng,
                &variable_registry,
                &allowed_ops,
                strategy.disabled_constant_types(),
            );
            let mut ind = Individual::new(ast, constants, strategy.disabled_constant_types().to_vec());
            ind.simplify();
            individuals.push(ind);
        }

        let best_individual = individuals[0].clone();
        Self {
            individuals,
            next_gen_buffer: Vec::with_capacity(size),
            best_individual,
            rng,
            stagnation_counter: 0,
            local_hof: HashMap::new(),
            strategy,
            allowed_ops,
            variable_registry,
        }
    }

    pub fn step_generation(&mut self, dataset: &Dataset, mini_batch: &Dataset) {
        let old_best_fitness = self.best_individual.fitness;
        for ind in self.individuals.iter_mut() {
            ind.age += 1;
        }
        assign_rank_and_crowding_distance(&mut self.individuals);

        self.fill_next_generation(mini_batch);
        self.evaluate_buffer(dataset, mini_batch);
        mem::swap(&mut self.individuals, &mut self.next_gen_buffer);

        let improvement = old_best_fitness - self.best_individual.fitness;
        if improvement > self.strategy.min_improvement() {
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }

        self.strategy
            .on_generation_end(self.best_individual.fitness, self.stagnation_counter);
        if self.stagnation_counter >= self.strategy.stagnation_threshold() * 4 {
            self.nuke(dataset);
            self.strategy.on_nuke();
        }
    }

    fn fill_next_generation(&mut self, mini_batch: &Dataset) {
        self.next_gen_buffer.clear();
        let pop_size = self.individuals.capacity();

        self.next_gen_buffer.push(self.best_individual.clone());

        let num_randoms = (pop_size as f32 * self.strategy.random_injection_rate())
            .max(self.strategy.min_random_injection() as f32) as usize;
        for _ in 0..num_randoms {
            if self.next_gen_buffer.len() >= pop_size {
                break;
            }
            let (ast, constants) = generate_random_ast(
                ValueType::Float,
                5,
                &mut self.rng,
                &self.variable_registry,
                &self.allowed_ops,
                self.strategy.disabled_constant_types(),
            );
            let mut ind = Individual::new(ast, constants, self.strategy.disabled_constant_types().to_vec());
            ind.simplify();
            self.next_gen_buffer.push(ind);
        }

        while self.next_gen_buffer.len() < pop_size {
            let p: f32 = self.rng.random();
            let tourn_size = self.strategy.tournament_size();

            if p < self.strategy.crossover_rate() {
                let parent1 =
                    tournament_selection_pareto(&self.individuals, tourn_size, &mut self.rng);
                let parent2 =
                    tournament_selection_pareto(&self.individuals, tourn_size, &mut self.rng);

                let mut child = crossover(
                    parent1,
                    parent2,
                    &mut self.rng,
                    self.strategy.max_tree_size(),
                );
                if child.has_forbidden_patterns() {
                    child = parent1.clone();
                }
                child.age = parent1.age.max(parent2.age);
                child.simplify();
                child.invalidate();
                self.next_gen_buffer.push(child);
            } else {
                let mut candidate = tournament_selection_pareto(&self.individuals, tourn_size, &mut self.rng).clone();
                let mut current_mse = candidate.calculate_mse(mini_batch);
                candidate.fitness = f32::MAX;

                for _ in 0..self.strategy.mutation_cycles() {
                    let mut mutated_candidate = candidate.clone();
                    match self.rng.random_range(0..3) {
                        0 => point_mutation(
                            &mut mutated_candidate,
                            &mut self.rng,
                            &self.variable_registry,
                            &self.allowed_ops,
                            self.strategy.disabled_constant_types(),
                        ),
                        1 => constant_perturbation(&mut mutated_candidate, &mut self.rng),
                        _ => subtree_mutation(
                            &mut mutated_candidate,
                            &mut self.rng,
                            &self.variable_registry,
                            self.strategy.max_tree_size(),
                            self.strategy.mutation_max_depth(),
                            &self.allowed_ops,
                            self.strategy.disabled_constant_types(),
                        ),
                    }
                    mutated_candidate.simplify();
                    if !mutated_candidate.has_forbidden_patterns() {
                        let new_mse = mutated_candidate.calculate_mse(mini_batch);
                        if new_mse < current_mse {
                            candidate = mutated_candidate;
                            current_mse = new_mse;
                        }
                    }
                }
                self.next_gen_buffer.push(candidate);
            }
        }
    }

    pub fn nuke(&mut self, dataset: &Dataset) {
        println!("NUKE");
        let pop_size = self.individuals.capacity();
        self.individuals.clear();
        self.individuals.push(self.best_individual.clone());
        for _ in 1..pop_size {
            let (ast, constants) = generate_random_ast(
                ValueType::Float,
                5,
                &mut self.rng,
                &self.variable_registry,
                &self.allowed_ops,
                self.strategy.disabled_constant_types(),
            );
            let mut new_ind = Individual::new(ast, constants, self.strategy.disabled_constant_types().to_vec());
            new_ind.simplify();

            let mse = new_ind.calculate_mse(dataset);

            if mse.is_finite() {
                let mse_floor = dataset.target_variance * 0.01;
                let dynamic_penalty_rate = self.strategy.parsimony_penalty() * 
                    mse.max(mse_floor).max(self.strategy.target_mse());
                
                let penalty = (new_ind.complexity() as f32) * dynamic_penalty_rate;
                new_ind.fitness = mse + penalty;
            } else {
                new_ind.fitness = f32::MAX;
            }
            self.individuals.push(new_ind);
        }
        self.stagnation_counter = 0;
    }

    fn evaluate_buffer(&mut self, dataset: &Dataset, mini_batch: &Dataset) {
        let mut best_clone = self.best_individual.clone();
        let baseline_mini_mse = best_clone.calculate_mse(mini_batch);

        for ind in self.next_gen_buffer.iter_mut() {
            let mini_mse = ind.calculate_mse(mini_batch);
            let mut is_promising = mini_mse < (baseline_mini_mse * 2.0);
            if !is_promising && self.rng.random::<f32>() < 0.05 {
                is_promising = true;
            }

            let mut final_mse = mini_mse;

            if is_promising {
                final_mse = ind.calculate_mse(dataset);
                
                let complexity = ind.complexity();
                let mse_floor = dataset.target_variance * 0.01;
                let dynamic_penalty_rate = self.strategy.parsimony_penalty() * 
                    final_mse.max(mse_floor).max(self.strategy.target_mse());
                
                let temp_fitness = final_mse + ((complexity as f32) * dynamic_penalty_rate);

                let required_improvement = (self.best_individual.fitness * 0.005)
                    .max(self.strategy.min_improvement());
                let is_potential_elite = temp_fitness < (self.best_individual.fitness - required_improvement);
                let random_opt = self.rng.random::<f32>() < self.strategy.opt_prob();

                if (is_potential_elite || random_opt) && final_mse.is_finite() {
                    ind.optimize_constants(dataset, self.strategy.opt_iterations());
                    if ind.program.is_none() {
                        ind.compile();
                    }
                    final_mse = ind.calculate_mse(dataset);
                }
            }
            if final_mse.is_finite() {
                let complexity = ind.complexity();
                let is_new_best = match self.local_hof.get(&complexity) {
                    Some(&(best_mse, _)) => final_mse < best_mse,
                    None => true,
                };
                if is_new_best {
                    self.local_hof.insert(complexity, (final_mse, ind.clone()));
                }
                
                let mse_floor = dataset.target_variance * 0.01;
                let dynamic_penalty_rate = self.strategy.parsimony_penalty() * 
                    final_mse.max(mse_floor).max(self.strategy.target_mse());
                
                let complexity_penalty = (complexity as f32) * dynamic_penalty_rate;
                ind.fitness = final_mse + complexity_penalty;
            } else {
                ind.fitness = f32::MAX;
            }

            if ind.fitness < self.best_individual.fitness {
                self.best_individual = ind.clone();
            }
        }
    }
}
