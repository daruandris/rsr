// src/engine/island.rs
use std::mem;
use std::collections::HashMap;
use rand::SeedableRng;
use rand::RngExt;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::domain::Domain;
use crate::engine::strategy::Strategy;
use crate::engine::individual::Individual;
use crate::metrics::dataset::SimdDataset;

// Az operátorok is generikusak lesznek: `<D: Domain>`
use crate::operators::crossover::crossover;
use crate::operators::generator::generate_random_ast;
use crate::operators::mutation::{constant_perturbation, point_mutation, subtree_mutation};
use crate::operators::selection::{assign_rank_and_crowding_distance, tournament_selection_pareto};

pub struct Island<S: Strategy, D: Domain> {
    pub individuals: Vec<Individual<D>>,
    pub next_gen_buffer: Vec<Individual<D>>,
    pub best_individual: Individual<D>,
    pub rng: Xoshiro256PlusPlus,
    pub stagnation_counter: usize,
    pub local_hof: HashMap<usize, (f32, Individual<D>)>,
    pub strategy: S,
}

impl<S: Strategy, D: Domain> Island<S, D> {
    pub fn new(seed: u64, num_features: u8, strategy: S) -> Self {
        let size = strategy.island_size();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut individuals = Vec::with_capacity(size);
        for _ in 0..size {
            let ast = generate_random_ast::<D>(5, &mut rng, num_features);
            let mut ind = Individual::new(ast);
            ind.simplify();
            individuals.push(ind);
        }

        let best_individual = individuals[0].clone();

        Self {
            individuals,
            next_gen_buffer : Vec::with_capacity(size),
            best_individual,
            rng,
            stagnation_counter: 0,
            local_hof: HashMap::new(),
            strategy,
        }
    }

    pub fn step_generation(&mut self, dataset: &SimdDataset) {
        let num_features = dataset.num_features;
        let old_best_fitness = self.best_individual.fitness;
        
        for ind in self.individuals.iter_mut() { ind.age += 1; }
        assign_rank_and_crowding_distance(&mut self.individuals);

        self.fill_next_generation(dataset, num_features);
        self.evaluate_buffer(dataset);

        mem::swap(&mut self.individuals, &mut self.next_gen_buffer);
        
        let improvement = old_best_fitness - self.best_individual.fitness;
        if improvement > self.strategy.min_improvement() {
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }

        self.strategy.on_generation_end(self.best_individual.fitness, self.stagnation_counter);

        if self.stagnation_counter >= self.strategy.stagnation_threshold() * 4 { 
            self.nuke(dataset);
            self.strategy.on_nuke(); 
        }
    }

    fn fill_next_generation(&mut self, dataset: &SimdDataset, num_features: u8) {
        self.next_gen_buffer.clear();
        let pop_size = self.individuals.capacity();

        self.next_gen_buffer.push(self.best_individual.clone());

        let num_randoms = (pop_size as f32 * self.strategy.random_injection_rate())
            .max(self.strategy.min_random_injection() as f32) as usize; 

        for _ in 0..num_randoms {
            if self.next_gen_buffer.len() >= pop_size { break; }
            let ast = generate_random_ast::<D>(5, &mut self.rng, num_features);
            let mut ind = Individual::new(ast);
            ind.simplify();
            self.next_gen_buffer.push(ind);
        }

        while self.next_gen_buffer.len() < pop_size {
            let p: f32 = self.rng.random();
            let tourn_size = self.strategy.tournament_size();
            
            if p < self.strategy.crossover_rate() {
                let parent1 = tournament_selection_pareto(&self.individuals, tourn_size, &mut self.rng);
                let parent2 = tournament_selection_pareto(&self.individuals, tourn_size, &mut self.rng);

                let mut child = crossover::<D>(parent1, parent2, &mut self.rng, self.strategy.max_tree_size());
                child.age = parent1.age.max(parent2.age);
                child.simplify();
                child.invalidate();
                self.next_gen_buffer.push(child);
            } else {
                let mut candidate = tournament_selection_pareto(&self.individuals, tourn_size, &mut self.rng).clone();
                
                if candidate.fitness == f32::MAX { candidate.fitness = candidate.calculate_mse(dataset); }
                let mut current_fitness = candidate.fitness;

                for _ in 0..self.strategy.mutation_cycles() {
                    let mut mutated_candidate = candidate.clone();
                    
                    match self.rng.random_range(0..3) {
                        0 => point_mutation::<D>(&mut mutated_candidate, &mut self.rng, num_features),
                        1 => constant_perturbation::<D>(&mut mutated_candidate, &mut self.rng),
                        _ => subtree_mutation::<D>(&mut mutated_candidate, &mut self.rng, num_features, self.strategy.max_tree_size(), self.strategy.mutation_max_depth()),
                    }
                    mutated_candidate.simplify();
                    let new_mse = mutated_candidate.calculate_mse(dataset);
                    
                    if new_mse < current_fitness {
                        candidate = mutated_candidate;
                        current_fitness = new_mse;
                        candidate.fitness = new_mse;
                    }
                }
                self.next_gen_buffer.push(candidate);
            }
        }
    }

    pub fn nuke(&mut self, dataset: &SimdDataset) {
        let num_features = dataset.num_features;
        let pop_size = self.individuals.capacity();

        self.individuals.clear();
        self.individuals.push(self.best_individual.clone());

        for _ in 1..pop_size {
            let ast = generate_random_ast::<D>(5, &mut self.rng, num_features);
            let mut new_ind = Individual::new(ast);
            new_ind.simplify();
            
            let mse = new_ind.calculate_mse(dataset);
            let penalty = (new_ind.complexity() as f32) * self.strategy.parsimony_penalty();
            if mse.is_finite() { new_ind.fitness = mse + penalty; }
            self.individuals.push(new_ind);
        }
        self.stagnation_counter = 0;
    }

    fn evaluate_buffer(&mut self, dataset: &SimdDataset) {
        for ind in self.next_gen_buffer.iter_mut() {
            if self.rng.random::<f32>() < self.strategy.opt_prob() {
                ind.optimize_constants(dataset, self.strategy.opt_iterations());
                if ind.program.is_none() { ind.compile(); } 
            }

            let mse = ind.calculate_mse(dataset);

            if mse.is_finite() {
                let complexity = ind.complexity();
                let is_new_best = match self.local_hof.get(&complexity) {
                    Some(&(best_mse, _)) => mse < best_mse,
                    None => true,
                };
                if is_new_best {
                    self.local_hof.insert(complexity, (mse, ind.clone()));
                }

                let complexity_penalty = (complexity as f32) * self.strategy.parsimony_penalty();
                ind.fitness = mse + complexity_penalty;
            } else {
                ind.fitness = f32::MAX;
            }

            if ind.fitness < self.best_individual.fitness {
                self.best_individual = ind.clone();
            }
        }
    }
}