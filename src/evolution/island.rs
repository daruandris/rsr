use rand::RngExt;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::HashMap;
use std::mem;

use crate::evolution::config::EvolutionConfig;
use crate::evolution::individual::Individual;
use crate::metrics::dataset::SimdDataset;
use crate::operators::crossover::crossover;
use crate::operators::generator::generate_random_ast;
use crate::operators::mutation::{constant_perturbation, point_mutation, subtree_mutation};
use crate::operators::selection::assign_rank_and_crowding_distance;
use crate::operators::selection::{tournament_selection_pareto};

pub struct Island {
    pub individuals: Vec<Individual>,
    pub next_gen_buffer: Vec<Individual>,
    pub best_individual: Individual,
    pub rng: Xoshiro256PlusPlus,
    pub stagnation_counter: usize,
    pub local_hof: HashMap<usize, (f32, Individual)>,
}

impl Island {
    pub fn new(size: usize, seed: u64, num_features: u8) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut individuals = Vec::with_capacity(size);
        for _ in 0..size {
            let ast = generate_random_ast(5, &mut rng, num_features);
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
        }
    }

    pub fn step_generation(&mut self, dataset: &SimdDataset, config: &EvolutionConfig) {
        let num_features = dataset.num_features;
        let old_best_fitness = self.best_individual.fitness;
        for ind in self.individuals.iter_mut() {
            ind.age += 1;
        }

        assign_rank_and_crowding_distance(&mut self.individuals);

        self.fill_next_generation(config, num_features);

        self.evaluate_buffer(dataset, config);

        mem::swap(&mut self.individuals, &mut self.next_gen_buffer);
        
        let improvement = old_best_fitness - self.best_individual.fitness;
        if improvement > config.min_improvement {
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }

        if self.stagnation_counter >= config.stagnation_threshold * 4 { 
            self.nuke(dataset, config);
        }
    }

    fn fill_next_generation(&mut self, config: &EvolutionConfig, num_features: u8) {
        self.next_gen_buffer.clear();
        let pop_size = self.individuals.capacity();

        let elite = self.best_individual.clone();
        self.next_gen_buffer.push(elite);

        let num_randoms = (pop_size as f32 * config.random_injection_rate)
            .max(config.min_random_injection as f32) as usize; 

        for _ in 0..num_randoms {
            if self.next_gen_buffer.len() >= pop_size { break; }
            let ast = generate_random_ast(5, &mut self.rng, num_features);
            let mut ind = Individual::new(ast);
            ind.simplify();
            self.next_gen_buffer.push(ind);
        }

        while self.next_gen_buffer.len() < pop_size {
            let p: f32 = self.rng.random();
            
            if p < config.crossover_rate {
                let parent1 = tournament_selection_pareto(&self.individuals, config.tournament_size, &mut self.rng);
                let parent2 = tournament_selection_pareto(&self.individuals, config.tournament_size, &mut self.rng);

                let mut child = crossover(parent1, parent2, &mut self.rng, config.max_tree_size);
                child.age = parent1.age.max(parent2.age);
                
                child.simplify();
                child.invalidate();
                self.next_gen_buffer.push(child);
            } else {
                let parent = tournament_selection_pareto(&self.individuals, config.tournament_size, &mut self.rng);
                let mut child = parent.clone();
                child.age = parent.age; 
                
                let mut_type = self.rng.random_range(0..3);
                match mut_type {
                    0 => point_mutation(&mut child, &mut self.rng, num_features),
                    1 => constant_perturbation(&mut child, &mut self.rng),
                    _ => subtree_mutation(&mut child, &mut self.rng, num_features, config.max_tree_size),
                }
                child.simplify();
                child.invalidate();
                self.next_gen_buffer.push(child);
            }
        }
    }

    pub fn nuke(&mut self, dataset: &SimdDataset, config: &EvolutionConfig) {
        if config.verbose {  println!("NUKING ISLAND!"); }
       
        let num_features = dataset.num_features;
        let pop_size = self.individuals.capacity();

        self.individuals.clear();
        self.individuals.push(self.best_individual.clone());

        for _ in 1..pop_size {
            let ast = generate_random_ast(5, &mut self.rng, num_features);
            let mut new_ind = Individual::new(ast);
            new_ind.simplify();
            
            let mse = new_ind.calculate_mse(dataset);
            let penalty = (new_ind.complexity() as f32) * config.parsimony_penalty;
            if mse.is_finite() {
                new_ind.fitness = mse + penalty;
            }
            self.individuals.push(new_ind);
        }
        self.stagnation_counter = 0;
    }


    fn evaluate_buffer(&mut self, dataset: &SimdDataset, config: &EvolutionConfig) {
        for ind in self.next_gen_buffer.iter_mut() {
            
            if self.rng.random::<f32>() < config.opt_prob {
                ind.optimize_constants(dataset, config.opt_iterations);
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

                let complexity_penalty = (complexity as f32) * config.parsimony_penalty;
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