use rand::RngExt;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::HashMap;

use crate::evolution::config::EvolutionConfig;
use crate::evolution::individual::Individual;
use crate::operators::crossover::crossover;
use crate::operators::generator::generate_random_ast;
use crate::operators::mutation::{constant_perturbation, point_mutation, subtree_mutation};
use crate::operators::selection::tournament_selection;

pub struct Island {
    pub individuals: Vec<Individual>,
    pub best_individual: Individual,
    pub rng: Xoshiro256PlusPlus,
    pub stagnation_counter: usize,
    pub local_hof: HashMap<usize, (f64, Individual)>,
}

impl Island {
    pub fn new(size: usize, seed: u64, num_features: usize) -> Self {
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
            best_individual,
            rng,
            stagnation_counter: 0,
            local_hof: HashMap::new(),
        }
    }

    pub fn step_generation(&mut self, data_x: &[Vec<f64>], data_y: &[f64], config: &EvolutionConfig) {
        let num_features = if data_x.is_empty() { 1 } else { data_x[0].len() };
        
        let mut next_gen = self.create_next_generation(config, num_features);
        let improved_this_gen = self.evaluate_and_optimize(&mut next_gen, data_x, data_y, config);
        
        self.individuals = next_gen;
        self.handle_stagnation(improved_this_gen, config, num_features, data_x, data_y);
    }

    fn create_next_generation(&mut self, config: &EvolutionConfig, num_features: usize) -> Vec<Individual> {
        let pop_size = self.individuals.capacity();
        let mut next_gen = Vec::with_capacity(pop_size);
        next_gen.push(self.best_individual.clone());

        while next_gen.len() < pop_size {
            let p: f64 = self.rng.random();
            if p < config.crossover_rate {
                let parent1 = tournament_selection(&self.individuals, config.tournament_size, &mut self.rng);
                let parent2 = tournament_selection(&self.individuals, config.tournament_size, &mut self.rng);

                let mut child = crossover(parent1, parent2, &mut self.rng);
                child.simplify();
                next_gen.push(child);
            } else {
                let mut child = tournament_selection(&self.individuals, config.tournament_size, &mut self.rng).clone();
                let mut_type = self.rng.random_range(0..3);
                
                match mut_type {
                    0 => point_mutation(&mut child, &mut self.rng, num_features),
                    1 => constant_perturbation(&mut child, &mut self.rng),
                    _ => subtree_mutation(&mut child, &mut self.rng, num_features),
                }
                child.simplify();
                next_gen.push(child);
            }
        }
        next_gen
    }

    fn evaluate_and_optimize(&mut self, next_gen: &mut Vec<Individual>, data_x: &[Vec<f64>], data_y: &[f64], config: &EvolutionConfig) -> bool {
        let mut improved_this_gen = false;

        for ind in next_gen.iter_mut() {
            if self.rng.random::<f64>() < config.opt_prob {
                ind.optimize_constants(
                    data_x, 
                    data_y, 
                    config.opt_iterations,
                    config.opt_lr,
                    config.opt_epsilon
                );
            }

            let mse = ind.calculate_mse(data_x, data_y);

            if mse.is_finite() {
                let complexity = ind.complexity();
                let is_new_best = match self.local_hof.get(&complexity) {
                    Some(&(best_mse, _)) => mse < best_mse,
                    None => true,
                };
                if is_new_best {
                    self.local_hof.insert(complexity, (mse, ind.clone()));
                }
            }

            let complexity_penalty = (ind.complexity() as f64) * config.parsimony_penalty;

            if mse.is_finite() {
                ind.fitness = mse + complexity_penalty;
            } else {
                ind.fitness = f64::MAX;
            }

            if ind.fitness < self.best_individual.fitness {
                self.best_individual = ind.clone();
                improved_this_gen = true;
            }
        }
        improved_this_gen
    }

    fn handle_stagnation(&mut self, improved: bool, config: &EvolutionConfig, num_features: usize, data_x: &[Vec<f64>], data_y: &[f64]) {
        if improved {
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }

        if self.stagnation_counter >= config.stagnation_threshold {
            println!("NUKING!");
            let pop_size = self.individuals.capacity();
            self.individuals.clear();
            self.individuals.push(self.best_individual.clone());
            
            for _ in 1..pop_size {
                let ast = generate_random_ast(5, &mut self.rng, num_features);
                let mut new_ind = Individual::new(ast);
                new_ind.simplify();
                
                let mse = new_ind.calculate_mse(data_x, data_y);
                let penalty = (new_ind.complexity() as f64) * config.parsimony_penalty;
                if mse.is_finite() {
                    new_ind.fitness = mse + penalty;
                }
                
                self.individuals.push(new_ind);
            }
            
            self.stagnation_counter = 0;
        }
    }
}