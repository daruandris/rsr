// src/engine/mod.rs
use rayon::prelude::*;
use std::collections::HashMap;

use crate::domain::Domain;
use crate::engine::strategy::Strategy;
use crate::engine::individual::Individual;
use crate::engine::island::Island;
use crate::metrics::dataset::SimdDataset;

pub mod strategy;
pub mod individual;
pub mod island;
pub mod config;

pub struct Engine<S: Strategy, D: Domain> {
    pub islands: Vec<Island<S, D>>,
    pub global_strategy: S,
    pub global_hof: HashMap<usize, (f32, Individual<D>)>,
}

impl<S: Strategy, D: Domain> Engine<S, D> {
    pub fn new(strategy: S, num_features: u8, allowed_ops: Vec<D::Operator>) -> Self {
        let num_islands = strategy.num_islands();
        let mut islands = Vec::with_capacity(num_islands);
        
        for i in 0..num_islands {
            islands.push(Island::new(42 + i as u64, num_features, strategy.clone(), allowed_ops.clone()));
        }
        
        Self { islands, global_strategy: strategy, global_hof: HashMap::new() }
    }

    pub fn run_evolution(&mut self, dataset: &SimdDataset) {
        let max_generations = self.global_strategy.max_generations();
        let target_mse = self.global_strategy.target_mse();
        let migration_interval = self.global_strategy.migration_interval();
        let verbose = self.global_strategy.verbose();
        
        for generation in 0..max_generations {
            self.islands.par_iter_mut().for_each(|island| {
                island.step_generation(dataset);
            });

            for island in &self.islands {
                for (&complexity, &(mse, ref ind)) in &island.local_hof {
                    let is_global_best = match self.global_hof.get(&complexity) {
                        Some(&(best_mse, _)) => mse < best_mse,
                        None => true,
                    };
                    if is_global_best {
                        self.global_hof.insert(complexity, (mse, ind.clone()));
                    }
                }
            }

            let pure_mse = self.get_global_best().clone().calculate_mse(dataset);

            if pure_mse <= target_mse {
                if verbose { println!("\n>>> Target reached in {}. generation: {}! <<<", generation, self.get_global_best()); }
                break;
            }

            if generation > 0 && generation % migration_interval == 0 {
                self.migrate_individuals();
                if verbose { println!("Generation: {}, Best MSE: {}, Equation: {}", generation, pure_mse, self.get_global_best()); }
            }
        }
        
        let final_opt_iters = self.global_strategy.final_opt_iterations();
        let mut final_best = self.get_global_best().clone();
        final_best.optimize_constants(dataset, final_opt_iters);
        
        // Csatold ide a végső printelést (a dataset real-world skálázásával)
    }

    fn migrate_individuals(&mut self) {
        let num_islands = self.islands.len();
        if num_islands < 2 { return; }
        let mut migrants: Vec<Individual<D>> = self.islands.iter()
            .map(|island| island.best_individual.clone())
            .collect();
        migrants.rotate_right(1);

        for (island, migrant) in self.islands.iter_mut().zip(migrants.into_iter()) {
            let last_idx = island.individuals.len() - 1;
            island.individuals[last_idx] = migrant;
        }
    }

    pub fn get_global_best(&self) -> &Individual<D> {
        self.islands.iter()
            .min_by(|a, b| a.best_individual.fitness.partial_cmp(&b.best_individual.fitness).unwrap())
            .map(|island| &island.best_individual)
            .unwrap()
    }

    pub fn get_pareto_front(&self) -> Vec<(usize, f32, Individual<D>)> {
        let mut front: Vec<(usize, f32, Individual<D>)> = self.global_hof.iter()
            .map(|(&c, &(mse, ref ind))| (c, mse, ind.clone()))
            .collect();
        front.sort_by_key(|k| k.0);

        let mut pareto = Vec::new();
        let mut best_mse = f32::MAX;

        for (comp, mse, ind) in front {
            if mse < best_mse {
                best_mse = mse;
                pareto.push((comp, mse, ind));
            }
        }
        pareto
    }
}