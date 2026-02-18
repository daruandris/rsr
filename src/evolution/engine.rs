use rayon::prelude::*;
use std::collections::HashMap;

use crate::evolution::config::EvolutionConfig;
use crate::evolution::individual::Individual;
use crate::evolution::island::Island;
use crate::metrics::dataset::SimdDataset;
use crate::ast::format::format_real_equation;

pub struct Engine {
    pub islands: Vec<Island>,
    pub config: EvolutionConfig,
    pub global_hof: HashMap<usize, (f32, Individual)>,
}

impl Engine {
    pub fn new(config: EvolutionConfig, num_features: u8) -> Self {
        let mut islands = Vec::with_capacity(config.num_islands);
        for i in 0..config.num_islands {
            islands.push(Island::new(config.island_size, 42 + i as u64, num_features));
        }
        Self { islands, config, global_hof: HashMap::new() }
    }

    pub fn run_evolution(&mut self, dataset: &SimdDataset) {
        let config = self.config.clone();
        for generation in 0..config.max_generations {
            self.islands.par_iter_mut().for_each(|island| {
                island.step_generation(dataset, &config);
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

            let mut global_best = self.get_global_best().clone();
            let pure_mse = global_best.calculate_mse(dataset);

            if pure_mse <= config.target_mse {
                if config.verbose {
                    println!("\n>>> Targert reached in {}. genereation! <<<", generation);
                }
                break;
            }

            if generation > 0 && generation % config.migration_interval == 0 {
                self.migrate_individuals();
                if self.config.verbose {
                    println!("Generation: {}, Best MSE: {}\n Expression: {}", generation, pure_mse, self.get_global_best());
                }
            }
        }
        if config.verbose {
            println!("\n>>> Starting Final Hard Optimization ({} iterations) <<<", config.final_opt_iterations);
        }
        let mut final_best = self.get_global_best().clone();
        final_best.optimize_constants(dataset, config.final_opt_iterations);

        if config.verbose {
            println!("--------------------------------------------------");
            println!("FINAL RESULT AFTER OPTIMIZATION:");

            let norm_mse = final_best.calculate_mse(dataset);
            println!("Normalized MSE: {:.8}", norm_mse);

            let real_mse = norm_mse * (dataset.target_std_dev * dataset.target_std_dev);
            println!("Real-world MSE: {:.8}", real_mse);

            println!("Final Expression (Real): {}", format_real_equation(&final_best, dataset));
            println!("--------------------------------------------------");
        }
        
    }

    fn migrate_individuals(&mut self) {
        let num_islands = self.islands.len();
        if num_islands < 2 { return; }
        let mut migrants: Vec<Individual> = self.islands.iter()
            .map(|island| island.best_individual.clone())
            .collect();
        migrants.rotate_right(1);

        for (island, migrant) in self.islands.iter_mut().zip(migrants.into_iter()) {
            let last_idx = island.individuals.len() - 1;
            island.individuals[last_idx] = migrant;
        }
    }

    pub fn get_global_best(&self) -> &Individual {
        self.islands.iter()
            .min_by(|a, b| a.best_individual.fitness.partial_cmp(&b.best_individual.fitness).unwrap())
            .map(|island| &island.best_individual)
            .unwrap()
    }

    pub fn get_pareto_front(&self) -> Vec<(usize, f32, Individual)> {
        let mut front: Vec<(usize, f32, Individual)> = self.global_hof.iter()
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