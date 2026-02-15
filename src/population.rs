use rand::RngExt;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::individual::{Individual};
use crate::operators::*;

#[derive(Clone, Copy, Debug)]
pub struct EvolutionConfig {
    pub num_islands: usize,
    pub island_size: usize,
    pub max_generations: usize,
    pub crossover_rate: f64,
    pub tournament_size: usize,
    pub migration_interval: usize,
    pub parsimony_penalty: f64,

    pub opt_prob: f64,
    pub opt_iterations: usize,
    pub opt_lr: f64,
    pub opt_epsilon: f64,

    pub stagnation_threshold: usize,
    pub target_mse: f64,
}

pub struct Island {
    pub individuals: Vec<Individual>,
    pub best_individual: Individual,
    pub rng: Xoshiro256PlusPlus,
    pub stagnation_counter: usize,
}

impl Island {
    pub fn new(size: usize, seed: u64, num_features: usize) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut individuals = Vec::with_capacity(size);
        for _ in 0..size {
            let ast = generate_random_ast(5, &mut rng, num_features);
            individuals.push(Individual::new(ast));
        }

        let best_individual = individuals[0].clone();

        Self {
            individuals,
            best_individual,
            rng,
            stagnation_counter: 0,
        }
    }

    pub fn step_generation(&mut self, data_x: &[Vec<f64>], data_y: &[f64], config: &EvolutionConfig) {
        let num_features = if data_x.is_empty() {1} else {data_x[0].len()};
        let pop_size = self.individuals.capacity();
        let mut next_gen = Vec::with_capacity(pop_size);
        next_gen.push(self.best_individual.clone());
        while next_gen.len() < pop_size {
            let p: f64 = self.rng.random();
            if p < config.crossover_rate {
                let parent1 = tournament_selection(&self.individuals, config.tournament_size, &mut self.rng);
                let parent2 = tournament_selection(&self.individuals, config.tournament_size, &mut self.rng);

                let child = crossover(parent1, parent2, &mut self.rng);
                next_gen.push(child);
            }
            else{
                let mut child = tournament_selection(&self.individuals, config.tournament_size, &mut self.rng).clone();
                let mut_type = self.rng.random_range(0..3);
                
                match mut_type{
                    0 => point_mutation(&mut child, &mut self.rng, num_features),
                    1 => constant_perturbation(&mut child, &mut self.rng),
                    _ => subtree_mutation(&mut child, &mut self.rng, num_features),
                }
                next_gen.push(child);
            }
        }

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
            let complexity_penalty = (ind.nodes.len() as f64) * config.parsimony_penalty;

            if mse.is_finite() {
                ind.fitness = mse + complexity_penalty;
            }
            else{
                ind.fitness = f64::MAX;
            }

            if ind.fitness < self.best_individual.fitness {
                self.best_individual = ind.clone();
                improved_this_gen = true;
            }
            
        }

        self.individuals = next_gen;

        if improved_this_gen {
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }

        if self.stagnation_counter >= config.stagnation_threshold {
            self.individuals.clear();
            self.individuals.push(self.best_individual.clone());
            
            for _ in 1..pop_size {
                let ast = generate_random_ast(5, &mut self.rng, num_features);
                let mut new_ind = Individual::new(ast);
                
                let mse = new_ind.calculate_mse(data_x, data_y);
                let penalty = (new_ind.nodes.len() as f64) * config.parsimony_penalty;
                if mse.is_finite() {
                    new_ind.fitness = mse + penalty;
                }
                
                self.individuals.push(new_ind);
            }
            
            self.stagnation_counter = 0;
        }
    }
}

pub struct Engine {
    pub islands: Vec<Island>,
    pub config: EvolutionConfig,
}

impl Engine {
    pub fn new(config: EvolutionConfig, num_features: usize) -> Self {
        let mut islands = Vec::with_capacity(config.num_islands);
        for i in 0..config.num_islands{
            islands.push(Island::new(config.island_size, 42 + i as u64, num_features));
        }
        Self { islands, config }
    }

    pub fn run_evolution(&mut self, data_x: &[Vec<f64>], data_y: &[f64]){
        let config = self.config.clone();
        for generation in 0..config.max_generations{
            self.islands.par_iter_mut().for_each(|island| {
                island.step_generation(data_x, data_y, &config);
            });

            let global_best = self.get_global_best();
            let pure_mse = global_best.calculate_mse(data_x, data_y);

            if pure_mse <= config.target_mse {
                println!("\n>>> CÉL ELÉRVE a(z) {}. generációban! <<<", generation);
                println!("Tiszta MSE: {:.8}", pure_mse);
                println!("Egyenlet: {}", global_best);
                break;
            }

            if generation > 0 && generation % config.migration_interval == 0 {
                self.migrate_individuals();
                println!("Generáció: {}, Legjobb MSE: {}\n Egyenlet: {}", generation, pure_mse, self.get_global_best());
            }
        }
    }

    fn migrate_individuals(&mut self) {
        let num_islands = self.islands.len();
        if num_islands < 2 { return; }
        let mut migrants: Vec<Individual> = self.islands.iter()
            .map(|island| island.best_individual.clone())
            .collect();
        migrants.rotate_right(1);

        for (island, migrant) in self.islands.iter_mut().zip(migrants.into_iter()){
            let last_idx = island.individuals.len() -1;
            island.individuals[last_idx] = migrant;
        }
    }

    pub fn get_global_best(&self) -> &Individual {
        self.islands.iter()
            .min_by(|a,b| a.best_individual.fitness.partial_cmp(&b.best_individual.fitness).unwrap())
            .map(|island| &island.best_individual)
            .unwrap()
    }
}
