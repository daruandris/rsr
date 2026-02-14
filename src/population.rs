use rand::RngExt;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::individual::{Individual};
use crate::operators::*;

pub struct Island {
    pub individuals: Vec<Individual>,
    pub best_individual: Individual,
    pub rng: Xoshiro256PlusPlus,
}

impl Island {
    pub fn new(size: usize, seed: u64) -> Self {
        let rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut individuals = Vec::with_capacity(size);
        for _ in 0..size {
            // TODO: ast generálás
            individuals.push(Individual::new(vec![]));
        }

        let best_individual = individuals[0].clone();

        Self {
            individuals,
            best_individual,
            rng
        }
    }

    pub fn step_generation(&mut self, data_x: &[Vec<f64>], data_y: &[f64]) {
        let pop_size = self.individuals.capacity();
        let mut next_gen = Vec::with_capacity(pop_size);
        next_gen.push(self.best_individual.clone());
        while next_gen.len() < pop_size {
            let p: f64 = self.rng.random();
            if p < 0.85 {
                let parent1 = tournament_selection(&self.individuals, 3, &mut self.rng);
                let parent2 = tournament_selection(&self.individuals, 3, &mut self.rng);

                let child = crossover(parent1, parent2, &mut self.rng);
                next_gen.push(child);
            }
            else{
                let child = tournament_selection(&self.individuals, 3, &mut self.rng).clone();
                
                // TODO: mutation
                next_gen.push(child);
            }
        }

        for ind in next_gen.iter_mut() {
            if ind.fitness < self.best_individual.fitness {
                self.best_individual = ind.clone();
            }
        }

        self.individuals = next_gen;
    }
}

pub struct Engine {
    pub islands: Vec<Island>,
}

impl Engine {
    pub fn new(num_islands: usize, island_size: usize) -> Self {
        let mut islands = Vec::with_capacity(num_islands);
        for i in 0..num_islands{
            islands.push(Island::new(island_size, 42 + i as u64));
        }
        Self { islands }
    }

    pub fn run_evolution(&mut self, generations: usize, data_x: &[Vec<f64>], data_y: &[f64]){
        for generation in 0..generations{
            self.islands.par_iter_mut().for_each(|island| {
                island.step_generation(data_x, data_y);
            });

            if generation > 0 && generation % 20 == 0 {
                self.migrate_individuals();
                let global_best = self.get_global_best();
                println!("Generáció: {}, Legjobb MSE: {}", generation, global_best.fitness);
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
