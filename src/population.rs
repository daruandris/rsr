use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::individual::{Individual};

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

    pub fn step_generation(&mut self, data_x: &[Vec<f64>], data_y: &[Vec<f64>]) {
        let pop_size = self.individuals.capacity();
        let mut next_gen = Vec::with_capacity(pop_size);
        next_gen.push(self.best_individual.clone());
        while next_gen.len() < pop_size {
            // TODO: operátorok meghívása
            next_gen.push(self.best_individual.clone());
        }

        for ind in next_gen.iter_mut() {
            if ind.fitness < self.best_individual.fitness {
                self.best_individual = ind.clone();
            }
        }

        self.individuals = next_gen;
    }
}
