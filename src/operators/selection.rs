use crate::evolution::individual::Individual;
use rand::RngExt;

pub fn tournament_selection<'a>(population: &'a [Individual], k: usize, rng: &mut impl RngExt) -> &'a Individual {
    let mut best = &population[rng.random_range(0..population.len())];
    for _ in 0..k {
        let contender = &population[rng.random_range(0..population.len())];
        if contender.fitness < best.fitness {
            best = contender;
        }
    }
    best
}