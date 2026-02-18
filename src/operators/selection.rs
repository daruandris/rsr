use crate::evolution::individual::Individual;
use rand::RngExt;

fn dominates(a: &Individual, b: &Individual) -> bool {
    (a.fitness <= b.fitness && a.age <= b.age) && 
    (a.fitness < b.fitness || a.age < b.age)
}

pub fn tournament_selection_pareto<'a>(
    population: &'a [Individual],
    k: usize,
    rng: &mut impl RngExt
) -> &'a Individual {
    let best_idx = rng.random_range(0..population.len());
    let mut best_ind = &population[best_idx];

    for _ in 1..k {
        let challenger_idx = rng.random_range(0..population.len());
        let challenger = &population[challenger_idx];

        if dominates(challenger, best_ind) {
            best_ind = challenger;
        } else if !dominates(best_ind, challenger) {
            if challenger.fitness < best_ind.fitness {
                best_ind = challenger;
            }
        }
    }

    best_ind
}