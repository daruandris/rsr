use crate::evolution::individual::Individual;
use rand::RngExt;

fn dominates(a: &Individual, b: &Individual) -> bool {
    let fit_better_eq = a.fitness <= b.fitness;
    let age_better_eq = a.age <= b.age;
    
    let fit_strictly_better = a.fitness < b.fitness;
    let age_strictly_better = a.age < b.age;

    fit_better_eq && age_better_eq && (fit_strictly_better || age_strictly_better)
}

pub fn tournament_selection_pareto<'a>(
    population: &'a [Individual],
    k: usize,
    rng: &mut impl RngExt
) -> &'a Individual {
    let mut candidates = Vec::with_capacity(k);
    for _ in 0..k {
        candidates.push(&population[rng.random_range(0..population.len())]);
    }

    let mut best_candidate = candidates[0];
    
    for i in 1..k {
        let challenger = candidates[i];
        
        if dominates(challenger, best_candidate) {
            best_candidate = challenger;
        } else if !dominates(best_candidate, challenger) {
            if challenger.fitness < best_candidate.fitness {
                best_candidate = challenger;
            }
        }
    }

    best_candidate
}