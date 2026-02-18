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
    // 1. Kiválasztunk 'k' véletlenszerű versenyzőt
    let mut candidates = Vec::with_capacity(k);
    for _ in 0..k {
        candidates.push(&population[rng.random_range(0..population.len())]);
    }

    // 2. Keressük a "nem dominált" egyedeket a csoportban
    let mut best_candidate = candidates[0];
    
    // Egyszerűsített logika: végigmegyünk a jelölteken, és keressük a legjobbat
    // Ha Pareto összehasonlítást végzünk, a dominancia a döntő.
    for i in 1..k {
        let challenger = candidates[i];
        
        if dominates(challenger, best_candidate) {
            // Ha a kihívó dominálja a jelenlegi legjobbat (fiatalabb és/vagy fittebb), ő nyer
            best_candidate = challenger;
        } else if !dominates(best_candidate, challenger) {
            // Ha egyik sem dominálja a másikat (pl. egyik fittebb, másik fiatalabb),
            // akkor a Fitnesz dönt (hagyományos módon)
            if challenger.fitness < best_candidate.fitness {
                best_candidate = challenger;
            }
        }
    }

    best_candidate
}