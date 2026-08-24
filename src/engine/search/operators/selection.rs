use crate::engine::search::individual::Individual;
use rand::RngExt;
use std::cmp::Ordering;

fn dominates(a: &Individual, b: &Individual) -> bool {
    let better_eq_fit = a.fitness <= b.fitness;
    let better_fit = a.fitness < b.fitness;
    let better_eq_age = a.age <= b.age;
    let better_age = a.age < b.age;
    (better_eq_fit && better_eq_age) && (better_fit || better_age)
}

pub fn tournament_selection_pareto<'a>(
    population: &'a [Individual],
    k: usize,
    rng: &mut impl RngExt,
) -> &'a Individual {
    let mut best_idx = rng.random_range(0..population.len());
    for _ in 1..k {
        let challenger_idx = rng.random_range(0..population.len());
        if nsga2_compare(&population[challenger_idx], &population[best_idx]) == Ordering::Less {
            best_idx = challenger_idx;
        }
    }
    &population[best_idx]
}

pub fn nsga2_compare(a: &Individual, b: &Individual) -> Ordering {
    match a.rank.cmp(&b.rank) {
        Ordering::Less => Ordering::Less,
        Ordering::Greater => Ordering::Greater,
        Ordering::Equal => b
            .crowding_distance
            .partial_cmp(&a.crowding_distance)
            .unwrap_or(Ordering::Equal),
    }
}

pub fn assign_rank_and_crowding_distance(pop: &mut [Individual]) {
    let n = pop.len();
    let mut domination_count = vec![0; n];
    let mut dominated_individuals = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    fronts.push(Vec::new());

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if dominates(&pop[i], &pop[j]) {
                dominated_individuals[i].push(j);
            } else if dominates(&pop[j], &pop[i]) {
                domination_count[i] += 1;
            }
        }
        if domination_count[i] == 0 {
            pop[i].rank = 0;
            fronts[0].push(i);
        }
    }

    let mut i = 0;
    while i < fronts.len() {
        let mut next_front = Vec::new();
        for &p_idx in &fronts[i] {
            for &q_idx in &dominated_individuals[p_idx] {
                domination_count[q_idx] -= 1;
                if domination_count[q_idx] == 0 {
                    pop[q_idx].rank = (i + 1) as u32;
                    next_front.push(q_idx);
                }
            }
        }
        if next_front.is_empty() {
            break;
        }
        fronts.push(next_front);
        i += 1;
    }

    for front in fronts {
        if front.is_empty() {
            continue;
        }

        for &idx in &front {
            pop[idx].crowding_distance = 0.0;
        }

        if front.len() < 3 {
            for &idx in &front {
                pop[idx].crowding_distance = f32::MAX;
            }
            continue;
        }

        let mut sorted_front = front.clone();
        sorted_front.sort_by(|&a, &b| pop[a].fitness.partial_cmp(&pop[b].fitness).unwrap());

        pop[sorted_front[0]].crowding_distance = f32::MAX;
        pop[*sorted_front.last().unwrap()].crowding_distance = f32::MAX;

        let mse_range = pop[*sorted_front.last().unwrap()].fitness - pop[sorted_front[0]].fitness;
        if mse_range > 1e-9 {
            for k in 1..sorted_front.len() - 1 {
                let dist = (pop[sorted_front[k + 1]].fitness - pop[sorted_front[k - 1]].fitness)
                    / mse_range;
                if pop[sorted_front[k]].crowding_distance != f32::MAX {
                    pop[sorted_front[k]].crowding_distance += dist;
                }
            }
        }

        sorted_front.sort_by_key(|&a| pop[a].age);
        pop[sorted_front[0]].crowding_distance = f32::MAX;
        pop[*sorted_front.last().unwrap()].crowding_distance = f32::MAX;

        let age_min = pop[sorted_front[0]].age as f32;
        let age_max = pop[*sorted_front.last().unwrap()].age as f32;
        let age_range = age_max - age_min;

        if age_range > 1e-9 {
            for k in 1..sorted_front.len() - 1 {
                let a_next = pop[sorted_front[k + 1]].age as f32;
                let a_prev = pop[sorted_front[k - 1]].age as f32;
                let dist = (a_next - a_prev) / age_range;
                if pop[sorted_front[k]].crowding_distance != f32::MAX {
                    pop[sorted_front[k]].crowding_distance += dist;
                }
            }
        }
    }
}
