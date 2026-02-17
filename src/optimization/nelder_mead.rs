use crate::evolution::individual::Individual;
use crate::metrics::dataset::SimdDataset;

pub fn optimize_individual_constants(
    ind: &mut Individual, 
    dataset: &SimdDataset, 
    max_iterations: usize
) {
    let base_consts = ind.get_constants();
    let n = base_consts.len();
    if n == 0 { return; } // Ha nincs konstans, nincs mit optimalizálni

    // Nelder-Mead hiperparaméterek (Standard értékek)
    let alpha = 1.0; // Tükrözés (Reflection)
    let gamma = 2.0; // Kiterjesztés (Expansion)
    let rho = 0.5;   // Összehúzás (Contraction)
    let sigma = 0.5; // Zsugorodás (Shrink)

    // 1. A Szimplex inicializálása (N + 1 pont)
    // Tároló: (MSE_hiba, Konstans_vektor)
    let mut simplex: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n + 1);
    
    // Alappont
    ind.set_constants(&base_consts);
    let base_mse = ind.calculate_mse(dataset);
    simplex.push((base_mse, base_consts.clone()));

    // További N pont létrehozása (mindegyik dimenzióban lépünk egy kicsit)
    for i in 0..n {
        let mut new_point = base_consts.clone();
        // Ha a konstans 0, akkor fix lépés, különben 5% változtatás
        let step = if new_point[i].abs() < 1e-4 { 0.05 } else { new_point[i] * 0.05 };
        new_point[i] += step;
        
        ind.set_constants(&new_point);
        let mse = ind.calculate_mse(dataset);
        simplex.push((mse, new_point));
    }

    // 2. A Fő Optimalizációs Ciklus
    for _ in 0..max_iterations {
        // Rendezzük a szimplexet MSE szerint növekvő sorrendbe (legjobb elöl)
        simplex.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Ha a legjobb és a legrosszabb pont közötti különbség minimális, leállunk
        if (simplex.last().unwrap().0 - simplex.first().unwrap().0).abs() < 1e-6 {
            break;
        }

        // Súlypont (Centroid) kiszámítása a legrosszabb pont kivételével
        let mut centroid = vec![0.0; n];
        for i in 0..n {
            for j in 0..n { // Csak az első N pontot adjuk össze
                centroid[j] += simplex[i].1[j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        let worst_point = &simplex[n].1;
        let worst_mse = simplex[n].0;
        let second_worst_mse = simplex[n - 1].0;
        let best_mse = simplex[0].0;

        // --- TÜKRÖZÉS (Reflection) ---
        let mut reflected = vec![0.0; n];
        for j in 0..n {
            reflected[j] = centroid[j] + alpha * (centroid[j] - worst_point[j]);
        }
        ind.set_constants(&reflected);
        let reflected_mse = ind.calculate_mse(dataset);

        if reflected_mse >= best_mse && reflected_mse < second_worst_mse {
            simplex[n] = (reflected_mse, reflected);
            continue;
        }

        // --- KITERJESZTÉS (Expansion) ---
        if reflected_mse < best_mse {
            let mut expanded = vec![0.0; n];
            for j in 0..n {
                expanded[j] = centroid[j] + gamma * (reflected[j] - centroid[j]);
            }
            ind.set_constants(&expanded);
            let expanded_mse = ind.calculate_mse(dataset);

            if expanded_mse < reflected_mse {
                simplex[n] = (expanded_mse, expanded);
            } else {
                simplex[n] = (reflected_mse, reflected);
            }
            continue;
        }

        // --- ÖSSZEHÚZÁS (Contraction) ---
        let mut contracted = vec![0.0; n];
        
        if reflected_mse < worst_mse {
            // Külső összehúzás
            for j in 0..n { contracted[j] = centroid[j] + rho * (reflected[j] - centroid[j]); }
            ind.set_constants(&contracted);
            let contract_mse = ind.calculate_mse(dataset); // <--- Lokális deklaráció
            if contract_mse <= reflected_mse {
                simplex[n] = (contract_mse, contracted);
                continue;
            }
        } else {
            // Belső összehúzás
            for j in 0..n { contracted[j] = centroid[j] + rho * (worst_point[j] - centroid[j]); }
            ind.set_constants(&contracted);
            let contract_mse = ind.calculate_mse(dataset); // <--- Lokális deklaráció
            if contract_mse < worst_mse {
                simplex[n] = (contract_mse, contracted);
                continue;
            }
        }

        // --- ZSUGORODÁS (Shrink) ---
        // Ha semmi sem jött be, minden pontot a legjobb pont felé húzunk
        let best_point = simplex[0].1.clone();
        for i in 1..=n {
            let mut shrunk = vec![0.0; n];
            for j in 0..n {
                shrunk[j] = best_point[j] + sigma * (simplex[i].1[j] - best_point[j]);
            }
            ind.set_constants(&shrunk);
            simplex[i].0 = ind.calculate_mse(dataset);
            simplex[i].1 = shrunk;
        }
    }

    // A végén a legkisebb hibájú konstanst állítjuk be az egyedre
    simplex.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    ind.set_constants(&simplex[0].1);
}