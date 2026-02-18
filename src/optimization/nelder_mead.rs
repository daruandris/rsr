use crate::evolution::individual::Individual;
use crate::metrics::dataset::SimdDataset;
use crate::metrics::mse::calculate_mse_simd;

pub fn optimize_individual_constants(
    ind: &mut Individual, 
    dataset: &SimdDataset, 
    max_iterations: usize
) {
    if ind.program.is_none() {
        ind.compile();
    }
    
    let program = match &mut ind.program {
        Some(p) => p,
        None => return,
    };

    let n = program.constants.len();
    if n == 0 { return; }

    // --- HIPERPARAMÉTEREK ---
    const ALPHA: f32 = 1.0;
    const GAMMA: f32 = 2.0;
    const RHO: f32 = 0.5;
    const SIGMA: f32 = 0.5;
    
    let mut simplex: Vec<(f32, Vec<f32>)> = Vec::with_capacity(n + 1);
    let start_consts = program.constants.clone();
    let start_mse = calculate_mse_simd(program, dataset);
    simplex.push((start_mse, start_consts.clone()));

    for i in 0..n {
        let mut new_point = start_consts.clone();
        let val = new_point[i];
        let step = if val.abs() < 1e-4 { 
            0.01 // Ha 0, lépjünk el
        } else { 
            val * 0.10 // 10%-os elmozdulás a 0.5 helyett (bátrabb nyitás)
        };
        new_point[i] += step;
        
        program.constants = new_point.clone(); 
        let mse = calculate_mse_simd(program, dataset);
        simplex.push((mse, new_point));
    }

    let mut centroid = vec![0.0; n];
    let mut reflected = vec![0.0; n];
    let mut expanded = vec![0.0; n];
    let mut contracted = vec![0.0; n];

    for _ in 0..max_iterations {
        simplex.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let best_mse = simplex[0].0;
        let worst_mse = simplex[n].0;

        if (worst_mse - best_mse).abs() < 1e-6 || best_mse < 1e-8 {
            break;
        }
        centroid.fill(0.0);
        
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i].1[j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f32;
        }

        let worst_point = &simplex[n].1;
        let second_worst_mse = simplex[n - 1].0;

        // --- REFLECTION ---
        for j in 0..n {
            reflected[j] = centroid[j] + ALPHA * (centroid[j] - worst_point[j]);
        }
        program.constants.copy_from_slice(&reflected);
        let reflected_mse = calculate_mse_simd(program, dataset);

        if reflected_mse >= best_mse && reflected_mse < second_worst_mse {
            simplex[n] = (reflected_mse, reflected.clone());
            continue;
        }

        // --- EXPANSION ---
        if reflected_mse < best_mse {
            for j in 0..n {
                expanded[j] = centroid[j] + GAMMA * (reflected[j] - centroid[j]);
            }
            
            program.constants.copy_from_slice(&expanded);
            let expanded_mse = calculate_mse_simd(program, dataset);

            if expanded_mse < reflected_mse {
                simplex[n] = (expanded_mse, expanded.clone());
            } else {
                simplex[n] = (reflected_mse, reflected.clone());
            }
            continue;
        }
        
        let limit_mse = if reflected_mse < worst_mse {
            for j in 0..n { contracted[j] = centroid[j] + RHO * (reflected[j] - centroid[j]); }
            reflected_mse
        } else {
            for j in 0..n { contracted[j] = centroid[j] + RHO * (worst_point[j] - centroid[j]); }
            worst_mse
        };

        program.constants.copy_from_slice(&contracted);
        let contracted_mse = calculate_mse_simd(program, dataset);

        if contracted_mse < limit_mse {
            simplex[n] = (contracted_mse, contracted.clone());
            continue;
        }

        // --- SHRINK ---
        let best_point = simplex[0].1.clone();
        for i in 1..=n {
            for j in 0..n {
                simplex[i].1[j] = best_point[j] + SIGMA * (simplex[i].1[j] - best_point[j]);
            }
            
            program.constants.copy_from_slice(&simplex[i].1);
            simplex[i].0 = calculate_mse_simd(program, dataset);
        }
    }

    simplex.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    program.constants = simplex[0].1.clone();
    
    let best_consts = &simplex[0].1;
    let mut const_idx = 0;
    for node in &mut ind.nodes {
        if let crate::ast::node::Node::Constant(val) = node {
            if const_idx < best_consts.len() {
                *val = best_consts[const_idx];
                const_idx += 1;
            }
        }
    }
    ind.fitness = simplex[0].0; 
}