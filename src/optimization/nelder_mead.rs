use crate::engine::individual::Individual;
use crate::metrics::dataset::SimdDataset;
use crate::domain::Domain;

pub fn optimize_individual_constants<D: Domain>(
    ind: &mut Individual<D>, 
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

    const ALPHA: f32 = 1.0; const GAMMA: f32 = 2.0; const RHO: f32 = 0.5; const SIGMA: f32 = 0.5;
    
    let mut simplex: Vec<(f32, Vec<f32>)> = Vec::with_capacity(n + 1);
    
    // Konvertáljuk a generikus konstansokat f32-be a matekhoz!
    let start_consts_f32: Vec<f32> = program.constants.iter().map(|c| D::scalar_to_f32(c)).collect();
    let start_mse = D::compute_mse(&program.code, &program.constants, dataset);
    simplex.push((start_mse, start_consts_f32.clone()));

    for i in 0..n {
        let mut new_point = start_consts_f32.clone();
        let val = new_point[i];
        let step = if val.abs() < 1e-4 { 0.01 } else { val * 0.10 };
        new_point[i] += step;
        
        program.constants = new_point.iter().map(|&x| D::scalar_from_f32(x)).collect(); 
        let mse = D::compute_mse(&program.code, &program.constants, dataset);
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

        if (worst_mse - best_mse).abs() < 1e-7 || best_mse < 1e-8 { break; }
        centroid.fill(0.0);
        
        for i in 0..n {
            for j in 0..n { centroid[j] += simplex[i].1[j]; }
        }
        for j in 0..n { centroid[j] /= n as f32; }

        let worst_point = &simplex[n].1;
        let second_worst_mse = simplex[n - 1].0;

        // --- REFLECTION ---
        for j in 0..n { reflected[j] = centroid[j] + ALPHA * (centroid[j] - worst_point[j]); }
        program.constants = reflected.iter().map(|&x| D::scalar_from_f32(x)).collect();
        let reflected_mse = D::compute_mse(&program.code, &program.constants, dataset);

        if reflected_mse >= best_mse && reflected_mse < second_worst_mse {
            simplex[n] = (reflected_mse, reflected.clone());
            continue;
        }

        // --- EXPANSION ---
        if reflected_mse < best_mse {
            for j in 0..n { expanded[j] = centroid[j] + GAMMA * (reflected[j] - centroid[j]); }
            program.constants = expanded.iter().map(|&x| D::scalar_from_f32(x)).collect();
            let expanded_mse = D::compute_mse(&program.code, &program.constants, dataset);

            if expanded_mse < reflected_mse { simplex[n] = (expanded_mse, expanded.clone()); } 
            else { simplex[n] = (reflected_mse, reflected.clone()); }
            continue;
        }
        
        // --- CONTRACTION ---
        let limit_mse = if reflected_mse < worst_mse {
            for j in 0..n { contracted[j] = centroid[j] + RHO * (reflected[j] - centroid[j]); }
            reflected_mse
        } else {
            for j in 0..n { contracted[j] = centroid[j] + RHO * (worst_point[j] - centroid[j]); }
            worst_mse
        };

        program.constants = contracted.iter().map(|&x| D::scalar_from_f32(x)).collect();
        let contracted_mse = D::compute_mse(&program.code, &program.constants, dataset);

        if contracted_mse < limit_mse {
            simplex[n] = (contracted_mse, contracted.clone());
            continue;
        }

        // --- SHRINK ---
        let best_point = simplex[0].1.clone();
        for i in 1..=n {
            for j in 0..n { simplex[i].1[j] = best_point[j] + SIGMA * (simplex[i].1[j] - best_point[j]); }
            program.constants = simplex[i].1.iter().map(|&x| D::scalar_from_f32(x)).collect();
            simplex[i].0 = D::compute_mse(&program.code, &program.constants, dataset);
        }
    }

    simplex.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    program.constants = simplex[0].1.iter().map(|&x| D::scalar_from_f32(x)).collect();
    
    let best_consts = &program.constants;
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