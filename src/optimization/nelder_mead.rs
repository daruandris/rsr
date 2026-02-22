use crate::engine::individual::Individual;
use crate::metrics::dataset::SimdDataset;
use crate::domain::Domain;

pub fn optimize_individual_constants<D: Domain>(
    ind: &mut Individual<D>, 
    dataset: &SimdDataset, 
    max_iterations: usize
) {
    if ind.program.is_none() { ind.compile(); }
    let program = match &mut ind.program { Some(p) => p, None => return, };

    // 1. Kigyűjtjük CSAK a Float konstansokat fix méretű, stack-en tárolt tömbökbe (ZERO HEAP!)
    let mut start_consts = [0.0; 32];
    let mut opt_indices = [0usize; 32];
    let mut n = 0;
    
    for (i, c) in program.constants.iter().enumerate() {
        if let Some(f) = D::scalar_to_f32(c) {
            if n < 32 {
                start_consts[n] = f;
                opt_indices[n] = i;
                n += 1;
            }
        }
    }
    
    if n == 0 { return; } // Nincs mit optimalizálni

    const ALPHA: f32 = 1.0; const GAMMA: f32 = 2.0; const RHO: f32 = 0.5; const SIGMA: f32 = 0.5;
    
    // A Simplex pontok is fix tömbben, memóriafoglalás nélkül!
    let mut simplex = [(0.0f32, [0.0f32; 32]); 33];
    
    // Zero-cost belső mutáló lambdánk
    let update_constants = |prog_consts: &mut Vec<D::ScalarValue>, vals: &[f32; 32]| {
        for j in 0..n {
            prog_consts[opt_indices[j]] = D::scalar_from_f32(vals[j]);
        }
    };

    let start_mse = D::compute_mse(&program.code, &program.constants, dataset);
    simplex[0] = (start_mse, start_consts);

    for i in 0..n {
        let mut new_point = start_consts;
        let val = new_point[i];
        new_point[i] += if val.abs() < 1e-4 { 0.01 } else { val * 0.10 };
        
        update_constants(&mut program.constants, &new_point);
        let mse = D::compute_mse(&program.code, &program.constants, dataset);
        simplex[i + 1] = (mse, new_point);
    }

    let mut centroid = [0.0; 32]; let mut reflected = [0.0; 32];
    let mut expanded = [0.0; 32]; let mut contracted = [0.0; 32];

    for _ in 0..max_iterations {
        // Csak a használt (n+1) elemet rendezzük
        simplex[0..=n].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let best_mse = simplex[0].0;
        let worst_mse = simplex[n].0;

        if (worst_mse - best_mse).abs() < 1e-7 || best_mse < 1e-8 { break; }
        
        centroid.fill(0.0);
        for i in 0..n {
            for j in 0..n { centroid[j] += simplex[i].1[j]; }
        }
        for j in 0..n { centroid[j] /= n as f32; }

        let worst_point = simplex[n].1;
        let second_worst_mse = simplex[n - 1].0;

        // --- REFLECTION ---
        for j in 0..n { reflected[j] = centroid[j] + ALPHA * (centroid[j] - worst_point[j]); }
        update_constants(&mut program.constants, &reflected);
        let reflected_mse = D::compute_mse(&program.code, &program.constants, dataset);

        if reflected_mse >= best_mse && reflected_mse < second_worst_mse {
            simplex[n] = (reflected_mse, reflected);
            continue;
        }

        // --- EXPANSION ---
        if reflected_mse < best_mse {
            for j in 0..n { expanded[j] = centroid[j] + GAMMA * (reflected[j] - centroid[j]); }
            update_constants(&mut program.constants, &expanded);
            let expanded_mse = D::compute_mse(&program.code, &program.constants, dataset);
            
            if expanded_mse < reflected_mse { simplex[n] = (expanded_mse, expanded); } 
            else { simplex[n] = (reflected_mse, reflected); }
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

        update_constants(&mut program.constants, &contracted);
        let contracted_mse = D::compute_mse(&program.code, &program.constants, dataset);

        if contracted_mse < limit_mse {
            simplex[n] = (contracted_mse, contracted);
            continue;
        }

        // --- SHRINK ---
        let best_point = simplex[0].1;
        for i in 1..=n {
            for j in 0..n { simplex[i].1[j] = best_point[j] + SIGMA * (simplex[i].1[j] - best_point[j]); }
            update_constants(&mut program.constants, &simplex[i].1);
            simplex[i].0 = D::compute_mse(&program.code, &program.constants, dataset);
        }
    }

    simplex[0..=n].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    update_constants(&mut program.constants, &simplex[0].1);
    
    let best_consts = &program.constants;
    let mut const_idx = 0;
    
    for node in ind.nodes.iter_mut() {
        if let crate::ast::node::Node::Constant(val, _) = node {
            if const_idx < best_consts.len() {
                *val = best_consts[const_idx];
                const_idx += 1;
            }
        }
    }
    
    ind.fitness = simplex[0].0;
}