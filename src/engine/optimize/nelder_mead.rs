use crate::engine::data::dataset::Dataset;
use crate::engine::eval::evaluator;
use crate::engine::eval::scalar::Scalar;
use crate::engine::expr::program::Program;
use crate::engine::search::individual::Individual;

const L1_REG_LAMBDA: f32 = 0.01;

pub fn run_nelder_mead(ind: &mut Individual, dataset: &Dataset, max_iterations: usize) {
    if ind.program.is_none() {
        ind.compile();
    }
    let program = match &mut ind.program {
        Some(p) => p,
        None => return,
    };

    let mut start_consts = [0.0; 32];
    let mut opt_indices = [0usize; 32];
    let mut n = 0;

    for (i, c) in program.constants.iter().enumerate() {
        if let Scalar::Float(f) = c {
            if n < 32 {
                start_consts[n] = *f;
                opt_indices[n] = i;
                n += 1;
            }
        }
    }

    if n == 0 {
        return;
    }

    const ALPHA: f32 = 1.0;
    const GAMMA: f32 = 2.0;
    const RHO: f32 = 0.5;
    const SIGMA: f32 = 0.5;

    let mut simplex = [(0.0f32, 0.0f32, [0.0f32; 32]); 33];
    let evaluate = |prog: &mut Program, vals: &[f32; 32]| -> (f32, f32) {
        for j in 0..n {
            prog.constants[opt_indices[j]] = Scalar::Float(vals[j]);
        }
        let mse = evaluator::compute_mse(prog, dataset);
        let mut l1 = 0.0;
        for j in 0..n {
            l1 += vals[j].abs();
        }
        (mse + L1_REG_LAMBDA * l1, mse)
    };

    let (start_fit, start_mse) = evaluate(program, &start_consts);
    simplex[0] = (start_fit, start_mse, start_consts);

    for i in 0..n {
        let mut new_point = start_consts;
        let val = new_point[i];
        new_point[i] += if val.abs() < 1e-4 { 0.01 } else { val * 0.10 };
        let (fit, mse) = evaluate(program, &new_point);
        simplex[i + 1] = (fit, mse, new_point);
    }

    let mut centroid = [0.0; 32];
    let mut reflected = [0.0; 32];
    let mut expanded = [0.0; 32];
    let mut contracted = [0.0; 32];

    for _ in 0..max_iterations {
        simplex[0..=n]
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let best_fit = simplex[0].0;
        let best_mse = simplex[0].1;
        let worst_fit = simplex[n].0;

        if (worst_fit - best_fit).abs() < 1e-7 || best_mse < 1e-8 {
            break;
        }

        centroid.fill(0.0);
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i].2[j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f32;
        }

        let worst_point = simplex[n].2;
        let second_worst_fit = simplex[n - 1].0;

        for j in 0..n {
            reflected[j] = centroid[j] + ALPHA * (centroid[j] - worst_point[j]);
        }
        let (reflected_fit, reflected_mse) = evaluate(program, &reflected);

        if reflected_fit >= best_fit && reflected_fit < second_worst_fit {
            simplex[n] = (reflected_fit, reflected_mse, reflected);
            continue;
        }

        if reflected_fit < best_fit {
            for j in 0..n {
                expanded[j] = centroid[j] + GAMMA * (reflected[j] - centroid[j]);
            }
            let (expanded_fit, expanded_mse) = evaluate(program, &expanded);

            if expanded_fit < reflected_fit {
                simplex[n] = (expanded_fit, expanded_mse, expanded);
            } else {
                simplex[n] = (reflected_fit, reflected_mse, reflected);
            }
            continue;
        }

        let limit_fit = if reflected_fit < worst_fit {
            for j in 0..n {
                contracted[j] = centroid[j] + RHO * (reflected[j] - centroid[j]);
            }
            reflected_fit
        } else {
            for j in 0..n {
                contracted[j] = centroid[j] + RHO * (worst_point[j] - centroid[j]);
            }
            worst_fit
        };

        let (contracted_fit, contracted_mse) = evaluate(program, &contracted);

        if contracted_fit < limit_fit {
            simplex[n] = (contracted_fit, contracted_mse, contracted);
            continue;
        }

        let best_point = simplex[0].2;
        for i in 1..=n {
            for j in 0..n {
                simplex[i].2[j] = best_point[j] + SIGMA * (simplex[i].2[j] - best_point[j]);
            }
            let (fit, mse) = evaluate(program, &simplex[i].2);
            simplex[i].0 = fit;
            simplex[i].1 = mse;
        }
    }

    simplex[0..=n]
        .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    for j in 0..n {
        program.constants[opt_indices[j]] = Scalar::Float(simplex[0].2[j]);
    }

    let best_consts = &program.constants;
    let mut const_idx = 0;

    for node in ind.nodes.iter_mut() {
        if let crate::engine::expr::node::Node::Constant(val, _) = node {
            if const_idx < best_consts.len() {
                *val = best_consts[const_idx];
                const_idx += 1;
            }
        }
    }
    ind.fitness = simplex[0].1;
}
