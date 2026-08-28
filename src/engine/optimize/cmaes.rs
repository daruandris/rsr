use crate::engine::data::dataset::Dataset;
use crate::engine::eval::evaluator;
use crate::engine::eval::scalar::Scalar;
use crate::engine::optimize::Parameterized;
use crate::engine::search::individual::Individual;
use rand::RngExt;

const MAX_PARAMS: usize = 32;
const LAMBDA: usize = 16;
const MU: usize = LAMBDA / 2;
const L1_REG_LAMBDA: f32 = 0.01;

fn rand_normal(rng: &mut impl RngExt) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-8);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

pub fn run_cma_es(ind: &mut Individual, dataset: &Dataset, max_iterations: usize) {
    let program = match ind.program.as_mut() {
        Some(p) => p,
        None => return,
    };

    let mut n = program.param_count().min(MAX_PARAMS);
    if n == 0 {
        return;
    }

    let mut mean = [0.0f32; MAX_PARAMS];
    program.flatten_params(&mut mean);

    for c in &program.constants {
        match c {
            Scalar::Float(f) => {
                if n < MAX_PARAMS {
                    mean[n] = *f;
                    n += 1;
                }
            }
            Scalar::Vec2(v) => {
                for item in v.iter().take(2) {
                    if n < MAX_PARAMS {
                        mean[n] = *item;
                        n += 1;
                    }
                }
            }
            Scalar::Vec3(v) => {
                for item in v.iter().take(3) {
                    if n < MAX_PARAMS {
                        mean[n] = *item;
                        n += 1;
                    }
                }
            }
            Scalar::Mat2(m) => {
                for item in m.iter().take(4) {
                    if n < MAX_PARAMS {
                        mean[n] = *item;
                        n += 1;
                    }
                }
            }
            Scalar::Mat3(m) => {
                for item in m.iter().take(9) {
                    if n < MAX_PARAMS {
                        mean[n] = *item;
                        n += 1;
                    }
                }
            }
            _ => {}
        }
    }

    if n == 0 {
        return;
    }

    let mut sigma = 0.5f32;
    let mut c_diag = [1.0f32; MAX_PARAMS];
    let mut p_c = [0.0f32; MAX_PARAMS];

    let mut weights = [0.0f32; MU];
    let mut sum_weights = 0.0;

    for (i, weight) in weights.iter_mut().enumerate().take(MU) {
        *weight = ((MU as f32 + 0.5).ln() - ((i + 1) as f32).ln()).max(0.0);
        sum_weights += *weight;
    }

    let mut mu_eff = 0.0;
    for weight in weights.iter_mut().take(MU) {
        *weight /= sum_weights;
        mu_eff += *weight * *weight;
    }
    mu_eff = 1.0 / mu_eff;

    let c_c = (4.0 + mu_eff / (n as f32)) / ((n as f32) + 4.0 + 2.0 * mu_eff / (n as f32));
    let c_cov = (1.0 / mu_eff) * (2.0 / ((n as f32) + 1.414).powi(2))
        + (1.0 - 1.0 / mu_eff) * ((2.0 * mu_eff - 1.0) / (((n as f32) + 2.0).powi(2) + mu_eff));

    let initial_mse = evaluator::compute_mse(program, dataset);

    let mut initial_l1 = 0.0f32;
    for m in mean.iter().take(n) {
        initial_l1 += m.abs();
    }

    let mut best_overall_point = mean;
    let mut best_overall_mse = initial_mse;
    let mut best_overall_fitness = initial_mse + L1_REG_LAMBDA * initial_l1;

    let mut rng = rand::rng();
    let mut population = [[0.0f32; MAX_PARAMS]; LAMBDA];
    let mut step_vectors = [[0.0f32; MAX_PARAMS]; LAMBDA];
    let mut pop_fitness = [(0.0f32, 0.0f32, 0usize); LAMBDA];

    for _ in 0..max_iterations {
        for i in 0..LAMBDA {
            let mut l1_norm = 0.0f32;
            for j in 0..n {
                let step = rand_normal(&mut rng) * c_diag[j].sqrt();
                step_vectors[i][j] = step;
                let val = mean[j] + sigma * step;
                population[i][j] = val;
                l1_norm += val.abs();
            }

            program.unflatten_params(&population[i]);
            let mse = evaluator::compute_mse(program, dataset);
            let fitness = mse + L1_REG_LAMBDA * l1_norm;
            pop_fitness[i] = (fitness, mse, i);
        }

        pop_fitness
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let current_best_fitness = pop_fitness[0].0;
        let current_best_mse = pop_fitness[0].1;
        let best_idx = pop_fitness[0].2;

        if current_best_fitness < best_overall_fitness {
            best_overall_fitness = current_best_fitness;
            best_overall_mse = current_best_mse;
            best_overall_point = population[best_idx];
        }

        if best_overall_mse < 1e-8 {
            break;
        }

        let mut step_mean = [0.0f32; MAX_PARAMS];
        let old_mean = mean;
        for i in 0..MU {
            let idx = pop_fitness[i].2;
            for j in 0..n {
                step_mean[j] += weights[i] * step_vectors[idx][j];
                mean[j] += weights[i] * (population[idx][j] - old_mean[j]);
            }
        }

        for j in 0..n {
            p_c[j] = (1.0 - c_c) * p_c[j] + (c_c * (2.0 - c_c) * mu_eff).sqrt() * step_mean[j];
            let mut cov_update = 0.0;
            for i in 0..MU {
                let idx = pop_fitness[i].2;
                cov_update += weights[i] * (step_vectors[idx][j] * step_vectors[idx][j]);
            }

            c_diag[j] = (1.0 - c_cov) * c_diag[j]
                + (c_cov / mu_eff) * (p_c[j] * p_c[j])
                + c_cov * (1.0 - 1.0 / mu_eff) * cov_update;
            c_diag[j] = c_diag[j].clamp(1e-6, 1e6);
        }

        sigma *= (-0.01_f32).exp();
    }

    program.unflatten_params(&best_overall_point);
    ind.fitness = best_overall_mse;
    let mut const_idx = 0;
    for node in ind.nodes.iter() {
        if let crate::engine::expr::node::Node::Constant(idx, _) = node
            && const_idx < program.constants.len() {
                ind.constants[*idx as usize] = program.constants[const_idx];
                const_idx += 1;
            }
    }
}
