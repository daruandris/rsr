use crate::engine::individual::Individual;
use crate::metrics::dataset::SimdDataset;
use crate::domain::Domain;
use crate::domain::universal::UniversalScalar;
use rand::RngExt;

const MAX_PARAMS: usize = 32;
const LAMBDA: usize = 16;  // Populáció méret (CMA-ES szabvány)
const MU: usize = LAMBDA / 2; // Kiválasztott szülők száma

// Box-Muller transzformáció (zero-dependency normál eloszlás generáláshoz)
fn rand_normal(rng: &mut impl RngExt) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-8);
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

pub fn run_cma_es<D: Domain<ScalarValue = UniversalScalar>>(
    ind: &mut Individual<D>, 
    dataset: &SimdDataset, 
    max_iterations: usize
) {
    let program = match ind.program.as_mut() { Some(p) => p, None => return, };

    // 1. KIGYŰJTÉS (Laposítás)
    let mut mean = [0.0f32; MAX_PARAMS];
    let mut n = 0;
    
    for c in &program.constants {
        match c {
            UniversalScalar::Float(f) => { if n < MAX_PARAMS { mean[n] = *f; n += 1; } },
            UniversalScalar::Vec2(v) => { for i in 0..2 { if n < MAX_PARAMS { mean[n] = v[i]; n += 1; } } },
            UniversalScalar::Vec3(v) => { for i in 0..3 { if n < MAX_PARAMS { mean[n] = v[i]; n += 1; } } },
            UniversalScalar::Mat2(m) => { for i in 0..4 { if n < MAX_PARAMS { mean[n] = m[i]; n += 1; } } },
            UniversalScalar::Mat3(m) => { for i in 0..9 { if n < MAX_PARAMS { mean[n] = m[i]; n += 1; } } },
            _ => {}
        }
    }

    if n == 0 { return; }

    // 2. VISSZACSOMAGOLÓ CLOSURE (Zero-cost, SIMD-kompatibilis)
    let update_constants = |prog_consts: &mut Vec<UniversalScalar>, flat_vals: &[f32; MAX_PARAMS]| {
        let mut ptr = 0;
        for c in prog_consts.iter_mut() {
            match c {
                UniversalScalar::Float(f) => { if ptr < n { *f = flat_vals[ptr]; ptr += 1; } },
                UniversalScalar::Vec2(v) => { for i in 0..2 { if ptr < n { v[i] = flat_vals[ptr]; ptr += 1; } } },
                UniversalScalar::Vec3(v) => { for i in 0..3 { if ptr < n { v[i] = flat_vals[ptr]; ptr += 1; } } },
                UniversalScalar::Mat2(m) => { for i in 0..4 { if ptr < n { m[i] = flat_vals[ptr]; ptr += 1; } } },
                UniversalScalar::Mat3(m) => { for i in 0..9 { if ptr < n { m[i] = flat_vals[ptr]; ptr += 1; } } },
                _ => {}
            }
        }
    };

    // --- SEP-CMA-ES (Diagonal) WORKSPACE --- (Teljesen Stack-en!)
    let mut sigma = 0.5f32; // Kezdeti lépésköz
    let mut c_diag = [1.0f32; MAX_PARAMS]; // Kovariancia mátrix (diagonális)
    let mut p_c = [0.0f32; MAX_PARAMS]; // Evolúciós útvonal
    
    // Súlyok generálása a MU legjobb kiválasztásához
    let mut weights = [0.0f32; MU];
    let mut sum_weights = 0.0;
    for i in 0..MU {
        weights[i] = ((MU as f32 + 0.5).ln() - ((i + 1) as f32).ln()).max(0.0);
        sum_weights += weights[i];
    }
    let mut mu_eff = 0.0;
    for i in 0..MU {
        weights[i] /= sum_weights;
        mu_eff += weights[i] * weights[i];
    }
    mu_eff = 1.0 / mu_eff;

    let c_c = (4.0 + mu_eff / (n as f32)) / ((n as f32) + 4.0 + 2.0 * mu_eff / (n as f32));
    let c_cov = (1.0 / mu_eff) * (2.0 / ((n as f32) + 1.414).powi(2)) 
              + (1.0 - 1.0 / mu_eff) * ((2.0 * mu_eff - 1.0) / (((n as f32) + 2.0).powi(2) + mu_eff));

    let mut best_overall_point = mean;
    let mut best_overall_mse = D::compute_mse(&program.code, &program.constants, dataset);

    let mut rng = rand::rng();
    let mut population = [[0.0f32; MAX_PARAMS]; LAMBDA];
    let mut step_vectors = [[0.0f32; MAX_PARAMS]; LAMBDA];
    let mut pop_fitness = [(0.0f32, 0usize); LAMBDA];

    // --- OPTIMALIZÁCIÓS CIKLUS ---
    for _ in 0..max_iterations {
        // A. Minta vételezés a normál eloszlásból (SIMD kiértékeléssel)
        for i in 0..LAMBDA {
            for j in 0..n {
                let step = rand_normal(&mut rng) * c_diag[j].sqrt();
                step_vectors[i][j] = step;
                population[i][j] = mean[j] + sigma * step;
            }
            
            // SIMD-gyorsított MSE kalkuláció a módosított konstansokkal
            update_constants(&mut program.constants, &population[i]);
            let mse = D::compute_mse(&program.code, &program.constants, dataset);
            pop_fitness[i] = (mse, i);
        }

        // B. Szortírozás Fitness (MSE) szerint
        pop_fitness.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let current_best_mse = pop_fitness[0].0;
        if current_best_mse < best_overall_mse {
            best_overall_mse = current_best_mse;
            best_overall_point = population[pop_fitness[0].1];
        }

        if best_overall_mse < 1e-8 { break; } // Korai kilépés

        // C. Mean frissítés
        let mut step_mean = [0.0f32; MAX_PARAMS];
        let old_mean = mean;
        for i in 0..MU {
            let idx = pop_fitness[i].1;
            for j in 0..n {
                step_mean[j] += weights[i] * step_vectors[idx][j];
                mean[j] += weights[i] * (population[idx][j] - old_mean[j]);
            }
        }

        // D. Evolúciós útvonalak és Diagonális Kovariancia frissítése
        for j in 0..n {
            p_c[j] = (1.0 - c_c) * p_c[j] + (c_c * (2.0 - c_c) * mu_eff).sqrt() * step_mean[j];
            
            let mut cov_update = 0.0;
            for i in 0..MU {
                let idx = pop_fitness[i].1;
                cov_update += weights[i] * (step_vectors[idx][j] * step_vectors[idx][j]);
            }
            
            c_diag[j] = (1.0 - c_cov) * c_diag[j] 
                      + (c_cov / mu_eff) * (p_c[j] * p_c[j]) 
                      + c_cov * (1.0 - 1.0 / mu_eff) * cov_update;
            
            // Limitáljuk, hogy ne szálljon el a mátrix
            c_diag[j] = c_diag[j].clamp(1e-6, 1e6);
        }
        
        // Sep-CMA esetén a sigma frissítése sokszor egyszerűsített
        sigma *= (-0.01_f32).exp(); // Finom annealing, mivel a P_sigma utat most a sebesség miatt lehagytuk
    }

    // 3. LEGJOBB EREDMÉNY SZINKRONIZÁLÁSA
    update_constants(&mut program.constants, &best_overall_point);
    ind.fitness = best_overall_mse;
    
    // Szinkronizálás az AST struktúrával (Hogy a String printelés jó számokat mutasson)
    let mut const_idx = 0;
    for node in ind.nodes.iter_mut() {
        if let crate::ast::node::Node::Constant(val, _) = node {
            if const_idx < program.constants.len() {
                *val = program.constants[const_idx];
                const_idx += 1;
            }
        }
    }
}