use rsr::prelude::*;
use rsr::ffi::symengine::simplify_symengine;
use rand::RngExt;
use std::time::Instant;
use std::f32::consts::TAU;

fn get_config() -> Config {
    Config {
        num_islands: 24,
        island_size: 25,
        max_generations: 5000,   
        crossover_rate: 0.10,
        tournament_size: 2,
        migration_interval: 25,
        parsimony_penalty: 0.000005,

        opt_prob: 0.2,
        opt_iterations: 100,
        final_opt_iterations: 5000,
        
        stagnation_threshold: 1000,
        target_mse: 1e-7,
        min_improvement: 1e-6,

        random_injection_rate: 0.10,
        min_random_injection: 2,
        max_tree_size: 32,
        mutation_max_depth: 7,
        mutation_cycles: 5,
        verbose: true,
       
        allowed_modules: vec![OpModule::Basic, OpModule::Linalg],
        custom_ops: vec![], 
        // TELJESEN KIZÁRJUK A CSALÁST: Nincs trigonometria, nincs logaritmus!
        excluded_ops: vec![
            Op::SinF, 
            Op::CosF, 
            Op::ExpF, 
            Op::LnF
        ],
    }
}

// Segédfüggvény: Véletlenszerű, egyenletes 3D térbeli forgatás (Haar-mérték)
fn get_random_rotation(rng: &mut impl RngExt) -> [f32; 9] {
    let u1: f32 = rng.random_range(0.0..1.0);
    let u2: f32 = rng.random_range(0.0..1.0);
    let u3: f32 = rng.random_range(0.0..1.0);
    let w = (1.0 - u1).sqrt() * (TAU * u2).sin();
    let x = (1.0 - u1).sqrt() * (TAU * u2).cos();
    let y = u1.sqrt() * (TAU * u3).sin();
    let z = u1.sqrt() * (TAU * u3).cos();
    
    [
        1.0 - 2.0*y*y - 2.0*z*z, 2.0*x*y - 2.0*z*w,     2.0*x*z + 2.0*y*w,
        2.0*x*y + 2.0*z*w,       1.0 - 2.0*x*x - 2.0*z*z, 2.0*y*z - 2.0*x*w,
        2.0*x*z - 2.0*y*w,       2.0*y*z + 2.0*x*w,     1.0 - 2.0*x*x - 2.0*y*y
    ]
}

fn run_open_problem_benchmark(name: &str, data_x: Vec<Vec<f32>>, data_y: Vec<f32>, feature_types: Vec<ValueType>) {
    println!("\n========================================================");
    println!(">>> RUNNING EXACT TENSOR INVARIANT PROBLEM: {} <<<", name);
    println!("Samples: {}", data_x.len());

    let dataset = Dataset::new(&data_x, &data_y, feature_types, true);
    let config = get_config();
    let strategy = StaticStrategy::new(config);
    let allowed_ops = strategy.get_allowed_operators();

    let var_registry = dataset.get_variable_registry();
    let mut engine = Engine::new(strategy, var_registry, allowed_ops);
    
    let start_time = Instant::now();
    engine.run(&dataset);
    let duration = start_time.elapsed();

    println!("--- RESULT FOR: {} ---", name);
    println!("Time taken: {:?}", duration);
    
    let pareto_front = engine.get_pareto_front();
    println!("{:<6} | {:<15} | {}", "Compl", "MSE", "Discovered Equation");
    println!("--------------------------------------------------------");
    
    let display_count = pareto_front.len().min(10);
    for (complexity, mse, ind) in pareto_front.into_iter().take(display_count) {
        let raw_eq = ind.to_string();
        let clean_eq = simplify_symengine(&raw_eq); 
        println!("{:<6} | {:<15.8} | {}", complexity, mse, clean_eq);
    }
    println!("========================================================\n");
}

fn main() {
    let mut rng = rand::rng();
    let num_samples = 400; 

    let ns = [
        [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, -1.0],
    ];
    let ms = [
        [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0],
        [0.0, 1.0, -1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, -1.0, 0.0],
    ];

    println!("Precomputing 1000 random crystal orientations (Microstructure)...");
    let num_grains = 1000;
    let mut all_schmid_tensors = Vec::with_capacity(num_grains * 12);

    for _ in 0..num_grains {
        let rot = get_random_rotation(&mut rng);
        for s in 0..12 {
            let mut p_local = [0.0; 9];
            for i in 0..3 {
                for j in 0..3 {
                    p_local[i*3 + j] = (ms[s][i] * ns[s][j] + ns[s][i] * ms[s][j]) / (2.0 * 6.0f32.sqrt());
                }
            }
            
            let mut p_global = [0.0; 9];
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        for l in 0..3 {
                            p_global[i*3 + j] += rot[i*3 + k] * p_local[k*3 + l] * rot[j*3 + l];
                        }
                    }
                }
            }
            all_schmid_tensors.push(p_global);
        }
    }

    let mut dx = Vec::with_capacity(num_samples);
    let mut dy = Vec::with_capacity(num_samples);

    println!("Simulating Macroscopic Yield Dissipation...");
    for _ in 0..num_samples {
        let s11 = rng.random_range(-1.0..1.0);
        let s22 = rng.random_range(-1.0..1.0);
        let s33 = -s11 - s22;
        let s12 = rng.random_range(-1.0..1.0);
        let s13 = rng.random_range(-1.0..1.0);
        let s23 = rng.random_range(-1.0..1.0);
        
        let stress_tensor = [
            s11, s12, s13,
            s12, s22, s23,
            s13, s23, s33
        ];

        let mut macro_yield = 0.0;
        for p in &all_schmid_tensors {
            let mut tau = 0.0;
            for i in 0..9 {
                tau += p[i] * stress_tensor[i];
            }
            macro_yield += tau.powi(6);
        }
        macro_yield /= num_grains as f32;

        let mut row = Vec::with_capacity(9);
        row.extend_from_slice(&stress_tensor);
        
        dx.push(row);
        dy.push(macro_yield);
    }

    let feature_types = vec![ValueType::Mat3];

    run_open_problem_benchmark(
        "FCC Polycrystal Exact Macroscopic Yield Function", 
        dx, 
        dy, 
        feature_types
    );
}