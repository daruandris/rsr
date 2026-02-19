use rsr::EvolutionConfig;
use rsr::Engine;
use rsr::SimdDataset;
use rsr::StaticStrategy;
use rsr::BasicDomain;
// Ha a symengine import máshogy van nálad, igazítsd a sajátodhoz:
use rsr::ffi::symengine::simplify_symengine; 
use rand::RngExt;
use std::time::Instant;

fn get_config() -> EvolutionConfig {
    EvolutionConfig {
        num_islands: 24,
        island_size: 25,
        max_generations: 10000,   
        crossover_rate: 0.10,
        tournament_size: 2,
        migration_interval: 25,
        parsimony_penalty: 0.0,

        opt_prob: 0.2,
        opt_iterations: 100,
        final_opt_iterations: 5000,
        
        stagnation_threshold: 1000,
        target_mse: 1e-7,
        min_improvement: 1e-6,

        random_injection_rate: 0.10,
        min_random_injection: 2,
        max_tree_size: 32,
        mutation_max_depth: 4,
        mutation_cycles: 5,
        verbose: true,
    }
}

fn run_benchmark(
    name: &str, 
    data_x: Vec<Vec<f32>>, 
    data_y: Vec<f32>, 
    num_features: u8,
    expected_noise_mse: Option<f32>
) {
    println!("\n========================================================");
    println!(">>> RUNNING BENCHMARK: {} <<<", name);
    println!("Features: {}, Samples: {}", num_features, data_x.len());
    
    let dataset = SimdDataset::new(&data_x, &data_y, num_features);
    let mut config = get_config();
    
    if let Some(noise_mse) = expected_noise_mse {
        config.target_mse = noise_mse;
        println!("Note: Noisy data detected. Adjusted Target MSE to {:.6}", noise_mse);
    }

    let strategy = StaticStrategy::new(config);

    let mut engine = Engine::<StaticStrategy, BasicDomain>::new(strategy, num_features);
    
    let start_time = Instant::now();
    engine.run_evolution(&dataset);
    let duration = start_time.elapsed();

    println!("--- RESULT FOR: {} ---", name);
    println!("Time taken: {:?}", duration);
    
    let pareto_front = engine.get_pareto_front();
    println!("{:<6} | {:<15} | {}", "Compl", "MSE", "Simplified Expression");
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
    let n = 400; // Minták száma

    // ----------------------------------------------------------------------
    // 1. Egyszerű Négyzetes: y = 2.5 * x^2 - 1.2
    let mut dx1 = Vec::new(); let mut dy1 = Vec::new();
    for _ in 0..n {
        let x = rng.random_range(-5.0..5.0);
        dx1.push(vec![x]);
        dy1.push(2.5 * x * x - 1.2);
    }
    run_benchmark("1. Simple Square (Sqr)", dx1, dy1, 1, None);

    // ----------------------------------------------------------------------
    // 2. Egyszerű Trigonometria: y = 3.0 * cos(2.0 * x) + 1.0
    let mut dx2 = Vec::new(); let mut dy2 = Vec::new();
    for _ in 0..n {
        let x: f32 = rng.random_range(-3.14..3.14);
        dx2.push(vec![x]);
        dy2.push(3.0 * (2.0 * x).cos() + 1.0);
    }
    run_benchmark("2. Simple Trigonometry (Cos)", dx2, dy2, 1, None);

    // ----------------------------------------------------------------------
    // 3. Egyszerű Exponenciális: y = 1.5 * exp(0.5 * x)
    let mut dx3 = Vec::new(); let mut dy3 = Vec::new();
    for _ in 0..n {
        let x: f32 = rng.random_range(-2.0..4.0);
        dx3.push(vec![x]);
        dy3.push(1.5 * (0.5 * x).exp());
    }
    run_benchmark("3. Simple Exponential (Exp)", dx3, dy3, 1, None);

    // ----------------------------------------------------------------------
    // 4. Bonyolult Egyváltozós: y = exp(-0.5 * x) * cos(3.0 * x)
    let mut dx4 = Vec::new(); let mut dy4 = Vec::new();
    for _ in 0..n {
        let x: f32 = rng.random_range(0.0..10.0);
        dx4.push(vec![x]);
        dy4.push((-0.5 * x).exp() * (3.0 * x).cos());
    }
    run_benchmark("4. Complex 1D (Damped Osc.)", dx4, dy4, 1, None);

    // ----------------------------------------------------------------------
    // 5. Egyszerű Többváltozós: y = 2.0*x0 - 3.5*x1 + 1.2*x2
    let mut dx5 = Vec::new(); let mut dy5 = Vec::new();
    for _ in 0..n {
        let x0 = rng.random_range(-5.0..5.0);
        let x1 = rng.random_range(-5.0..5.0);
        let x2 = rng.random_range(-5.0..5.0);
        dx5.push(vec![x0, x1, x2]);
        dy5.push(2.0 * x0 - 3.5 * x1 + 1.2 * x2);
    }
    run_benchmark("5. Simple Multivariable (Linear)", dx5, dy5, 3, None);

    // ----------------------------------------------------------------------
    // 6. Bonyolult Többváltozós: y = x0^2 + sin(x1) - x2
    let mut dx6 = Vec::new(); let mut dy6 = Vec::new();
    for _ in 0..n {
        let x0 = rng.random_range(-3.0..3.0);
        let x1: f32 = rng.random_range(-3.14..3.14);
        let x2 = rng.random_range(-5.0..5.0);
        dx6.push(vec![x0, x1, x2]);
        dy6.push(x0 * x0 + x1.sin() - x2);
    }
    run_benchmark("6. Complex Multivariable", dx6, dy6, 3, None);

    // ----------------------------------------------------------------------
    // 7. Zajos Adat: y = 2.5 * x^2 + noise(-0.5..0.5)
    let mut dx7 = Vec::new(); let mut dy7 = Vec::new();
    for _ in 0..n {
        let x = rng.random_range(-5.0..5.0);
        let noise = rng.random_range(-0.5..0.5);
        dx7.push(vec![x]);
        dy7.push(2.5 * x * x + noise);
    }
    // Noise range 1.0 -> Variance (Expected MSE) = 1.0^2 / 12 = 0.0833
    // Mivel az adataid Z-score normalizálva lesznek belül, az elvárt MSE is skálázódik, 
    // de hagyjuk None-on, és nézzük meg, hol áll meg.
    run_benchmark("7. Noisy Data (Robustness)", dx7, dy7, 1, None); 

    // ----------------------------------------------------------------------
    // 8. Rejtett Dimenziók: 5 bemenet, de csak 2 számít (y = x0 * x3)
    let mut dx8 = Vec::new(); let mut dy8 = Vec::new();
    for _ in 0..n {
        let x0 = rng.random_range(-5.0..5.0);
        let x1 = rng.random_range(-5.0..5.0);
        let x2 = rng.random_range(-5.0..5.0);
        let x3 = rng.random_range(-5.0..5.0);
        let x4 = rng.random_range(-5.0..5.0); 
        dx8.push(vec![x0, x1, x2, x3, x4]);
        dy8.push(x0 * x3);
    }
    run_benchmark("8. Hidden Dimensions (Feature Select)", dx8, dy8, 5, None);

    // ----------------------------------------------------------------------
    // 9. Racionális Törtfüggvény: y = (x0 + 1.5) / (x0^2 + 2.0)
    let mut dx9 = Vec::new(); let mut dy9 = Vec::new();
    for _ in 0..n {
        let x = rng.random_range(-5.0..5.0);
        dx9.push(vec![x]);
        dy9.push((x + 1.5) / (x * x + 2.0));
    }
    run_benchmark("9. Rational Function (Div)", dx9, dy9, 1, None);

    // ----------------------------------------------------------------------
    // 10. "Minden Egyben": y = exp(-x0) + cos(x1) - (x2^2 / (x3 + 1.1))
    // Csel: az adathatárt úgy állítjuk, hogy a nevező ne lehessen 0.
    let mut dx10 = Vec::new(); let mut dy10 = Vec::new();
    for _ in 0..n {
        let x0: f32 = rng.random_range(0.0..3.0);
        let x1: f32 = rng.random_range(-3.14..3.14);
        let x2 = rng.random_range(-3.0..3.0);
        let x3 = rng.random_range(0.0..5.0);
        dx10.push(vec![x0, x1, x2, x3]);
        dy10.push((-x0).exp() + x1.cos() - ((x2 * x2) / (x3 + 1.1)));
    }
    run_benchmark("10. Ultimate Multivariable All-Ops", dx10, dy10, 4, None);
}