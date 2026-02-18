mod common;

use rsr::{Engine, EvolutionConfig, SimdDataset};
use rand::RngExt;

fn get_heavy_config() -> EvolutionConfig {
    EvolutionConfig {
        num_islands: 5,
        island_size: 500,
        max_generations: 10_000,
        crossover_rate: 0.75,
        tournament_size: 4,
        migration_interval: 50,
        parsimony_penalty: 1e-7,
        opt_prob: 0.05,
        opt_iterations: 20,
        stagnation_threshold: 100,
        target_mse: 1e-7,
        min_improvement: 1e-8,
        random_injection_rate: 0.04,
        min_random_injection: 2,
        verbose : false
    }
}

// 1. FIZIKA: Csillapított Harmonikus Oszcillátor
// Képlet: y = A * exp(-gamma * t) * cos(omega * t)
#[test]
#[ignore]
fn benchmark_physics_damped_oscillator() {
    println!("--- BENCHMARK: Physics (Damped Oscillator) ---");
    let num_samples = 300;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    let mut rng = rand::rng();

    // Paraméterek: A=5.0, gamma=0.5, omega=3.0
    // Képlet: 5.0 * exp(-0.5 * t) * cos(3.0 * t)
    for _ in 0..num_samples {
        let t: f64 = rng.random_range(0.0..10.0); 
        x_data.push(vec![t]);
        
        let noise = rng.random_range(-0.01..0.01);
        let y = 5.0 * (-0.5 * t).exp() * (3.0 * t).cos() + noise;
        y_data.push(y);
    }

    let dataset = SimdDataset::new(&x_data, &y_data, 1);
    let mut engine = Engine::new(get_heavy_config(), 1);

    let start = std::time::Instant::now();
    engine.run_evolution(&dataset);

    let duration = start.elapsed();
    let best = engine.get_global_best();
    let mse = best.calculate_mse(&dataset);
    
    println!("Physics solved in: {:?}", duration);
    println!("Best: {}", best);
    println!("Mse: {}", mse);

    common::append_benchmark_result("Physics", mse, duration.as_millis());

    assert!(mse < 0.01); 
}

// 2. BIOLÓGIA / KÉMIA: Michaelis-Menten Kinetika
// Képlet: v = (Vmax * [S]) / (Km + [S])
#[test]
#[ignore]
fn benchmark_biology_enzyme_kinetics() {
    println!("--- BENCHMARK: Biology (Michaelis-Menten) ---");
    let num_samples = 300;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    let mut rng = rand::rng();

    // Paraméterek: Vmax = 10.0, Km = 2.5
    // Képlet: (10.0 * x) / (2.5 + x)
    for _ in 0..num_samples {
        let s = rng.random_range(0.1..20.0);
        x_data.push(vec![s]);
        
        let y = (10.0 * s) / (2.5 + s);
        y_data.push(y);
    }

    let dataset = SimdDataset::new(&x_data, &y_data, 1);
    let mut engine = Engine::new(get_heavy_config(), 1);

    let start = std::time::Instant::now();
    engine.run_evolution(&dataset);

    let duration = start.elapsed();
    let best = engine.get_global_best();
    let mse = best.calculate_mse(&dataset);
    
    println!("Biology solved in: {:?}", duration);
    println!("Best: {}", best);
    println!("Mse: {}", mse);

    common::append_benchmark_result("Biology", mse, duration.as_millis());

    
    assert!(mse < 1e-4);
}

// 3. STATISZTIKA: Maxwell-Boltzmann Eloszlás (Sebesség)
// Képlet: f(v) ~ v^2 * exp(-v^2 / a)
#[test]
#[ignore]
fn benchmark_stats_maxwell_boltzmann() {
    println!("--- BENCHMARK: Statistics (Maxwell-Boltzmann) ---");
    let num_samples = 400;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();
    let mut rng = rand::rng();

    // Egyszerűsített alak: y = x^2 * exp(-0.5 * x^2)
    for _ in 0..num_samples {
        let v: f64 = rng.random_range(0.0..5.0);
        x_data.push(vec![v]);
        
        // y = v^2 * exp(-0.5 * v^2)
        let y = (v * v) * (-0.5 * v * v).exp();
        
        // Felszorozzuk, hogy emberibb léptékű legyen a hiba
        y_data.push(y * 10.0); 
    }

    let dataset = SimdDataset::new(&x_data, &y_data, 1);
    let config = get_heavy_config();
    
    let mut engine = Engine::new(config, 1);

    let start = std::time::Instant::now();
    engine.run_evolution(&dataset);

    let duration = start.elapsed();
    let best = engine.get_global_best();
    let mse = best.calculate_mse(&dataset);
    
    println!("PhStatisticsysics solved in: {:?}", duration);
    println!("Best: {}", best);
    println!("Mse: {}", mse);

    common::append_benchmark_result("Statistics", mse, duration.as_millis());

    
    assert!(mse < 1e-3);
}