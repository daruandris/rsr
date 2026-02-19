use rsr::{Engine, SimdDataset, EvolutionConfig};
use std::time::Instant;
mod common;

fn get_basic_config() -> EvolutionConfig {
    EvolutionConfig {
        num_islands: 8,
        island_size: 100,
        max_generations: 5000,
        crossover_rate: 0.85,
        tournament_size: 3,
        migration_interval: 10,
        parsimony_penalty: 0.01,
        
        opt_prob: 0.1,
        opt_iterations: 5,
        final_opt_iterations: 1000,
        
        stagnation_threshold: 10,
        min_improvement: 1e-5,
        target_mse: 1e-7,

        random_injection_rate: 0.05,
        min_random_injection: 1,
        max_tree_size : 30,
        verbose : false,
    }
}

fn create_linear_data() -> (SimdDataset, u8) {
    // Cél: y = 2 * X0 + 5
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        let val = i as f32;
        x.push(vec![val]);
        y.push(2.0 * val + 5.0);
    }
    (SimdDataset::new(&x, &y, 1), 1)
}

fn create_quadratic_data() -> (SimdDataset, u8) {
    // Cél: y = X0^2 - 10
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        let val = (i as f32) / 5.0;
        x.push(vec![val]);
        y.push(val * val - 10.0);
    }
    (SimdDataset::new(&x, &y, 1), 1)
}

// Cél: y = 2.5 * sin(3.0 * X0)
fn create_sine_wave_data() -> (SimdDataset, u8) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        let val = (i as f32) * 0.15; 
        x.push(vec![val]);
        y.push(2.5 * (3.0 * val).sin());
    }
    (SimdDataset::new(&x, &y, 1), 1)
}

// Cél: y = exp(0.5 * X0)
fn create_exponential_data() -> (SimdDataset, u8) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        let val = (i as f32) * 0.1;
        x.push(vec![val]);
        y.push((0.5 * val).exp());
    }
    (SimdDataset::new(&x, &y, 1), 1)
}

#[test]
fn test_linear_convergence() {
    let (dataset, num_features) = create_linear_data();
    let config = get_basic_config();
    let mut engine = Engine::new(config, num_features);
    
    let start = Instant::now();
    engine.run_evolution(&dataset);
    let duration = start.elapsed();
    
    let mut best = engine.get_global_best().clone();
    let mse = best.calculate_mse(&dataset);

    println!("Linear Solved in: {:.2?}", duration);
    println!("Best Linear Equation: {}", best);
    println!("MSE: {}", mse);
    assert!(mse < 1e-5, "Too much mse with linear equation!");
}

#[test]
fn test_quadratic_convergence() {
    let (dataset, num_features) = create_quadratic_data();
    let mut config = get_basic_config();
    config.max_generations = 200;
    
    let mut engine = Engine::new(config, num_features);

    let start = Instant::now();
    engine.run_evolution(&dataset);
    let duration = start.elapsed();
    
    let mut best = engine.get_global_best().clone();
    let mse = best.calculate_mse(&dataset);

    println!("Quadratic Solved in: {:.2?}", duration);
    println!("Best Quadratic Equation: {}", best);
    println!("MSE: {}", mse);
    assert!(mse < 1e-4, "Too much mse with quadratic equation!");
}

#[test]
fn test_sine_convergence() {
    let (dataset, num_features) = create_sine_wave_data();
    let mut config = get_basic_config();
    config.max_generations = 200;
    let mut engine = Engine::new(config, num_features);

    let start = Instant::now();
    engine.run_evolution(&dataset);
    let duration = start.elapsed();

    let mut best = engine.get_global_best().clone();
    let mse = best.calculate_mse(&dataset);


    println!("Sine Solved in: {:.2?}", duration);
    println!("Best Equation: {}", best);
    println!("MSE: {:.8}", mse);

    assert!(mse < 1e-4, "Too much mse with sine equation!");
}

#[test]
fn test_exp_convergence() {
    let (dataset, num_features) = create_exponential_data();
    let mut config = get_basic_config();
    config.max_generations = 500;
    let mut engine = Engine::new(config, num_features);
    
    let start = Instant::now();
    engine.run_evolution(&dataset);
    let duration = start.elapsed();

    let mut best = engine.get_global_best().clone();
    let mse = best.calculate_mse(&dataset);


    println!("Exp Solved in: {:.2?}", duration);
    println!("Best Equation: {}", best);
    println!("MSE: {:.8}", mse);

    assert!(mse < 1e-4, "Too much mse with exp equation!");
}