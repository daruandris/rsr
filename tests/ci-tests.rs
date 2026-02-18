use rsr::Engine;
use std::time::Instant;
mod common;

#[test]
fn test_linear_convergence() {
    let (dataset, num_features) = common::create_linear_data();
    let config = common::get_basic_config();
    let mut engine = Engine::new(config, num_features);
    
    let start = Instant::now();
    engine.run_evolution(&dataset);
    let duration = start.elapsed();
    
    let best = engine.get_global_best();
    let mse = best.calculate_mse(&dataset);

    println!("Linear Solved in: {:.2?}", duration);
    println!("Best Linear Equation: {}", best);
    println!("MSE: {}", mse);
    assert!(mse < 1e-5, "Too much mse with linear equation!");
}

#[test]
fn test_quadratic_convergence() {
    let (dataset, num_features) = common::create_quadratic_data();
    let mut config = common::get_basic_config();
    config.max_generations = 200;
    
    let mut engine = Engine::new(config, num_features);

    let start = Instant::now();
    engine.run_evolution(&dataset);
    let duration = start.elapsed();
    
    let best = engine.get_global_best();
    let mse = best.calculate_mse(&dataset);

    println!("Quadratic Solved in: {:.2?}", duration);
    println!("Best Quadratic Equation: {}", best);
    println!("MSE: {}", mse);
    assert!(mse < 1e-4, "Too much mse with quadratic equation!");
}

#[test]
fn test_sine_convergence() {
    let (dataset, num_features) = common::create_sine_wave_data();
    let mut config = common::get_basic_config();
    config.max_generations = 200;
    let mut engine = Engine::new(config, num_features);

    let start = Instant::now();
    engine.run_evolution(&dataset);
    let duration = start.elapsed();

    let best = engine.get_global_best();
    let mse = best.calculate_mse(&dataset);


    println!("Sine Solved in: {:.2?}", duration);
    println!("Best Equation: {}", best);
    println!("MSE: {:.8}", mse);

    assert!(mse < 1e-4, "Too much mse with sine equation!");
}

#[test]
fn test_exp_convergence() {
    let (dataset, num_features) = common::create_exponential_data();
    let mut config = common::get_basic_config();
    config.max_generations = 500;
    let mut engine = Engine::new(config, num_features);
    
    let start = Instant::now();
    engine.run_evolution(&dataset);
    let duration = start.elapsed();

    let best = engine.get_global_best();
    let mse = best.calculate_mse(&dataset);


    println!("Exp Solved in: {:.2?}", duration);
    println!("Best Equation: {}", best);
    println!("MSE: {:.8}", mse);

    assert!(mse < 1e-4, "Too much mse with exp equation!");
}