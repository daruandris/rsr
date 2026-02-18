use rsr::Engine;
mod common;

#[test]
fn test_linear_convergence() {
    let (dataset, num_features) = common::create_linear_data();
    let config = common::get_basic_config();
    let mut engine = Engine::new(config, num_features);
    
    engine.run_evolution(&dataset);
    
    let best = engine.get_global_best();
    let mse = best.calculate_mse(&dataset);

    println!("Best Linear Equation: {}", best);
    println!("MSE: {}", mse);
    assert!(mse < 1e-5, "A motornak meg kellett volna találnia a lineáris függvényt!");
}

#[test]
fn test_quadratic_convergence() {
    let (dataset, num_features) = common::create_quadratic_data();
    let mut config = common::get_basic_config();
    config.max_generations = 200;
    
    let mut engine = Engine::new(config, num_features);
    engine.run_evolution(&dataset);
    
    let best = engine.get_global_best();
    let mse = best.calculate_mse(&dataset);

    println!("Best Quadratic Equation: {}", best);
    
    assert!(mse < 1e-4, "A motornak meg kellett volna találnia a másodfokú függvényt!");
}

#[test]
fn test_sine_convergence() {
    let (dataset, num_features) = common::create_sine_wave_data();
    let mut config = common::get_basic_config();
    config.max_generations = 200;
    let mut engine = Engine::new(config, num_features);
    engine.run_evolution(&dataset);
    assert!(engine.get_global_best().calculate_mse(&dataset) < 1e-4);
}

#[test]
fn test_exp_convergence() {
    let (dataset, num_features) = common::create_exponential_data();
    let mut config = common::get_basic_config();
    config.max_generations = 500;
    let mut engine = Engine::new(config, num_features);
    engine.run_evolution(&dataset);
    assert!(engine.get_global_best().calculate_mse(&dataset) < 1e-4);
}