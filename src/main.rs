use rsr::EvolutionConfig;
use rsr::Engine;
use rsr::SimdDataset;
use rsr::ffi::symengine::simplify_symengine;
use rand::RngExt;

fn main() {
    println!("=== Symbolic Regression Engine ===");
    
    let num_samples = 400;
    let mut data_x = Vec::with_capacity(num_samples);
    let mut data_y = Vec::with_capacity(num_samples);
    
    let mut rng = rand::rng();
    
    let num_features = 3;
    let sqrt_2pi = (2.0 * std::f32::consts::PI).sqrt();
    for _ in 0..num_samples {
        let x = rng.random_range(-5.0..15.0);
        let mu = rng.random_range(0.0..10.0);
        let sigma = rng.random_range(0.1..3.0);
        data_x.push(vec![x, mu, sigma]);
        let z = (x - mu) / sigma;
        let pdf = (1.0 / (sigma * sqrt_2pi)) * (-0.5 * z * z).exp();
        data_y.push(pdf * 10.0);
    }

    let dataset = SimdDataset::new(&data_x, &data_y, num_features);

    let config = EvolutionConfig {
        num_islands: 5,
        island_size: 1000,
        max_generations: 500,         
        crossover_rate: 0.85,
        tournament_size: 3,
        migration_interval: 25,
        parsimony_penalty: 0.005,

        opt_prob: 0.1,
        opt_iterations: 15,
        final_opt_iterations: 2000,
        
        stagnation_threshold: 100,
        target_mse: 1e-6,
        min_improvement: 1e-6,

        random_injection_rate: 0.04,
        min_random_injection: 2,
        verbose : true,
    };
    
    println!("Initializing with {} islands, each of {} individuals...", config.num_islands, config.island_size);
    let mut engine = Engine::new(config, num_features);

    println!("Starting ({} generations)...", config.max_generations);
    engine.run_evolution(&dataset);

    println!("\n=== Evolution ended ===");
    let pareto_front = engine.get_pareto_front();
    
    println!("\n--- PARETO FRONT ---");
    println!("{:<6} | {:<20} | {}", "Complexity", "Pure MSE error", "Expression");
    println!("------------------------------------------------------------");
    for (complexity, mse, ind) in pareto_front {
        let raw_eq = ind.to_string();
        let clean_eq = simplify_symengine(&raw_eq);
        println!("{:<6} | {:<20.8} | {}", complexity, mse, clean_eq);
        println!();
    }
}