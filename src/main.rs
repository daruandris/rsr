use rsr::EvolutionConfig; // Itt használd a Cargo.toml-ben megadott nevet!
use rsr::Engine;
use rsr::SimdDataset;
use rsr::ffi::symengine::simplify_symengine;
use rand::RngExt;

fn main() {
    println!("=== Szimbolikus Regressziós Motor ===");
    println!("Cél: y=2.5⋅exp(-0.5⋅X0​)⋅cos(3.0⋅X1​)");
    
    let num_samples = 400;
    let mut data_x = Vec::with_capacity(num_samples);
    let mut data_y = Vec::with_capacity(num_samples);
    
    let mut rng = rand::rng();
    
    let num_features = 3;
    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt(); // ~2.506628
    for _ in 0..num_samples {
        let x = rng.random_range(-5.0..15.0);   // X0: x érték
        let mu = rng.random_range(0.0..10.0);   // X1: várható érték
        let sigma = rng.random_range(0.5..3.0); // X2: szórás (nem lehet 0!)
        
        data_x.push(vec![x, mu, sigma]);
        
        // Z-érték (standardizálás)
        let z = (x - mu) / sigma;
        
        // Valószínűségi sűrűség
        let pdf = (1.0 / (sigma * sqrt_2pi)) * (-0.5 * z * z).exp();
        data_y.push(pdf);
    }

    let dataset = SimdDataset::new(&data_x, &data_y, num_features);

    let config = EvolutionConfig {
        num_islands: 8,
        island_size: 500,
        max_generations: 10000,         
        crossover_rate: 0.85,
        tournament_size: 3,
        migration_interval: 25,
        parsimony_penalty: 0.00001,

        opt_prob: 0.1,
        opt_iterations: 5,
        opt_lr: 0.001,
        opt_epsilon: 1e-5,
        
        stagnation_threshold: 50,
        target_mse: 1e-5,
    };
    
    println!("Motor inicializálása: {} sziget, egyenként {} egyeddel...", config.num_islands, config.island_size);
    let mut engine = Engine::new(config, num_features);

    println!("Evolúció indítása ({} generáció)...", config.max_generations);
    engine.run_evolution(&dataset);

    println!("\n=== Evolúció Befejeződött ===");
    let pareto_front = engine.get_pareto_front();
    
    println!("\n--- PARETO FRONT (A Legjobb Egyenletek Hossz Szerint) ---");
    println!("{:<6} | {:<20} | {}", "Hossz", "Tiszta MSE Hiba", "Egyenlet");
    println!("------------------------------------------------------------");
    for (complexity, mse, ind) in pareto_front {
        let raw_eq = ind.to_string();
        let clean_eq = simplify_symengine(&raw_eq);
        println!("{:<6} | {:<20.8} | {}", complexity, mse, clean_eq);
    }
}