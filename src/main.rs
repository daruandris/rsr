use std::f64::consts::PI;
use population::{Engine, EvolutionConfig};
use rand::RngExt;

use simplify::simplify_symengine;

pub mod node;
pub mod individual;
pub mod population;
pub mod operators;
pub mod simplify;

fn main() {
    println!("=== Szimbolikus Regressziós Motor ===");
    println!("Cél: y=2.5⋅exp(-0.5⋅X0​)⋅cos(3.0⋅X1​)");
    
    let num_samples = 200;
    let mut data_x = Vec::with_capacity(num_samples);
    let mut data_y = Vec::with_capacity(num_samples);
    
    let mut rng = rand::rng();
    
    let num_features = 1;

    for _ in 0..num_samples {
        let x0: f64 = rng.random_range(-3.14..3.14);
        data_x.push(vec![x0]);
        data_y.push(2.0 * (3.0 * x0.cos()).sin());
    }

    let config = EvolutionConfig {
        num_islands: 4,
        island_size: 500,
        max_generations: 10000,         
        crossover_rate: 0.85,
        tournament_size: 3,
        migration_interval: 25,
        parsimony_penalty: 0.005,

        opt_prob: 0.05,
        opt_iterations: 3,
        opt_lr: 0.5,
        opt_epsilon: 1e-5,
        
        stagnation_threshold: 50,
        target_mse: 1e-6,
    };
    
    println!("Motor inicializálása: {} sziget, egyenként {} egyeddel...", config.num_islands, config.island_size);
    let mut engine = Engine::new(config, num_features);

    println!("Evolúció indítása ({} generáció)...", config.max_generations);
    engine.run_evolution(&data_x, &data_y);

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