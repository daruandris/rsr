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
    
    let num_samples = 300;
    let mut data_x = Vec::with_capacity(num_samples);
    let mut data_y = Vec::with_capacity(num_samples);
    
    let mut rng = rand::rng();
    
    let num_features = 3;

    for _ in 0..num_samples {
        let v0 = rng.random_range(0.0..15.0); // Kezdősebesség (pl. m/s)
        let t = rng.random_range(0.1..10.0);  // Idő (másodperc, csak pozitív!)
        let a = rng.random_range(-9.81..9.81); // Gyorsulás (lehet negatív is, pl. fékezés)
        
        data_x.push(vec![v0, t, a]);
        
        // s = v0*t + 0.5 * a * t^2
        let distance = v0 * t + 0.5 * a * t * t;
        data_y.push(distance);
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