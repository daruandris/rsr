// src/main.rs
use std::f64::consts::PI;
use population::{Engine, EvolutionConfig};

pub mod node;
pub mod individual;
pub mod population;
pub mod operators;

fn main() {
    println!("=== Szimbolikus Regressziós Motor ===");
    println!("Cél: y = (X0 * X0) + 3.0 * sin(X1)...");
    
    let num_samples = 200;
    let num_features = 2;
    let mut data_x = Vec::with_capacity(num_samples);
    let mut data_y = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let x0 = (i as f64 / num_samples as f64) * 4.0 - 2.0;
        let x1 = (i as f64 / num_samples as f64) * 2.0 * PI;
        data_x.push(vec![x0, x1]);
        data_y.push(x0 * x0 + 3.0 * x1.sin());
    }

    // A stratégiai paraméterek SOTA alapokon 
    let config = EvolutionConfig {
        num_islands: 4,               // Logikai magok száma
        island_size: 200,             // Növelve 50-ről 200-ra a jobb diverzitásért
        max_generations: 500,         
        crossover_rate: 0.85,         // 85% keresztezés, 15% mutáció
        tournament_size: 5,           // Nagyobb nyomás a pontosabb egyedekért
        migration_interval: 20,       // Minden 20. generációban sziget-migráció
        parsimony_penalty: 0.001,     // Büntetés a hosszú kifejezésekért
    };
    
    println!("Motor inicializálása: {} sziget, egyenként {} egyeddel...", config.num_islands, config.island_size);
    let mut engine = Engine::new(config, num_features);

    println!("Evolúció indítása ({} generáció)...", config.max_generations);
    engine.run_evolution(&data_x, &data_y);

    let best = engine.get_global_best();
    println!("\n=== Evolúció Befejeződött ===");
    println!("Legjobb fitness (MSE + büntetés): {:.6}", best.fitness);
    println!("Kifejezés hossza: {} csomópont (AST)", best.nodes.len());
    
    // Az új Display trait használata a gyönyörű kiíráshoz
    println!("\nTalált Képlet: {}", best);
}