use std::f64::consts::PI;
use population::Engine;

pub mod node;
pub mod individual;
pub mod population;
pub mod operators;

fn main() {
    println!("=== Szimbolikus Regressziós Motor ===");
    println!("Szintetikus adatok generálása: y = X0 * X0 + 3.0 * sin(X1)...");
    
    let num_samples = 200;
    let num_features = 2;
    let mut data_x = Vec::with_capacity(num_samples);
    let mut data_y = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let x0 = (i as f64 / num_samples as f64) * 4.0 - 2.0;
        let x1 = (i as f64 / num_samples as f64) * 2.0 * PI;
        
        data_x.push(vec![x0, x1]);
        // Az elvárt y érték
        data_y.push(x0 * x0 + 3.0 * x1.sin());
    }

    // A SOTA stratégia szerint beállítjuk a sziget-modellt: [cite: 43]
    let num_islands = 4; // Logikai magok száma (pl. 4 mag = 4 sziget) [cite: 44]
    let island_size = 50; // 50 egyed per sziget [cite: 46]
    
    println!("Motor inicializálása: {} sziget, egyenként {} egyeddel...", num_islands, island_size);
    let mut engine = Engine::new(num_islands, island_size, num_features);

    let max_generations = 500;
    println!("Evolúció indítása ({} generáció)...", max_generations);
    
    // START! Itt a Rayon szétosztja a munkát a CPU magokon
    engine.run_evolution(max_generations, &data_x, &data_y);

    let best = engine.get_global_best();
    println!("\n=== Evolúció Befejeződött ===");
    println!("Legjobb fitness (MSE + büntetés): {}", best.fitness);
    println!("Kifejezés hossza: {} csomópont (AST)", best.nodes.len());
    
    // Kiírjuk a lapos postfix tömböt, hogy lássuk a struktúrát
    println!("Postfix struktúra: {:?}", best.nodes);
}
