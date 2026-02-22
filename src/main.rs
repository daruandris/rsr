// src/main.rs
use rsr::EvolutionConfig;
use rsr::Engine;
use rsr::SimdDataset;
use rsr::StaticStrategy;
use rsr::Strategy;
use rsr::{UniversalDomain, UniversalType, UniversalOp};
use rsr::ffi::symengine::simplify_symengine;
use rsr::engine::config::OpModule;
use rand::RngExt;
use std::time::Instant;

fn get_config() -> EvolutionConfig {
    EvolutionConfig {
        num_islands: 24,
        island_size: 25,
        max_generations: 6000,   
        crossover_rate: 0.10,
        tournament_size: 2,
        migration_interval: 25,
        parsimony_penalty: 0.00000,

        opt_prob: 0.2,
        opt_iterations: 100,
        final_opt_iterations: 5000,
        
        stagnation_threshold: 1000,
        target_mse: 1e-7,
        min_improvement: 1e-6,

        random_injection_rate: 0.10,
        min_random_injection: 2,
        max_tree_size: 32,
        mutation_max_depth: 7,
        mutation_cycles: 5,
        verbose: true,
       
        // Linalg és Basic modul engedélyezve
        allowed_modules: vec![OpModule::Basic, OpModule::Linalg],
        custom_ops: vec![], 
        excluded_ops: vec![
        ], // Ezzel drasztikusan felgyorsítod a keresést!
    }
}

// -----------------------------------------------------------------------------
// ADATGENERÁLÁS KÉPLET NÉLKÜL: Numerikus szimuláció
// -----------------------------------------------------------------------------
// Ez a függvény nem egy zárt egyenletet használ! Lépésről lépésre szimulálja 
// a gravitációt (Euler módszerrel). Nincs ismert véges képlet a végeredményre.
fn simulate_3body_final_distance(
    mut p: [[f32; 3]; 3], 
    mut v: [[f32; 3]; 3]
) -> f32 {
    let dt = 0.01;
    let steps = 100_000; // t = 1.0 másodperc szimulálása
    let g = 1.0;     // Gravitációs állandó
    let m = [1.0, 1.0, 1.0]; // Tömegek (legyenek azonosak az egyszerűség kedvéért)

    for _ in 0..steps {
        let mut f = [[0.0; 3]; 3];
        // Erők kiszámítása (Newton féle gravitáció)
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    let dx = p[j][0] - p[i][0];
                    let dy = p[j][1] - p[i][1];
                    let dz = p[j][2] - p[i][2];
                    
                    // Pici hozzáadás (softening), hogy ne szálljon el nullával osztásnál, ha ütköznek
                    let dist_sq = dx*dx + dy*dy + dz*dz + 0.01; 
                    let dist = dist_sq.sqrt();
                    
                    let force = (g * m[i] * m[j]) / dist_sq;
                    
                    f[i][0] += force * (dx / dist);
                    f[i][1] += force * (dy / dist);
                    f[i][2] += force * (dz / dist);
                }
            }
        }
        
        // Pozíciók és sebességek frissítése
        for i in 0..3 {
            v[i][0] += (f[i][0] / m[i]) * dt;
            v[i][1] += (f[i][1] / m[i]) * dt;
            v[i][2] += (f[i][2] / m[i]) * dt;
            
            p[i][0] += v[i][0] * dt;
            p[i][1] += v[i][1] * dt;
            p[i][2] += v[i][2] * dt;
        }
    }
    
    // A cél: Mi lesz a távolság a 0. és 1. test között a szimuláció végén?
    let dx = p[1][0] - p[0][0];
    let dy = p[1][1] - p[0][1];
    let dz = p[1][2] - p[0][2];
    (dx*dx + dy*dy + dz*dz).sqrt()
}

fn run_open_problem_benchmark(name: &str, data_x: Vec<Vec<f32>>, data_y: Vec<f32>, feature_types: Vec<UniversalType>) {
    println!("\n========================================================");
    println!(">>> RUNNING OPEN PROBLEM: {} <<<", name);
    println!("Samples: {}", data_x.len());

    let dataset = SimdDataset::new(&data_x, &data_y, feature_types, true);
    let config = get_config();
    let strategy = StaticStrategy::new(config);
    let allowed_ops = strategy.get_allowed_operators();

    let var_registry = dataset.get_variable_registry();
    let mut engine = Engine::<StaticStrategy, UniversalDomain>::new(strategy, var_registry, allowed_ops);
    
    let start_time = Instant::now();
    engine.run_evolution(&dataset);
    let duration = start_time.elapsed();

    println!("--- RESULT FOR: {} ---", name);
    println!("Time taken: {:?}", duration);
    
    let pareto_front = engine.get_pareto_front();
    println!("{:<6} | {:<15} | {}", "Compl", "MSE", "Discovered Equation");
    println!("--------------------------------------------------------");
    
    let display_count = pareto_front.len();
    for (complexity, mse, ind) in pareto_front.into_iter().take(display_count) {
        let raw_eq = ind.to_string();
        let clean_eq = simplify_symengine(&raw_eq); 
        println!("{:<6} | {:<15.8} | {}", complexity, mse, clean_eq);
    }
    println!("========================================================\n");
}

fn main() {
    let mut rng = rand::rng();
    let n = 400; // 500 egyedi univerzum-kifutás

    let mut dx = Vec::with_capacity(n);
    let mut dy = Vec::with_capacity(n);

    println!("Generating ground-truth data via numerical simulation...");
    for _ in 0..n {
        // Véletlenszerű kezdeti pozíciók (-5.0 .. 5.0)
        let p = [
            [rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)],
            [rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)],
            [rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)],
        ];
        
        // Véletlenszerű kezdeti sebességek (-1.0 .. 1.0)
        let v = [
            [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)],
            [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)],
            [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)],
        ];

        // "Megmérjük" a valóságot a szimulátorral
        let final_distance = simulate_3body_final_distance(p, v);

        // Bemeneti vektorok ellapítása a SR motornak (6 db Vec3 = 18 float)
        let mut row = Vec::with_capacity(18);
        row.extend_from_slice(&p[0]); row.extend_from_slice(&p[1]); row.extend_from_slice(&p[2]);
        row.extend_from_slice(&v[0]); row.extend_from_slice(&v[1]); row.extend_from_slice(&v[2]);
        
        dx.push(row);
        dy.push(final_distance);
    }

    // Elmondjuk a motornak, hogy a 18 float valójában 6 darab 3D vektor
    let feature_types = vec![
        UniversalType::Vec3, UniversalType::Vec3, UniversalType::Vec3, // P0, P1, P2
        UniversalType::Vec3, UniversalType::Vec3, UniversalType::Vec3  // V0, V1, V2
    ];

    run_open_problem_benchmark(
        "General 3-Body Problem (Finding closed-form approx)", 
        dx, 
        dy, 
        feature_types
    );
}