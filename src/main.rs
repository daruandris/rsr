// src/main.rs
use rsr::EvolutionConfig;
use rsr::Engine;
use rsr::SimdDataset;
use rsr::StaticStrategy;
use rsr::Strategy;
use rsr::{UniversalDomain, UniversalType};
use rsr::ffi::symengine::simplify_symengine;
use rsr::engine::config::OpModule;
use rsr::domain::universal::UniversalOp;
use rand::RngExt;
use std::time::Instant;

fn get_config() -> EvolutionConfig {
    EvolutionConfig {
        num_islands: 24,
        island_size: 25,
        max_generations: 5000,   
        crossover_rate: 0.10,
        tournament_size: 2,
        migration_interval: 25,
        parsimony_penalty: 0.000005,

        opt_prob: 0.2,
        opt_iterations: 100,
        final_opt_iterations: 5000,
        
        stagnation_threshold: 1000,
        target_mse: 1e-7,
        min_improvement: 1e-6,

        random_injection_rate: 0.10,
        min_random_injection: 2,
        max_tree_size: 32,
        mutation_max_depth: 4,
        mutation_cycles: 5,
        verbose: true,
       
        allowed_modules: vec![OpModule::Basic, OpModule::Linalg],
        custom_ops: vec![], 
        // Kivettük a periodikus/exponenciális dolgokat a kérésed szerint!
        excluded_ops: vec![
            UniversalOp::SinF, 
            UniversalOp::CosF, 
            UniversalOp::ExpF, 
            UniversalOp::LnF
        ],
    }
}

// -----------------------------------------------------------------------------
// VIRTUAL WIND TUNNEL: Turbulence Subgrid-Scale Simulation
// Nincs ismert zárt egyenlet, amely a grad_u-t (Mat3) tökéletesen leképezi a kimenetre.
// -----------------------------------------------------------------------------
fn simulate_micro_turbulence(grad_u: &[f32; 9]) -> f32 {
    // 3 pici virtuális folyadék-örvény sebességvektora
    let mut eddies = [[0.1, 0.0, -0.1], [-0.1, 0.1, 0.0], [0.0, -0.1, 0.1]];
    let dt = 0.02;
    let mut total_k = 0.0;
    
    for step in 0..500 {
        let mut next_eddies = [[0.0; 3]; 3];
        for i in 0..3 {
            // 1. Makroszkopikus áramlás ereje (Mátrix * Vektor)
            let fx = grad_u[0]*eddies[i][0] + grad_u[1]*eddies[i][1] + grad_u[2]*eddies[i][2];
            let fy = grad_u[3]*eddies[i][0] + grad_u[4]*eddies[i][1] + grad_u[5]*eddies[i][2];
            let fz = grad_u[6]*eddies[i][0] + grad_u[7]*eddies[i][1] + grad_u[8]*eddies[i][2];
            
            // 2. Nem-lineáris turbulens "kaszkád" (örvények egymásra hatása vektoriális szorzattal)
            let prev = if i == 0 { 2 } else { i - 1 };
            let next = if i == 2 { 0 } else { i + 1 };
            let cx = eddies[prev][1] * eddies[next][2] - eddies[prev][2] * eddies[next][1];
            let cy = eddies[prev][2] * eddies[next][0] - eddies[prev][0] * eddies[next][2];
            let cz = eddies[prev][0] * eddies[next][1] - eddies[prev][1] * eddies[next][0];
            
            // 3. Súrlódási veszteség
            let decay = 0.1;
            
            next_eddies[i][0] = eddies[i][0] + (fx + cx * 0.5 - decay * eddies[i][0]) * dt;
            next_eddies[i][1] = eddies[i][1] + (fy + cy * 0.5 - decay * eddies[i][1]) * dt;
            next_eddies[i][2] = eddies[i][2] + (fz + cz * 0.5 - decay * eddies[i][2]) * dt;
        }
        
        // Stabilitási limit (hogy a szimuláció ne szálljon el a végtelenbe)
        for i in 0..3 {
            let e = next_eddies[i][0].powi(2) + next_eddies[i][1].powi(2) + next_eddies[i][2].powi(2);
            if e > 5.0 {
                let scale = (5.0 / e).sqrt();
                next_eddies[i][0] *= scale;
                next_eddies[i][1] *= scale;
                next_eddies[i][2] *= scale;
            }
        }
        eddies = next_eddies;
        
        // Csak a szimuláció végén (amikor már "beállt" a turbulencia) mérjük az energiát
        if step >= 300 {
            for i in 0..3 {
                total_k += eddies[i][0].powi(2) + eddies[i][1].powi(2) + eddies[i][2].powi(2);
            }
        }
    }
    // Visszatérünk a stabilizálódott Átlagos Turbulens Kinetikus Energiával
    total_k / 200.0 
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
    
    let display_count = pareto_front.len().min(10);
    for (complexity, mse, ind) in pareto_front.into_iter().take(display_count) {
        let raw_eq = ind.to_string();
        let clean_eq = simplify_symengine(&raw_eq); 
        println!("{:<6} | {:<15.8} | {}", complexity, mse, clean_eq);
    }
    println!("========================================================\n");
}

fn main() {
    let mut rng = rand::rng();
    let n = 400; 

    let mut dx = Vec::with_capacity(n);
    let mut dy = Vec::with_capacity(n);

    println!("Running Virtual Wind Tunnel (Turbulence cascade)...");
    for _ in 0..n {
        // Generálunk egy véletlenszerű Makroszkopikus Sebesség Gradiens Mátrixot (3x3)
        let mut grad_u = [0.0; 9];
        for i in 0..9 {
            grad_u[i] = rng.random_range(-2.0..2.0);
        }

        // Lefuttatjuk a feketedoboz szimulátort
        let turbulent_energy = simulate_micro_turbulence(&grad_u);

        // Bemenet: Az 1 darab 3x3-as mátrix
        let mut row = Vec::with_capacity(9);
        row.extend_from_slice(&grad_u);
        
        dx.push(row);
        dy.push(turbulent_energy);
    }

    // Elmondjuk a motornak, hogy a bemenet egyetlen Mat3!
    let feature_types = vec![UniversalType::Mat3];

    run_open_problem_benchmark(
        "Navier-Stokes Algebraic Reynolds Stress Closure", 
        dx, 
        dy, 
        feature_types
    );
}