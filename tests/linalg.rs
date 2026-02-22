mod common;

use rsr::Engine;
use rsr::SimdDataset;
use rsr::StaticStrategy;
use rsr::Strategy;
use rsr::{UniversalDomain, UniversalType};
use rsr::UniversalOp;
use rsr::engine::config::OpModule;
use rand::RngExt;
use std::time::Instant;

fn run_linalg_test(name: &str, category: &str, data_x: Vec<Vec<f32>>, data_y: Vec<f32>, feature_types: Vec<UniversalType>) {
    println!(">>> RUNNING {} <<<", name);
    let dataset = SimdDataset::new(&data_x, &data_y, feature_types, false);
    // Bekapcsoljuk a Basic ÉS a Linalg modult is!
    let mut config = common::get_test_config(vec![OpModule::Basic, OpModule::Linalg]);
    config.excluded_ops = vec![UniversalOp::SinF, UniversalOp::CosF, UniversalOp::ExpF, UniversalOp::SqrtF, UniversalOp::LnF, UniversalOp::SqrF];
    config.mutation_max_depth = 7;
    config.parsimony_penalty = 0.0;
    
    let strategy = StaticStrategy::new(config);
    let allowed_ops = strategy.get_allowed_operators();
    let var_registry = dataset.get_variable_registry();
    let mut engine = Engine::<StaticStrategy, UniversalDomain>::new(strategy, var_registry, allowed_ops);

    let start_time = Instant::now();
    engine.run_evolution(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    let best_ind = engine.get_global_best();
    let best_mse = best_ind.fitness;

    common::update_history(category, best_mse, time_ms);
    println!("Result {}: MSE = {}, Time = {}ms", category, best_mse, time_ms);
}

#[test]
fn linalg_1_distance_3d() {
    // Két pont (P1, P2) távolsága 3D-ben: ||P1 - P2||
    let mut rng = rand::rng(); let mut dx = Vec::new(); let mut dy = Vec::new();
    for _ in 0..400 { 
        let x0 = rng.random_range(-5.0..5.0); let x1 = rng.random_range(-5.0..5.0); let x2 = rng.random_range(-5.0..5.0);
        let x3: f32 = rng.random_range(-5.0..5.0); let x4 = rng.random_range(-5.0..5.0); let x5 = rng.random_range(-5.0..5.0);
        dx.push(vec![x0, x1, x2, x3, x4, x5]); 
        
        let dist = ((x0-x3).powi(2) + (x1-x4).powi(2) + (x2-x5).powi(2)).sqrt();
        dy.push(dist); 
    }
    run_linalg_test("Linalg 1: 3D Distance", "Linalg1", dx, dy, vec![UniversalType::Vec3, UniversalType::Vec3]);
}

#[test]
fn linalg_2_determinant_2x2() {
    // 2x2 Mátrix determinánsa (ad = bc)
    let mut rng = rand::rng(); let mut dx = Vec::new(); let mut dy = Vec::new();
    for _ in 0..400 { 
        let x0 = rng.random_range(-5.0..5.0); let x1 = rng.random_range(-5.0..5.0);
        let x2 = rng.random_range(-5.0..5.0); let x3 = rng.random_range(-5.0..5.0);
        dx.push(vec![x0, x1, x2, x3]); 
        
        dy.push(x0*x3 - x1*x2); 
    }
    run_linalg_test("Linalg 2: 2x2 Determinant", "Linalg2", dx, dy, vec![UniversalType::Mat2]);
}

#[test]
fn linalg_3_dot_product_2d() {
    // Két 2D vektor skaláris szorzata
    let mut rng = rand::rng(); let mut dx = Vec::new(); let mut dy = Vec::new();
    for _ in 0..400 { 
        let x0 = rng.random_range(-5.0..5.0); let x1 = rng.random_range(-5.0..5.0);
        let x2 = rng.random_range(-5.0..5.0); let x3 = rng.random_range(-5.0..5.0);
        dx.push(vec![x0, x1, x2, x3]); 
        
        dy.push(x0*x2 + x1*x3); 
    }
    run_linalg_test("Linalg 3: 2D Dot Product", "Linalg3", dx, dy, vec![UniversalType::Vec2, UniversalType::Vec2]);
}

#[test]
fn linalg_4_cross_product_norm_3d() {
    // Két 3D vektor vektoriális szorzatának hossza: || V1 x V2 ||
    let mut rng = rand::rng(); let mut dx = Vec::new(); let mut dy = Vec::new();
    for _ in 0..400 { 
        let x0 = rng.random_range(-5.0..5.0); let x1 = rng.random_range(-5.0..5.0); let x2 = rng.random_range(-5.0..5.0);
        let x3 = rng.random_range(-5.0..5.0); let x4 = rng.random_range(-5.0..5.0); let x5 = rng.random_range(-5.0..5.0);
        dx.push(vec![x0, x1, x2, x3, x4, x5]); 
        
        let cx: f32 = x1*x5 - x2*x4;
        let cy = x2*x3 - x0*x5;
        let cz = x0*x4 - x1*x3;
        dy.push((cx*cx + cy*cy + cz*cz).sqrt()); 
    }
    run_linalg_test("Linalg 4: 3D Cross Product Norm", "Linalg4", dx, dy, vec![UniversalType::Vec3, UniversalType::Vec3]);
}