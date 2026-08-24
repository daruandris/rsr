mod common;

use rand::RngExt;
use rsr::prelude::*;
use rsr::engine::search::strategy::{Strategy,StaticStrategy};
use rsr::engine::search::engine::Engine;
use std::time::Instant;

fn run_test(
    name: &str,
    category: &str,
    data_x: Vec<Vec<f32>>,
    data_y: Vec<f32>,
    num_features: u8,
    normalize: bool,
) {
    println!(">>> RUNNING {} <<<", name);
    let feature_types = vec![ValueType::Float; num_features as usize];
    let dataset = Dataset::new(&data_x, &data_y, feature_types, normalize);
    let config = common::get_test_config(vec![OpModule::Basic]);

    let strategy = StaticStrategy::new(config);
    let allowed_ops = strategy.get_allowed_operators();
    let var_registry = dataset.get_variable_registry();
    let mut engine = Engine::new(strategy, var_registry, allowed_ops);

    let start_time = Instant::now();
    engine.run(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    let best_ind = engine.get_global_best();
    let best_mse = best_ind.fitness;

    common::update_history(category, best_mse, time_ms);
    println!(
        "Result {}: MSE = {}, Time = {}ms",
        category, best_mse, time_ms
    );
}

#[test]
fn basic_1_simple_square() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let x = rng.random_range(-5.0..5.0);
        dx.push(vec![x]);
        dy.push(2.5 * x * x - 1.2);
    }
    run_test("Basic 1: Sqr", "Basic1", dx, dy, 1, false);
}

#[test]
fn basic_2_trigonometry() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let x: f32 = rng.random_range(-4.0..4.0);
        dx.push(vec![x]);
        dy.push(3.0 * (2.0 * x).cos() + 1.0);
    }
    run_test("Basic 2: Cos", "Basic2", dx, dy, 1, false);
}

#[test]
fn basic_3_exponential() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let x: f32 = rng.random_range(-2.0..4.0);
        dx.push(vec![x]);
        dy.push(1.5 * (0.5 * x).exp());
    }
    run_test("Basic 3: Exp", "Basic3", dx, dy, 1, false);
}

#[test]
fn basic_4_damped_oscillator() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let x: f32 = rng.random_range(0.0..10.0);
        dx.push(vec![x]);
        dy.push((-0.5 * x).exp() * (3.0 * x).cos());
    }
    run_test("Basic 4: Damped Osc.", "Basic4", dx, dy, 1, false);
}

#[test]
fn basic_5_multivariable_linear() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let x0 = rng.random_range(-5.0..5.0);
        let x1 = rng.random_range(-5.0..5.0);
        let x2 = rng.random_range(-5.0..5.0);
        dx.push(vec![x0, x1, x2]);
        dy.push(2.0 * x0 - 3.5 * x1 + 1.2 * x2);
    }
    run_test("Basic 5: Linear 3D", "Basic5", dx, dy, 3, false);
}

#[test]
fn basic_6_complex_multivariable() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let x0 = rng.random_range(-3.0..3.0);
        let x1: f32 = rng.random_range(-4.0..4.0);
        let x2 = rng.random_range(-5.0..5.0);
        dx.push(vec![x0, x1, x2]);
        dy.push(x0 * x0 + x1.sin() - x2);
    }
    run_test("Basic 6: Complex 3D", "Basic6", dx, dy, 3, false);
}

#[test]
fn basic_7_noisy_data() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let x = rng.random_range(-10.0..10.0);
        let noise = rng.random_range(-0.001..0.001);
        dx.push(vec![x]);
        dy.push(2.5 * x * x + noise);
    }
    run_test("Basic 7: Noisy", "Basic7", dx, dy, 1, false);
}

#[test]
fn basic_8_feature_selection() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let x0 = rng.random_range(-5.0..5.0);
        let x1 = rng.random_range(-5.0..5.0);
        let x2 = rng.random_range(-5.0..5.0);
        let x3 = rng.random_range(-5.0..5.0);
        let x4 = rng.random_range(-5.0..5.0);
        dx.push(vec![x0, x1, x2, x3, x4]);
        dy.push(x0 * x3);
    }
    run_test("Basic 8: Hidden Dims", "Basic8", dx, dy, 5, false);
}

#[test]
fn basic_9_rational_function() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let x = rng.random_range(-5.0..5.0);
        dx.push(vec![x]);
        dy.push((x + 1.5) / (x * x + 2.0));
    }
    run_test("Basic 9: Division", "Basic9", dx, dy, 1, false);
}

#[test]
fn basic_10_ultimate() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let x0: f32 = rng.random_range(-3.0..3.0);
        let x1: f32 = rng.random_range(-5.0..5.0);
        let x2 = rng.random_range(-5.0..5.0);
        let x3 = rng.random_range(0.0..5.0);
        dx.push(vec![x0, x1, x2, x3]);
        dy.push((-x0).exp() + x1.cos() - ((x2 * x2) / (x3 + 1.1)));
    }
    run_test("Basic 10: All-Ops", "Basic10", dx, dy, 4, false);
}
