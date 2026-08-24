mod common;

use rand::RngExt;
use rsr::Instruction;
use rsr::domains::basic::BasicOpCode;
use rsr::engine::search::engine::Engine;
use rsr::engine::search::strategy::{StaticStrategy, Strategy};
use rsr::prelude::*;
use std::time::Instant;

fn run_linalg_test(
    name: &str,
    category: &str,
    data_x: Vec<Vec<f32>>,
    data_y: Vec<f32>,
    feature_types: Vec<ValueType>,
) {
    println!(">>> RUNNING {} <<<", name);
    let dataset = Dataset::new(&data_x, &data_y, feature_types, false);

    let mut config = common::get_test_config(vec![OpModule::Basic, OpModule::Linalg]);
    config.excluded_ops = vec![
        Instruction::Basic(BasicOpCode::SinF),
        Instruction::Basic(BasicOpCode::CosF),
        Instruction::Basic(BasicOpCode::ExpF),
        Instruction::Basic(BasicOpCode::SqrtF),
        Instruction::Basic(BasicOpCode::LnF),
        Instruction::Basic(BasicOpCode::SqrF),
    ];
    config.mutation_max_depth = 7;
    config.parsimony_penalty = 0.0;

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
fn linalg_1_rigid_body_energy() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();

    let i_mat = [2.0, 0.1, 0.0, 0.1, 1.5, -0.2, 0.0, -0.2, 1.0];

    for _ in 0..400 {
        let w0 = rng.random_range(-5.0..5.0);
        let w1 = rng.random_range(-5.0..5.0);
        let w2 = rng.random_range(-5.0..5.0);
        dx.push(vec![w0, w1, w2]);

        let iw0 = i_mat[0] * w0 + i_mat[3] * w1 + i_mat[6] * w2;
        let iw1 = i_mat[1] * w0 + i_mat[4] * w1 + i_mat[7] * w2;
        let iw2 = i_mat[2] * w0 + i_mat[5] * w1 + i_mat[8] * w2;

        dy.push(w0 * iw0 + w1 * iw1 + w2 * iw2);
    }
    run_linalg_test(
        "Linalg 1: CMA-ES Mat3 Discovery",
        "Linalg1",
        dx,
        dy,
        vec![ValueType::Vec3],
    );
}

#[test]
fn linalg_2_lorentz_force() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let e0 = rng.random_range(-5.0..5.0);
        let e1 = rng.random_range(-5.0..5.0);
        let e2 = rng.random_range(-5.0..5.0);
        let v0 = rng.random_range(-5.0..5.0);
        let v1 = rng.random_range(-5.0..5.0);
        let v2 = rng.random_range(-5.0..5.0);
        let b0 = rng.random_range(-5.0..5.0);
        let b1 = rng.random_range(-5.0..5.0);
        let b2 = rng.random_range(-5.0..5.0);
        dx.push(vec![e0, e1, e2, v0, v1, v2, b0, b1, b2]);

        let cx = v1 * b2 - v2 * b1;
        let cy = v2 * b0 - v0 * b2;
        let cz = v0 * b1 - v1 * b0;

        let fx = e0 + cx;
        let fy = e1 + cy;
        let fz: f32 = e2 + cz;

        dy.push((fx * fx + fy * fy + fz * fz).sqrt());
    }
    run_linalg_test(
        "Linalg 2: Lorentz Force",
        "Linalg2",
        dx,
        dy,
        vec![ValueType::Vec3, ValueType::Vec3, ValueType::Vec3],
    );
}

#[test]
fn linalg_3_matrix_inverse_trace() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let a0 = rng.random_range(1.0..8.0);
        let a1 = rng.random_range(-4.0..4.0);
        let a2 = rng.random_range(-1.0..1.0);
        let a3 = rng.random_range(1.0..8.0);
        let det_a = a0 * a3 - a1 * a2;

        let b0 = rng.random_range(-6.0..2.0);
        let b1 = rng.random_range(-2.0..6.0);
        let b2 = rng.random_range(-2.0..2.0);
        let b3 = rng.random_range(-10.0..0.0);

        dx.push(vec![a0, a1, a2, a3, b0, b1, b2, b3]);

        let inv_a = [a3 / det_a, -a1 / det_a, -a2 / det_a, a0 / det_a];
        let ab0 = inv_a[0] * b0 + inv_a[2] * b1;
        let ab3 = inv_a[1] * b2 + inv_a[3] * b3;

        let trace_ab = ab0 + ab3;
        dy.push(trace_ab + det_a);
    }
    run_linalg_test(
        "Linalg 3: Inverse & Trace",
        "Linalg3",
        dx,
        dy,
        vec![ValueType::Mat2, ValueType::Mat2],
    );
}

#[test]
fn linalg_4_transform_error() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    for _ in 0..400 {
        let mut row = Vec::new();
        let mut m = [0.0; 9];
        for item in &mut m {
            *item = rng.random_range(-5.0..5.0);
            row.push(*item);
        }

        let mut v = [0.0; 3];
        for item in &mut v {
            *item = rng.random_range(-5.0..5.0);
            row.push(*item);
        }

        let mut u = [0.0; 3];
        for item in &mut u {
            *item = rng.random_range(-5.0..5.0);
            row.push(*item);
        }

        dx.push(row);

        let mv0 = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
        let mv1 = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
        let mv2 = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];

        let d0 = mv0 - u[0];
        let d1: f32 = mv1 - u[1];
        let d2 = mv2 - u[2];

        dy.push((d0 * d0 + d1 * d1 + d2 * d2).sqrt());
    }
    run_linalg_test(
        "Linalg 4: 3D Transform Error",
        "Linalg4",
        dx,
        dy,
        vec![ValueType::Mat3, ValueType::Vec3, ValueType::Vec3],
    );
}
