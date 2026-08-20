use rand::RngExt;
use std::time::Instant;

use rsr::api::SymbolicRegressor;
use rsr::engine::data::dataset::Dataset;
use rsr::engine::data::schema::Schema;
use rsr::prelude::*;
use rsr::domains::basic::BasicOpCode;

fn main() {
    let mut rng = rand::rng();
    let num_samples = 400;

    let mut dx = Vec::with_capacity(num_samples);
    let mut dy = Vec::with_capacity(num_samples);
    for _ in 0..num_samples {
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

    let schema = Schema::new(vec![ValueType::Vec3, ValueType::Vec3, ValueType::Vec3])
        .with_normalization(false);
    
    let dataset = Dataset::from_arrays(&dx, &dy, &schema);
    let config = Config::default(vec![OpModule::Basic, OpModule::Linalg]).without_ops(vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
            Instruction::Basic(BasicOpCode::ExpF),
            Instruction::Basic(BasicOpCode::LnF),
            Instruction::Basic(BasicOpCode::SqrtF),
            Instruction::Basic(BasicOpCode::SqrF),
        ]);

    println!("Starting the algorithm...");
    
    let regressor = SymbolicRegressor { config };

    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let duration = start_time.elapsed();

    println!("\nEquation found in {:.3} seconds!", duration.as_secs_f64());
    println!("Equation: {}", result.equation);
    println!("MSE:    {:.10}", result.mse);
}