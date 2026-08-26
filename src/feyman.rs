use std::{time::Instant, vec};

use rsr::prelude::*;

pub fn test() {
    // 6 bemenet: az első 3 az első Vec3, a második 3 a második Vec3
    let schema = Schema::new(vec![
        ValueType::Float,
        ValueType::Float,
        ValueType::Float,
        ValueType::Float,
    ])
    .with_normalization(false); // Fontos kikapcsolni a tiszta fizikai képlethez

    let dataset = Dataset::from_csv("C:\\Users\\darua\\Documents\\rust_learning\\feyman_test\\srsd-benchmark\\temp_out\\feynman-i.29.16.txt.csv", &schema)
        .expect("Nem sikerült beolvasni a CSV-t");

    // LinalgDomain bekapcsolva, felesleges műveletek (pl. Sin, Cos) nélkül
    let config = Config::default(vec![OpModule::Basic, OpModule::Linalg]).parsimony_penalty(0.00005);
    let regressor = SymbolicRegressor::new(config)
        .train_subset_size(400)
        .generations(5000);

    println!(">>> Keresés indítása a Feynman I.29.16 (3D távolság) adathalmazon...");
    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let duration = start_time.elapsed();

    println!("\nEquation found in {:.3} seconds!", duration.as_secs_f64());
    println!("Equation: {}", result.equation);
    println!("Legjobb egyenlet: {}", result.equation);
    println!("MSE: {}", result.mse);
    println!("Komplexitás: {}", result.complexity);
}
