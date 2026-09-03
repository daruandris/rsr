use std::path::Path;
use std::time::Instant;
use rsr::prelude::*;

mod common;

fn load_yield_surface_csv(path: &str) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap_or_else(|_| panic!("Hiba a CSV betöltésekor: {}", path));

    let mut dx = Vec::new();
    let mut dy = Vec::new();

    for result in rdr.records() {
        if let Ok(record) = result {
            let sig11: f32 = record[0].parse().unwrap_or(0.0);
            let sig22: f32 = record[1].parse().unwrap_or(0.0);
            let sig12: f32 = record[2].parse().unwrap_or(0.0);

            let stress_base = vec![
                sig11, sig12, 0.0,
                sig12, sig22, 0.0,
                0.0,   0.0,   0.0,
            ];
            
            let scales = [0.1, 0.5, 1.0, 1.5, 2.0];
            
            for &k in &scales {
                let mut scaled_stress = stress_base.clone();
                for i in 0..9 {
                    scaled_stress[i] *= k;
                }
                
                dx.push(scaled_stress);
                dy.push(k * k);
            }
        }
    }
    
    (dx, dy)
}

#[test]
fn yield_surface_1_camclay() {
    let path = "test_data/cam_clay/yield_stress.csv";
    
    if !Path::new(path).exists() { 
        println!("A fájl nem található: {}", path);
        return; 
    }
    
    let (dx, dy) = load_yield_surface_csv(path);
    println!(">>> RUNNING Yield Surface (CamClay) <<<");
    
    let dataset = Dataset::new(&dx, &dy, vec![ValueType::Mat3], false)
        .with_scalar_extraction(false);

    let mut config = Config::yield_surface();
    
    config.max_generations = 5000;
    config.island_size = 500;
    config.num_islands = 32;
    config.subset_size = Some(dx.len().min(400));

    let regressor = SymbolicRegressor::new(config);

    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    common::update_history("Yield_CamClay", result.mse, time_ms);
    println!(
        "Result Yield_CamClay: MSE = {:.6}, Time = {}ms\nEquation: {}\n",
        result.mse, time_ms, result.equation
    );
}