use std::time::Instant;

use rsr::api::*;
use rsr::prelude::*;

mod common;

fn load_mat2_csv(path: &str) -> (Vec<Vec<f32>>, Vec<[f32; 4]>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .expect("Nem sikerült beolvasni a CSV-t");

    let mut dx = Vec::new();
    let mut dy = Vec::new();

    for result in rdr.records() {
        if let Ok(record) = result {
            // F11, F12, F21, F22
            let f_vec = vec![
                record[0].parse().unwrap_or(0.0),
                record[1].parse().unwrap_or(0.0),
                record[2].parse().unwrap_or(0.0),
                record[3].parse().unwrap_or(0.0),
            ];
            
            // P11, P12, P21, P22
            let p_arr = [
                record[4].parse().unwrap_or(0.0),
                record[5].parse().unwrap_or(0.0),
                record[6].parse().unwrap_or(0.0),
                record[7].parse().unwrap_or(0.0),
            ];

            dx.push(f_vec);
            dy.push(p_arr);
        }
    }
    (dx, dy)
}

#[test]
fn test_law_2_experimental_foam() {
    let path = "test_data/CANN_foam/foam_F_to_P_2d.csv";
    if !std::path::Path::new(path).exists() {
        println!("Fájl nem található: {}. Futtasd le a Python konvertert!", path);
        return;
    }

    let (dx, dy) = load_mat2_csv(path);

    // Mivel 2x2-es mátrixok, Mat2 típust használunk
    let dataset = Dataset::new_mat2(&dx, &dy, vec![ValueType::Mat2, ValueType::Mat2]);
    
    // Alapértelmezett konstitutív profil, de módosítva Mat2-re!
    let mut config = Config::constitutive_tensor_law_mat2();
    config.target_type = ValueType::Mat2;
    // Engedélyezzük a konstans mátrixokat
    config.disabled_constant_types = vec![ValueType::Vec2, ValueType::Vec3]; 
    config.max_generations = 5000;
    config.island_size = 500;
    config.num_islands = 32;
    // A valós adatok zajosak, kicsit növeljük a mini batch méretet a stabilabb gradiensekért
    config.mini_batch_size = 128; 

    let regressor = SymbolicRegressor::new(config);

    println!(">>> RUNNING Experimental Foam Material Law Discovery <<<");
    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;
    
    common::update_history("Tensor_Foam", result.mse, time_ms);
    println!("MSE = {:.8}, Time = {}ms\nEquation: {}\n", result.mse, time_ms, result.equation);
}