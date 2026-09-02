mod common;

use std::path::Path;
use std::time::Instant;
use rsr::prelude::*;

/// A szívizom adatok beolvasása: 1D nyírásból építünk fel 3D tenzorokat
fn load_heart_data(path: &str) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false) // Nincs fejléc a kimentett CSV-ben
        .from_path(path)
        .unwrap();

    let mut dx = Vec::new();
    let mut dy = Vec::new();

    for result in rdr.records() {
        if let Ok(record) = result {
            let gamma: f32 = record[0].parse().unwrap();
            let tau: f32 = record[1].parse().unwrap();

            // 1. Bemenet (X0_Mat3): Deformációs gradiens (Egyszerű nyírás)
            let mut f_matrix = vec![
                1.0, gamma, 0.0,
                0.0, 1.0,   0.0,
                0.0, 0.0,   1.0
            ];

            // 2. Bemenet (X1_Mat3): Strukturális tenzor (M = a0 x a0). 
            // A szívizomnál a rostok jellemzően az X tengely mentén (fiber direction) állnak.
            let m_matrix = vec![
                1.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0
            ];

            // A RSR motor lapos bemenetet vár, összefűzzük a két 9 elemű mátrixot (18 elem összesen)
            f_matrix.extend(m_matrix);
            dx.push(f_matrix);

            // Célváltozó (Y_Mat3): Névleges (Piola-Kirchhoff) feszültségtenzor
            // A mért feszültség (tau) a nyírási síkban ébred.
            let p_matrix = vec![
                0.0, tau, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0
            ];
            dy.push(p_matrix);
        }
    }
    (dx, dy)
}

fn run_tensor_test(name: &str, category: &str, dx: Vec<Vec<f32>>, dy: Vec<Vec<f32>>, input_types: Vec<ValueType>) {
    println!(">>> RUNNING {} <<<", name);

    // Mátrix kimenetet váró Dataset (A 'dy' itt most Vec<Vec<f32>>, ami majd 9 komponenst tartalmaz)
    let dataset = Dataset::new(&dx, &dy, input_types, false)
        .with_scalar_extraction(false);

    // Behúzzuk a kifejezetten tenzor-számításra kihegyezett ipari profilt!
    let mut config = Config::constitutive_tensor_law();
    
    config.max_generations = 5000;
    config.num_islands = 32;
    config.island_size = 100;
    config.subset_size = Some(dx.len());
    
    let regressor = SymbolicRegressor::new(config);

    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    println!(
        "Result {}: MSE = {:.8}, Time = {}ms\nEquation: {}\n",
        category, result.mse, time_ms, result.equation
    );
}

#[test]
fn constitutive_1_heart_shear() {
    let path = "test_data/heart_shear.csv";
    if Path::new(path).exists() {
        let (dx, dy) = load_heart_data(path);
        
        // Fontos: Két bemeneti mátrixot adunk meg a sémában! (F és M)
        run_tensor_test(
            "Constitutive 1: Heart Anisotropic Shear", 
            "HeartShear", 
            dx, 
            dy, 
            vec![ValueType::Mat3, ValueType::Mat3]
        );
    } else {
        println!("A heart_shear.csv nem talalhato.");
    }
}