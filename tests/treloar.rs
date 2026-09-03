mod common;

use std::path::Path;
use std::time::Instant;
use rsr::prelude::*;

fn run_treloar_test(
    name: &str,
    category: &str,
    data_x: Vec<Vec<f32>>,
    data_y: Vec<f32>,
) {
    println!(">>> RUNNING {} <<<", name);
    let dataset = Dataset::new(&data_x, &data_y, vec![ValueType::Mat3], false).with_scalar_extraction(false);

    let mut config = Config::hyperelastic_energy();
    // Fontos: Itt biztosítsd, hogy a config tartalmazza az 'exp', 'sqrt' és 'log' függvényeket!
    config.max_generations = 5000;
    config.island_size = 500;
    config.num_islands = 32;
    // A subset_size-t megnöveltük, mivel most 3 adatsor pontjait (köztük duplikáltakat) tartalmazza
    config.subset_size = Some(data_x.len().min(400));

    let regressor = SymbolicRegressor::new(config);

    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    common::update_history(category, result.mse, time_ms);
    println!(
        "Result {}: MSE = {:.6}, Time = {}ms\nEquation: {}\n",
        category, result.mse, time_ms, result.equation
    );
}

fn load_treloar_csv(path: &str, mode: &str) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .unwrap_or_else(|_| panic!("Hiba a CSV betoltesekor: {}", path));

    let mut lambdas = Vec::new();
    let mut stresses = Vec::new();

    for result in rdr.records() {
        let record = result.expect("CSV olvasasi hiba");
        lambdas.push(record[0].parse::<f32>().unwrap());
        stresses.push(record[1].parse::<f32>().unwrap());
    }

    let mut dx = Vec::with_capacity(lambdas.len());
    let mut dy = Vec::with_capacity(lambdas.len());
    let mut current_w = 0.0;

    for i in 0..lambdas.len() {
        let lam = lambdas[i];
        
        if i > 0 {
            let d_lam = lam - lambdas[i - 1];
            let avg_p = (stresses[i] + stresses[i - 1]) / 2.0;
            
            // Kéttengelyű húzásnál (equibiaxial) mindkét főirányban munkát végzünk (2 * P * d_lam)
            if mode == "biaxial" {
                current_w += 2.0 * avg_p * d_lam;
            } else {
                current_w += avg_p * d_lam;
            }
        }

        let f_matrix = match mode {
            "uniaxial" => vec![
                lam, 0.0, 0.0,
                0.0, 1.0 / lam.sqrt(), 0.0,
                0.0, 0.0, 1.0 / lam.sqrt()
            ],
            "biaxial" => vec![
                lam, 0.0, 0.0,
                0.0, lam, 0.0,
                0.0, 0.0, 1.0 / (lam * lam)
            ],
            "shear" => vec![
                lam, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0 / lam
            ],
            _ => panic!("Ismeretlen terhelesi mod!"),
        };
        
        dx.push(f_matrix);
        dy.push(current_w);
    }
    (dx, dy)
}

#[test]
fn treloar_combined_test() {
    let path_uni = "test_data/Treloar/Treloar_uniaxial.csv";
    let path_bi = "test_data/Treloar/Treloar_equibiaxial.csv";
    let path_shear = "test_data/Treloar/Treloar_pureshear.csv";

    if !Path::new(path_uni).exists() || !Path::new(path_bi).exists() || !Path::new(path_shear).exists() { 
        return; 
    }

    let (mut dx_uni, mut dy_uni) = load_treloar_csv(path_uni, "uniaxial");
    let (mut dx_bi, mut dy_bi) = load_treloar_csv(path_bi, "biaxial");
    let (mut dx_shear, mut dy_shear) = load_treloar_csv(path_shear, "shear");

    // Összefűzzük a bemeneteket egyetlen nagy adatsorrá
    let mut dx_combined = Vec::new();
    let mut dy_combined = Vec::new();

    // 1. Uniaxial adatok
    dx_combined.append(&mut dx_uni);
    dy_combined.append(&mut dy_uni);

    // 2. Pure Shear adatok
    dx_combined.append(&mut dx_shear);
    dy_combined.append(&mut dy_shear);

    // 3. Equibiaxial adatok (Duplikálva a súlyozás kiegyenlítése végett a cikk alapján)
    dx_combined.extend(dx_bi.clone());
    dy_combined.extend(dy_bi.clone());
    dx_combined.append(&mut dx_bi);
    dy_combined.append(&mut dy_bi);

    run_treloar_test(
        "Treloar: Combined Simultaneous Fit (Uni, Shear, Bi weighted)", 
        "TreloarCombined", 
        dx_combined, 
        dy_combined
    );
}