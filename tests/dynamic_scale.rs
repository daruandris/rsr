mod common;

use rand::RngExt;
use rsr::prelude::*;
use std::time::Instant;

fn run_dynamic_test(
    name: &str,
    category: &str,
    data_x: Vec<Vec<f32>>,
    data_y: Vec<f32>,
    num_features: u8,
    normalize: bool,
) {
    println!(">>> RUNNING {} <<<", name);
    let feature_types = vec![ValueType::Float; num_features as usize];
    
    // Itt hívjuk a Dataset::new-t, ami már beépítve számolja a target_variance-t
    let dataset = Dataset::new(&data_x, &data_y, feature_types, normalize);

    // Alapértelmezett beállítások, melyek az új bázis értékeket használják (1e-7 és 5e-6)
    let mut config = common::get_test_config(vec![OpModule::Basic]);
    config.max_generations = 2000;
    config.subset_size = Some(200);

    let regressor = SymbolicRegressor::new(config);

    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    println!(
        "Eredmeny [{}]:\n Adat Variancia = {:.2e}\n Elert MSE = {:.2e}\n Komplexitas = {}\n Ido = {}ms\n Egyenlet: {}\n",
        category, dataset.target_variance, result.mse, result.complexity, time_ms, result.equation
    );
}

#[test]
fn dyn_1_constant_target() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    // A cél egy fix érték. A variancia 0 (amit a motor 1.0-ra kell állítson a védelem miatt).
    for _ in 0..400 {
        let x = rng.random_range(-10.0..10.0);
        dx.push(vec![x]);
        dy.push(42.5);
    }
    run_dynamic_test("Dynamic 1: Constant Target (Var = 0 fallback)", "Dyn1", dx, dy, 1, false);
}

#[test]
fn dyn_2_micro_scale() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    // Mikroszkopikus skála. Korábban itt elnyomta a büntetés a gradienst.
    for _ in 0..400 {
        let x = rng.random_range(-0.0001..0.0001);
        dx.push(vec![x]);
        dy.push(2.5 * x - 1.2e-5); 
    }
    run_dynamic_test("Dynamic 2: Micro Scale [-1e-4, 1e-4]", "Dyn2", dx, dy, 1, false);
}

#[test]
fn dyn_3_macro_scale() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    // Nagyméretű skála. A dinamikus büntetésnek elég nagynak kell lennie, hogy működjön.
    for _ in 0..400 {
        let x = rng.random_range(-1_000_000.0..1_000_000.0);
        dx.push(vec![x]);
        dy.push(3.14 * x + 500_000.0);
    }
    run_dynamic_test("Dynamic 3: Macro Scale [-1e6, 1e6]", "Dyn3", dx, dy, 1, false);
}

#[test]
fn dyn_4_galactic_scale() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    // Extrém nagy skála, ahol az f32 pontossága már határeset. 
    for _ in 0..400 {
        let x = rng.random_range(-1e9..1e9);
        dx.push(vec![x]);
        dy.push(0.5 * x);
    }
    run_dynamic_test("Dynamic 4: Galactic Scale [-1e9, 1e9]", "Dyn4", dx, dy, 1, false);
}

#[test]
fn dyn_5_pure_noise() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    // Nincs összefüggés. A dinamikus büntetésnek le kell vágnia az ágakat, 
    // és egy szimpla konstanst (az átlagot) kell visszaadnia.
    for _ in 0..400 {
        let x = rng.random_range(-10.0..10.0);
        dx.push(vec![x]);
        dy.push(rng.random_range(-100.0..100.0));
    }
    run_dynamic_test("Dynamic 5: Pure Noise (Should collapse to constant)", "Dyn5", dx, dy, 1, false);
}

#[test]
fn dyn_6_heavy_noise_linear() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    // Erős zaj. Meg kell találnia az alapegyenletet anélkül, hogy túltanulná a zajt.
    for _ in 0..400 {
        let x = rng.random_range(-10.0..10.0);
        let noise = rng.random_range(-5.0..5.0); // 10% - 50% zaj
        dx.push(vec![x]);
        dy.push(10.0 * x + noise);
    }
    run_dynamic_test("Dynamic 6: Heavy Noise Linear", "Dyn6", dx, dy, 1, false);
}

#[test]
fn dyn_7_mixed_dims_normalized() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    // Kevert dimenziók, de itt BEKAPCSOLJUK a normalizációt.
    // A Warn log nem fog megjelenni, és a modellnek simán meg kell találnia a megoldást.
    for _ in 0..400 {
        let x0 = rng.random_range(-1.0..1.0);           
        let x1 = rng.random_range(1_000_000.0..5_000_000.0);      
        dx.push(vec![x0, x1]);
        dy.push(2.0 * x0 + 0.0001 * x1);
    }
    run_dynamic_test("Dynamic 7: Mixed Dims (Normalized = true)", "Dyn7", dx, dy, 2, true);
}

#[test]
fn dyn_8_complex_large_scale() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    // Magasabb fokú polinom makro skálán.
    for _ in 0..400 {
        let x = rng.random_range(-1000.0..1000.0);
        dx.push(vec![x]);
        dy.push(2.5 * x * x - 500.0 * x + 10000.0);
    }
    run_dynamic_test("Dynamic 8: Complex Large Scale", "Dyn8", dx, dy, 1, false);
}