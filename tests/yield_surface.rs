mod common;

use rand::RngExt;
use rsr::prelude::*;
use rsr::ValueType;
use std::time::Instant;

fn generate_sym_stress(rng: &mut impl RngExt) -> [f32; 9] {
    let s11 = rng.random_range(-100.0..100.0);
    let s22 = rng.random_range(-100.0..100.0);
    let s33 = rng.random_range(-100.0..100.0);
    let s12 = rng.random_range(-50.0..50.0);
    let s13 = rng.random_range(-50.0..50.0);
    let s23 = rng.random_range(-50.0..50.0);
    [s11, s12, s13, s12, s22, s23, s13, s23, s33]
}

fn run_yield_test(name: &str, data_x: Vec<Vec<f32>>, data_y: Vec<f32>) {
    println!(">>> RUNNING {} <<<", name);
    let dataset = Dataset::new(&data_x, &data_y, vec![ValueType::Mat3], false).with_scalar_extraction(false);
    
    // Folyási felület profil (csak feszültség-invariánsok)
    let config = Config::yield_surface(); 
    let regressor = SymbolicRegressor::new(config);

    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    println!("MSE = {:.8}, Time = {}ms\nEquation: {}\n", result.mse, time_ms, result.equation);
}

#[test]
fn test_yield_1_von_mises() {
    let mut rng = rand::rng();
    let (mut dx, mut dy) = (Vec::new(), Vec::new());
    let sigma_y = 250.0; // Folyáshatár [MPa]

    for _ in 0..500 {
        let sig = generate_sym_stress(&mut rng);
        dx.push(sig.to_vec());

        let p = (sig[0] + sig[4] + sig[8]) / 3.0; // Hidrosztatikus feszültség
        let mut s = sig; // Deviátor tenzor
        s[0] -= p; s[4] -= p; s[8] -= p;

        let j2 = 0.5 * (s[0]*s[0] + s[4]*s[4] + s[8]*s[8] + 2.0*(s[1]*s[1] + s[2]*s[2] + s[5]*s[5]));
        
        // Folyási feltétel: f = sqrt(3 * J2) - sigma_y
        dy.push((3.0 * j2).sqrt() - sigma_y);
    }
    run_yield_test("Yield Surface 1: von Mises Criterion (Metals)", dx, dy);
}

#[test]
fn test_yield_2_drucker_prager() {
    let mut rng = rand::rng();
    let (mut dx, mut dy) = (Vec::new(), Vec::new());
    let (alpha, k) = (0.2, 50.0);

    for _ in 0..500 {
        let sig = generate_sym_stress(&mut rng);
        dx.push(sig.to_vec());

        let i1 = sig[0] + sig[4] + sig[8];
        let p = i1 / 3.0;
        let mut s = sig;
        s[0] -= p; s[4] -= p; s[8] -= p;

        let j2 = 0.5 * (s[0]*s[0] + s[4]*s[4] + s[8]*s[8] + 2.0*(s[1]*s[1] + s[2]*s[2] + s[5]*s[5]));
        
        // Drucker-Prager: f = alpha * I1 + sqrt(J2) - k (Beton, talajmechanika)
        dy.push(alpha * i1 + j2.sqrt() - k);
    }
    run_yield_test("Yield Surface 2: Drucker-Prager (Geotech/Concrete)", dx, dy);
}

#[test]
fn test_yield_3_bresler_pister() {
    let mut rng = rand::rng();
    let (mut dx, mut dy) = (Vec::new(), Vec::new());
    let (a, b, c) = (0.005, 0.1, 1.2);

    for _ in 0..800 {
        let sig = generate_sym_stress(&mut rng);
        dx.push(sig.to_vec());

        let i1 = sig[0] + sig[4] + sig[8];
        let p = i1 / 3.0;
        let mut s = sig;
        s[0] -= p; s[4] -= p; s[8] -= p;

        let j2 = 0.5 * (s[0]*s[0] + s[4]*s[4] + s[8]*s[8] + 2.0*(s[1]*s[1] + s[2]*s[2] + s[5]*s[5]));
        
        // Bresler-Pister approximáció polimer habokra: f = a*I1^2 + b*I1 + c*J2 - k
        dy.push(a * i1.powi(2) + b * i1 + c * j2 - 100.0);
    }
    run_yield_test("Yield Surface 3: Bresler-Pister Poly (Lattice/Foams)", dx, dy);
}