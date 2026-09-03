/* 
mod common;

use rand::RngExt;
use rsr::prelude::*;
use rsr::ValueType;
use std::time::Instant;

fn generate_sym_strain(rng: &mut impl RngExt) -> [f32; 9] {
    let e11 = rng.random_range(-0.1..0.1);
    let e22 = rng.random_range(-0.1..0.1);
    let e33 = rng.random_range(-0.1..0.1);
    let e12 = rng.random_range(-0.05..0.05);
    let e13 = rng.random_range(-0.05..0.05);
    let e23 = rng.random_range(-0.05..0.05);
    [e11, e12, e13, e12, e22, e23, e13, e23, e33]
}

/// Speciális tesztfuttató, ahol a kimenet (Y) is egy 3x3-as mátrix
fn run_tensor_law_test(name: &str, data_x: Vec<Vec<f32>>, data_y: Vec<Vec<f32>>) {
    println!(">>> RUNNING {} <<<", name);
    
    // Bemenet: Mat3 (E alakváltozás), Kimenet: Mat3 (Sigma feszültség)
    let dataset = Dataset::new_tensor(&data_x, &data_y, vec![ValueType::Mat3], ValueType::Mat3);
    
    // Teljes mátrix aritmetikát engedélyező profil[cite: 1]
    let config = Config::constitutive_tensor_law(); 
    let regressor = SymbolicRegressor::new(config);

    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    println!("MSE = {:.8}, Time = {}ms\nEquation: {}\n", result.mse, time_ms, result.equation);
}

#[test]
fn test_law_1_isotropic_hooke() {
    let mut rng = rand::rng();
    let (mut dx, mut dy) = (Vec::new(), Vec::new());
    let (lambda, mu) = (120.0, 80.0); // Lame állandók

    for _ in 0..500 {
        let e = generate_sym_strain(&mut rng);
        dx.push(e.to_vec());

        let tr_e = e[0] + e[4] + e[8];
        let mut sigma = [0.0; 9];
        
        // Hooke törvény: Sigma = lambda * tr(E) * I + 2 * mu * E
        for i in 0..9 {
            sigma[i] = 2.0 * mu * e[i];
        }
        sigma[0] += lambda * tr_e;
        sigma[4] += lambda * tr_e;
        sigma[8] += lambda * tr_e;

        dy.push(sigma.to_vec());
    }
    run_tensor_law_test("Constitutive Law 1: Isotropic Hooke's Law (Matrix out)", dx, dy);
}

#[test]
fn test_law_2_deviatoric_stress() {
    let mut rng = rand::rng();
    let (mut dx, mut dy) = (Vec::new(), Vec::new());
    let mu = 80.0;

    for _ in 0..300 {
        let e = generate_sym_strain(&mut rng);
        dx.push(e.to_vec());

        let tr_e = e[0] + e[4] + e[8];
        let p_e = tr_e / 3.0;
        let mut sigma_dev = [0.0; 9];
        
        // Csak torzulás (deviátoros rész): S = 2 * mu * E_dev
        for i in 0..9 {
            sigma_dev[i] = 2.0 * mu * e[i];
        }
        sigma_dev[0] -= 2.0 * mu * p_e;
        sigma_dev[4] -= 2.0 * mu * p_e;
        sigma_dev[8] -= 2.0 * mu * p_e;

        dy.push(sigma_dev.to_vec());
    }
    run_tensor_law_test("Constitutive Law 2: Pure Deviatoric Stress", dx, dy);
}
*/