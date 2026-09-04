// tests/constitutive_law.rs
mod common;

use rand::RngExt;
use rsr::prelude::*;
use std::time::Instant;

fn generate_random_f(rng: &mut impl RngExt) -> [f32; 9] {
    // F = I + Grad(u). Kisebb deformációk, hogy a modell stabil maradjon
    [
        1.0 + rng.random_range(-0.1..0.1), rng.random_range(-0.1..0.1), rng.random_range(-0.1..0.1),
        rng.random_range(-0.1..0.1), 1.0 + rng.random_range(-0.1..0.1), rng.random_range(-0.1..0.1),
        rng.random_range(-0.1..0.1), rng.random_range(-0.1..0.1), 1.0 + rng.random_range(-0.1..0.1)
    ]
}

#[test]
fn test_law_1_st_venant_kirchhoff() {
    let mut rng = rand::rng();
    let mut dx: Vec<Vec<f32>> = Vec::new();
    let mut dy: Vec<[f32; 9]> = Vec::new(); // JAVÍTVA: Fix méretű tömbök listája
    
    // Anyagállandók
    let lambda = 120.0; 
    let mu = 80.0;

    for _ in 0..400 {
        let f = generate_random_f(&mut rng);
        dx.push(f.to_vec()); // Bemenet maradhat Vec<f32> a rugalmas sémakezelés miatt

        // C = F^T * F
        let c = [
            f[0]*f[0]+f[3]*f[3]+f[6]*f[6], f[0]*f[1]+f[3]*f[4]+f[6]*f[7], f[0]*f[2]+f[3]*f[5]+f[6]*f[8],
            f[1]*f[0]+f[4]*f[3]+f[7]*f[6], f[1]*f[1]+f[4]*f[4]+f[7]*f[7], f[1]*f[2]+f[4]*f[5]+f[7]*f[8],
            f[2]*f[0]+f[5]*f[3]+f[8]*f[6], f[2]*f[1]+f[5]*f[4]+f[8]*f[7], f[2]*f[2]+f[5]*f[5]+f[8]*f[8]
        ];

        // E = 0.5 * (C - I)
        let mut e = c;
        e[0] -= 1.0; e[4] -= 1.0; e[8] -= 1.0;
        for val in &mut e { *val *= 0.5; }

        let tr_e = e[0] + e[4] + e[8];
        let mut s = [0.0; 9];
        
        // S = lambda * tr(E) * I + 2 * mu * E
        for i in 0..9 { s[i] = 2.0 * mu * e[i]; }
        s[0] += lambda * tr_e;
        s[4] += lambda * tr_e;
        s[8] += lambda * tr_e;

        // JAVÍTVA: Közvetlenül az [f32; 9] tömböt pusholjuk
        dy.push(s);
    }

    let dataset = Dataset::new_mat3(&dx, &dy, vec![ValueType::Mat3, ValueType::Mat3]);
    
    let mut config = Config::constitutive_tensor_law_mat3();
    config.max_generations = 5000;
    config.island_size = 500;
    config.num_islands = 32;
    
    let regressor = SymbolicRegressor::new(config);

    println!(">>> RUNNING St. Venant-Kirchhoff Tensor Law Discovery <<<");
    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;
    
    common::update_history("Tensor_StVenant", result.mse, time_ms);
    println!("MSE = {:.8}, Time = {}ms\nEquation: {}\n", result.mse, time_ms, result.equation);
}