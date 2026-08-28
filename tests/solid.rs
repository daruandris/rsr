// tests/solid.rs
mod common;

use rand::RngExt;
use rsr::Instruction;
use rsr::domains::basic::BasicOpCode;
use rsr::prelude::*;
use std::time::Instant;

/// Segédfüggvény fizikailag reális Deformációs Grádiens (F) generálásához.
/// Az identitás körüli perturbáció garantálja a det(F) > 0 feltételt.
fn generate_valid_f(rng: &mut impl RngExt) -> [f32; 9] {
    [
        rng.random_range(0.8..1.2), rng.random_range(-0.2..0.2), rng.random_range(-0.2..0.2),
        rng.random_range(-0.2..0.2), rng.random_range(0.8..1.2), rng.random_range(-0.2..0.2),
        rng.random_range(-0.2..0.2), rng.random_range(-0.2..0.2), rng.random_range(0.8..1.2),
    ]
}

/// A determináns kiszámítása
fn det_m3(m: &[f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[3] * (m[1] * m[8] - m[2] * m[7]) + m[6] * (m[1] * m[5] - m[2] * m[4])
}

fn run_solid_test(
    name: &str,
    category: &str,
    data_x: Vec<Vec<f32>>,
    data_y: Vec<f32>,
    feature_types: Vec<ValueType>,
    subset_size: Option<usize>,
    generations: usize,
) {
    println!(">>> RUNNING {} <<<", name);
    let dataset = Dataset::new(&data_x, &data_y, feature_types, false);

    let mut config = common::get_test_config(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
    // A hiperelaszticitáshoz ritkán kell szinusz vagy logaritmus (kivéve a motoron belüli J-hez, amit mi intézünk)
    config.excluded_ops = vec![
        Instruction::Basic(BasicOpCode::SinF),
        Instruction::Basic(BasicOpCode::CosF),
        Instruction::Basic(BasicOpCode::LnF),
        Instruction::Basic(BasicOpCode::ExpF),
    ];
    config.subset_size = subset_size;
    config.max_generations = generations;

    let regressor = SymbolicRegressor::new(config);

    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    common::update_history(category, result.mse, time_ms);
    println!(
        "Result {}: MSE = {:.8}, Time = {}ms\nEquation: {}",
        category, result.mse, time_ms, result.equation
    );
}

#[test]
fn solid_1_neo_hookean() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    
    // Tiszta adat, kis minta. Feladat: W = C10 * (I1_bar - 3)
    let c10 = 2.5;

    for _ in 0..300 {
        let f = generate_valid_f(&mut rng);
        dx.push(f.to_vec());

        let j = det_m3(&f);
        let i1 = f.iter().map(|x| x * x).sum::<f32>();
        let i1_bar = j.powf(-2.0 / 3.0) * i1;
        
        dy.push(c10 * (i1_bar - 3.0));
    }
    
    run_solid_test(
        "Solid 1: Neo-Hookean Energy (Easy, Clean)",
        "Solid1", dx, dy, vec![ValueType::Mat3], Some(240), 2000
    );
}

#[test]
fn solid_2_mooney_rivlin_noisy() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    
    let c10 = 1.5;
    let c01 = 0.8;

    for _ in 0..800 {
        let f = generate_valid_f(&mut rng);
        dx.push(f.to_vec());

        let j = det_m3(&f);
        let i1 = f.iter().map(|x| x * x).sum::<f32>();
        
        let c00 = f[0]*f[0] + f[3]*f[3] + f[6]*f[6];
        let c01_m = f[0]*f[1] + f[3]*f[4] + f[6]*f[7];
        let c02 = f[0]*f[2] + f[3]*f[5] + f[6]*f[8];
        let c10_m = f[1]*f[0] + f[4]*f[3] + f[7]*f[6];
        let c11 = f[1]*f[1] + f[4]*f[4] + f[7]*f[7];
        let c12 = f[1]*f[2] + f[4]*f[5] + f[7]*f[8];
        let c20 = f[2]*f[0] + f[5]*f[3] + f[8]*f[6];
        let c21 = f[2]*f[1] + f[5]*f[4] + f[8]*f[7];
        let c22 = f[2]*f[2] + f[5]*f[5] + f[8]*f[8];
        
        let tr_c2 = c00*c00 + c01_m*c10_m + c02*c20 + c10_m*c01_m + c11*c11 + c12*c21 + c20*c02 + c21*c12 + c22*c22;
        
        let i1_bar = j.powf(-2.0 / 3.0) * i1;
        let i2_bar = j.powf(-4.0 / 3.0) * 0.5 * (i1 * i1 - tr_c2);
        
        let noise = rng.random_range(-0.01..0.01); // 1% körüli zaj
        dy.push(c10 * (i1_bar - 3.0) + c01 * (i2_bar - 3.0) + noise);
    }
    
    run_solid_test(
        "Solid 2: Mooney-Rivlin (Medium, Noisy)",
        "Solid2", dx, dy, vec![ValueType::Mat3], Some(400), 3000
    );
}

#[test]
fn solid_3_nanson_formula() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    
    // Feladat: Felület transzformációja (Nanson képlet). Új vektor hossza: || Cof(F) * N ||
    for _ in 0..500 {
        let f = generate_valid_f(&mut rng);
        let n = [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)];
        
        let mut row = f.to_vec();
        row.extend_from_slice(&n);
        dx.push(row);

        let cof = [
            f[4]*f[8] - f[5]*f[7], f[5]*f[6] - f[3]*f[8], f[3]*f[7] - f[4]*f[6],
            f[2]*f[7] - f[1]*f[8], f[0]*f[8] - f[2]*f[6], f[1]*f[6] - f[0]*f[7],
            f[1]*f[5] - f[2]*f[4], f[2]*f[3] - f[0]*f[5], f[0]*f[4] - f[1]*f[3],
        ];

        let out_x = cof[0] * n[0] + cof[3] * n[1] + cof[6] * n[2];
        let out_y = cof[1] * n[0] + cof[4] * n[1] + cof[7] * n[2];
        let out_z = cof[2] * n[0] + cof[5] * n[1] + cof[8] * n[2];

        dy.push((out_x*out_x + out_y*out_y + out_z*out_z).sqrt());
    }
    
    run_solid_test(
        "Solid 3: Nanson Formula Area Norm (Hard, Tensor-Vector)",
        "Solid3", dx, dy, vec![ValueType::Mat3, ValueType::Vec3], Some(280), 4000
    );
}

#[test]
fn solid_4_st_venant_kirchhoff() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    
    let lambda = 120.0;
    let mu = 80.0;

    for _ in 0..1000 {
        let f = generate_valid_f(&mut rng);
        dx.push(f.to_vec());

        let e = [
            0.5 * (f[0]*f[0] + f[3]*f[3] + f[6]*f[6] - 1.0), 0.5 * (f[0]*f[1] + f[3]*f[4] + f[6]*f[7]),       0.5 * (f[0]*f[2] + f[3]*f[5] + f[6]*f[8]),
            0.5 * (f[1]*f[0] + f[4]*f[3] + f[7]*f[6]),       0.5 * (f[1]*f[1] + f[4]*f[4] + f[7]*f[7] - 1.0), 0.5 * (f[1]*f[2] + f[4]*f[5] + f[7]*f[8]),
            0.5 * (f[2]*f[0] + f[5]*f[3] + f[8]*f[6]),       0.5 * (f[2]*f[1] + f[5]*f[4] + f[8]*f[7]),       0.5 * (f[2]*f[2] + f[5]*f[5] + f[8]*f[8] - 1.0),
        ];

        let tr_e = e[0] + e[4] + e[8];
        let tr_e2 = e[0]*e[0] + e[1]*e[3] + e[2]*e[6] + 
                    e[3]*e[1] + e[4]*e[4] + e[5]*e[7] + 
                    e[6]*e[2] + e[7]*e[5] + e[8]*e[8];

        let w = (lambda / 2.0) * tr_e * tr_e + mu * tr_e2;
        dy.push(w);
    }
    
    run_solid_test(
        "Solid 4: St. Venant-Kirchhoff (Large dataset, Very Hard)",
        "Solid4", dx, dy, vec![ValueType::Mat3], Some(600), 5000
    );
}

#[test]
fn solid_5_yeoh_3rd_order() {
    let mut rng = rand::rng();
    let mut dx = Vec::new();
    let mut dy = Vec::new();
    
    // Yeoh modell: W = c1(I1-3) + c2(I1-3)^2 + c3(I1-3)^3
    let c1 = 0.5;
    let c2 = -0.1;
    let c3 = 0.02;

    // Kifejezetten nagy adatmennyiség, ahogy az ipari regressziónál szokás
    for _ in 0..2000 {
        let f = generate_valid_f(&mut rng);
        dx.push(f.to_vec());

        let j = det_m3(&f);
        let i1 = f.iter().map(|x| x * x).sum::<f32>();
        let i1_bar = j.powf(-2.0 / 3.0) * i1;
        
        let diff = i1_bar - 3.0;
        dy.push(c1 * diff + c2 * diff * diff + c3 * diff * diff * diff);
    }
    
    run_solid_test(
        "Solid 5: Yeoh 3rd Order (Massive Dataset, Deep Polynomial)",
        "Solid5", dx, dy, vec![ValueType::Mat3], Some(800), 4000
    );
}