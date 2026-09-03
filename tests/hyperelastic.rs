mod common;

use rand::RngExt;
use rsr::prelude::*;
use rsr::ValueType;
use std::time::Instant;

fn generate_valid_f(rng: &mut impl RngExt) -> [f32; 9] {
    [
        rng.random_range(0.2..3.5), rng.random_range(-0.2..0.2), rng.random_range(-0.2..0.2),
        rng.random_range(-0.2..0.2), rng.random_range(0.2..3.5), rng.random_range(-0.2..0.2),
        rng.random_range(-0.2..0.2), rng.random_range(-0.2..0.2), rng.random_range(0.2..3.5),
    ]
}

fn det_m3(m: &[f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[3] * (m[1] * m[8] - m[2] * m[7])
        + m[6] * (m[1] * m[5] - m[2] * m[4])
}

fn run_hyperelastic_test(name: &str, data_x: Vec<Vec<f32>>, data_y: Vec<f32>) {
    println!(">>> RUNNING {} <<<", name);
    let dataset = Dataset::new(&data_x, &data_y, vec![ValueType::Mat3], false).with_scalar_extraction(false);
    
    let config = Config::hyperelastic_energy(); 
    let regressor = SymbolicRegressor::new(config);

    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    println!("MSE = {:.8}, Time = {}ms\nEquation: {}\n", result.mse, time_ms, result.equation);
}

#[test]
fn test_hyperelastic_1_neo_hooke() {
    let mut rng = rand::rng();
    let (mut dx, mut dy) = (Vec::new(), Vec::new());
    let c10 = 2.5;

    for _ in 0..300 {
        let f = generate_valid_f(&mut rng);
        dx.push(f.to_vec());

        let j = det_m3(&f);
        let i1 = f.iter().map(|x| x * x).sum::<f32>();
        let i1_bar = j.powf(-2.0 / 3.0) * i1;

        dy.push(c10 * (i1_bar - 3.0));
    }
    run_hyperelastic_test("Hyperelastic 1: Neo-Hooke (Easy, Clean)", dx, dy);
}

#[test]
fn test_hyperelastic_2_mooney_rivlin() {
    let mut rng = rand::rng();
    let (mut dx, mut dy) = (Vec::new(), Vec::new());
    let (c10, c01) = (1.5, 0.8);

    for _ in 0..800 {
        let f = generate_valid_f(&mut rng);
        dx.push(f.to_vec());
        let j = det_m3(&f);

        let i1 = f.iter().map(|x| x * x).sum::<f32>();
        

        let c00 = f[0]*f[0] + f[3]*f[3] + f[6]*f[6];
        let c11 = f[1]*f[1] + f[4]*f[4] + f[7]*f[7];
        let c22 = f[2]*f[2] + f[5]*f[5] + f[8]*f[8];
        let tr_c2 = c00*c00 + c11*c11 + c22*c22 + 2.0*(f[0]*f[1]+f[3]*f[4]+f[6]*f[7]).powi(2) 
                    + 2.0*(f[0]*f[2]+f[3]*f[5]+f[6]*f[8]).powi(2) + 2.0*(f[1]*f[2]+f[4]*f[5]+f[7]*f[8]).powi(2);
        
        let i1_bar = j.powf(-2.0 / 3.0) * i1;
        let i2_bar = j.powf(-4.0 / 3.0) * 0.5 * (i1 * i1 - tr_c2);

        dy.push(c10 * (i1_bar - 3.0) + c01 * (i2_bar - 3.0));
    }
    run_hyperelastic_test("Hyperelastic 2: Mooney-Rivlin (Industrial)", dx, dy);
}

#[test]
fn test_hyperelastic_3_yeoh_3rd_order() {
    let mut rng = rand::rng();
    let (mut dx, mut dy) = (Vec::new(), Vec::new());
    let (c1, c2, c3) = (0.5, -0.1, 0.02);

    for _ in 0..1000 {
        let f = generate_valid_f(&mut rng);
        dx.push(f.to_vec());
        
        let j = det_m3(&f);
        let i1 = f.iter().map(|x| x * x).sum::<f32>();
        let i1_bar = j.powf(-2.0 / 3.0) * i1;
        
        let diff = i1_bar - 3.0;
        dy.push(c1 * diff + c2 * diff.powi(2) + c3 * diff.powi(3));
    }
    run_hyperelastic_test("Hyperelastic 3: Yeoh 3rd Order (Deep Polynomial)", dx, dy);
}