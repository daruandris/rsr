mod common;

use std::path::Path;
use std::time::Instant;
use rsr::prelude::*;
use rsr::Instruction;
use rsr::domains::basic::BasicOpCode;
use rsr::domains::linalg::LinalgOpCode;
use rsr::domains::solid::SolidOpCode;

fn get_vector_exclusions() -> Vec<Instruction> {
    vec![
        Instruction::Linalg(LinalgOpCode::MakeVec2),
        Instruction::Linalg(LinalgOpCode::MakeVec3),
        Instruction::Linalg(LinalgOpCode::DotV2),
        Instruction::Linalg(LinalgOpCode::DotV3),
        Instruction::Linalg(LinalgOpCode::CrossV3),
        Instruction::Linalg(LinalgOpCode::GetXV3),
        Instruction::Linalg(LinalgOpCode::GetYV3),
        Instruction::Linalg(LinalgOpCode::GetZV3),
        Instruction::Linalg(LinalgOpCode::GetXV2),
        Instruction::Linalg(LinalgOpCode::GetYV2),
        Instruction::Linalg(LinalgOpCode::AddV3),
        Instruction::Linalg(LinalgOpCode::AddV2),
        Instruction::Linalg(LinalgOpCode::SubV3),
        Instruction::Linalg(LinalgOpCode::SubV2),
        Instruction::Linalg(LinalgOpCode::ScaleV2),
        Instruction::Linalg(LinalgOpCode::ScaleV3),
        Instruction::Linalg(LinalgOpCode::MulM2V2),
        Instruction::Linalg(LinalgOpCode::MulM3V3),
        Instruction::Basic(BasicOpCode::SinF),
        Instruction::Basic(BasicOpCode::CosF),
        Instruction::Basic(BasicOpCode::LnF),
        Instruction::Basic(BasicOpCode::ExpF),
        Instruction::Basic(BasicOpCode::SqrtF),
        Instruction::Basic(BasicOpCode::DivF),

        // ÚJ: Tenzor aritmetika szigorú tiltása! 
        // Ne tudjon mátrixokat összeadni, invertálni, csak invariánst képezni belőlük.
        Instruction::Linalg(LinalgOpCode::AddM3),
        Instruction::Linalg(LinalgOpCode::SubM3),
        Instruction::Linalg(LinalgOpCode::ScaleM3),
        Instruction::Linalg(LinalgOpCode::MulM3),
        Instruction::Linalg(LinalgOpCode::InverseM3),
        Instruction::Linalg(LinalgOpCode::TransposeM3),
        Instruction::Linalg(LinalgOpCode::DetM3),
        Instruction::Linalg(LinalgOpCode::TraceM3),
        
        // ÚJ: Felesleges kontinuummechanikai operátorok tiltása
        Instruction::Solid(SolidOpCode::CofactorM3),
        Instruction::Solid(SolidOpCode::DeviatoricM3),
        Instruction::Solid(SolidOpCode::RightCauchyGreenM3),
        Instruction::Solid(SolidOpCode::LeftCauchyGreenM3),
        Instruction::Solid(SolidOpCode::GreenLagrangeStrainM3),
        Instruction::Solid(SolidOpCode::TraceSqrM3),
    ]
}

fn run_treloar_test(
    name: &str,
    category: &str,
    data_x: Vec<Vec<f32>>,
    data_y: Vec<f32>,
) {
    println!(">>> RUNNING {} <<<", name);
    let dataset = Dataset::new(&data_x, &data_y, vec![ValueType::Mat3], false).with_scalar_extraction(false);

    let mut config = common::get_test_config(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
    config.excluded_ops = get_vector_exclusions();
    // A valós adatokhoz elegendő kisebb iterációszám és subset
    config.disabled_constant_types = vec![
        ValueType::Mat3,
        ValueType::Mat2,
        ValueType::Vec3,
        ValueType::Vec2,
    ];
    config.max_generations = 5000;
    config.island_size = 500;
    config.num_islands = 32;
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
fn treloar_1_uniaxial() {
    let path = "test_data/Treloar/Treloar_uniaxial.csv";
    if !Path::new(path).exists() { return; }
    let (dx, dy) = load_treloar_csv(path, "uniaxial");
    run_treloar_test("Treloar: Uniaxial", "TreloarUni", dx, dy);
}

#[test]
fn treloar_2_biaxial() {
    let path = "test_data/Treloar/Treloar_equibiaxial.csv";
    if !Path::new(path).exists() { return; }
    let (dx, dy) = load_treloar_csv(path, "biaxial");
    run_treloar_test("Treloar: Biaxial", "TreloarBi", dx, dy);
}

#[test]
fn treloar_3_shear() {
    let path = "test_data/Treloar/Treloar_pureshear.csv";
    if !Path::new(path).exists() { return; }
    let (dx, dy) = load_treloar_csv(path, "shear");
    run_treloar_test("Treloar: Pure Shear", "TreloarShear", dx, dy);
}
