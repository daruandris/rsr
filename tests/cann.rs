mod common;

use std::path::Path;
use std::time::Instant;
use rsr::prelude::*;
use rsr::Instruction;
use rsr::domains::basic::BasicOpCode;
use rsr::domains::linalg::LinalgOpCode;
use rsr::domains::solid::SolidOpCode;

// Szigorú fizikai szűrő, ahogy a Treloarnál is
fn get_cann_exclusions() -> Vec<Instruction> {
    vec![
        // Vektorok tiltása
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
        
        // Tenzorok aritmetikájának tiltása
        Instruction::Linalg(LinalgOpCode::AddM3),
        Instruction::Linalg(LinalgOpCode::SubM3),
        Instruction::Linalg(LinalgOpCode::ScaleM3),
        Instruction::Linalg(LinalgOpCode::MulM3),
        Instruction::Linalg(LinalgOpCode::InverseM3),
        Instruction::Linalg(LinalgOpCode::TransposeM3),
        Instruction::Linalg(LinalgOpCode::DetM3),
        Instruction::Linalg(LinalgOpCode::TraceM3),
        Instruction::Solid(SolidOpCode::CofactorM3),
        Instruction::Solid(SolidOpCode::DeviatoricM3),
        Instruction::Solid(SolidOpCode::RightCauchyGreenM3),
        Instruction::Solid(SolidOpCode::LeftCauchyGreenM3),
        Instruction::Solid(SolidOpCode::GreenLagrangeStrainM3),
        Instruction::Solid(SolidOpCode::TraceSqrM3),
        
        // Alap matek szűrése (DivF tiltva a stabil polinomokért)
        Instruction::Basic(BasicOpCode::SinF),
        Instruction::Basic(BasicOpCode::CosF),
        Instruction::Basic(BasicOpCode::LnF),
        Instruction::Basic(BasicOpCode::ExpF),
        Instruction::Basic(BasicOpCode::SqrtF),
        Instruction::Basic(BasicOpCode::DivF),
    ]
}

/// A CANN adatok beolvasása és energia-Mátrix transzformáció
fn load_cann_csv(path: &str, mode: &str) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false) // Figyelem: manuális CSV exportnál általában nincs fejléc!
        .from_path(path)
        .unwrap_or_else(|_| panic!("Hiba a CSV betoltesekor: {}", path));

    let mut strains = Vec::new();
    let mut stresses = Vec::new();

    for result in rdr.records() {
        if let Ok(record) = result {
            if let (Ok(x), Ok(y)) = (record[0].parse::<f32>(), record[1].parse::<f32>()) {
                strains.push(x);
                stresses.push(y);
            }
        }
    }

    let mut dx = Vec::with_capacity(strains.len());
    let mut dy = Vec::with_capacity(strains.len());
    let mut current_w = 0.0;

    for i in 0..strains.len() {
        let strain = strains[i]; 
        
        // Numerikus integrálás az energiasűrűség (W) kiszámításához
        if i > 0 {
            let d_strain = strain - strains[i - 1];
            let avg_stress = (stresses[i] + stresses[i - 1]) / 2.0;
            current_w += avg_stress * d_strain;
        }

        // F mátrix felépítése a terhelési mód alapján (agyszövet esetén is incompressible feltételezés)
        let f_matrix = match mode {
            "uniaxial" => {
                let lam = strain;
                vec![
                    lam, 0.0, 0.0,
                    0.0, 1.0 / lam.sqrt(), 0.0,
                    0.0, 0.0, 1.0 / lam.sqrt()
                ]
            },
            "shear" => {
                let gamma = strain;
                vec![
                    1.0, gamma, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0
                ]
            },
            _ => panic!("Ismeretlen terhelesi mod!"),
        };
        
        dx.push(f_matrix);
        dy.push(current_w);
    }
    (dx, dy)
}

fn run_cann_test(name: &str, category: &str, data_x: Vec<Vec<f32>>, data_y: Vec<f32>) {
    println!(">>> RUNNING {} <<<", name);
    
    // Szigorúan Mat3 bemenet, és letiltjuk a skalár kibontást a Datasetben!
    let dataset = Dataset::new(&data_x, &data_y, vec![ValueType::Mat3], false)
        .with_scalar_extraction(false);

    let mut config = common::get_test_config(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
    config.excluded_ops = get_cann_exclusions();
    
    // Tiltott konstansok beállítása (nincs több mátrix szivárgás!)
    config.disabled_constant_types = vec![
        ValueType::Vec2, ValueType::Vec3, ValueType::Mat2, ValueType::Mat3
    ];
    
    // A valós zajos adatok és a kombinált illesztés miatt erősítjük a keresést
    config.max_generations = 10000;
    config.num_islands = 32;
    config.island_size = 500;
    config.subset_size = Some(data_x.len());

    let regressor = SymbolicRegressor::new(config);

    let start_time = Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    common::update_history(category, result.mse, time_ms);
    println!(
        "Result {}: MSE = {:.8}, Time = {}ms\nEquation: {}\n",
        category, result.mse, time_ms, result.equation
    );
}

#[test]
fn cann_1_brain_combined() {
    let paths = [
        ("test_data/CANN/brain_uniaxial.csv", "uniaxial"),
        ("test_data/CANN/brain_shear.csv", "shear"),
    ];

    let mut combined_dx = Vec::new();
    let mut combined_dy = Vec::new();

    for (path, mode) in paths.iter() {
        let file_path = Path::new(path);
        if file_path.exists() {
            let (mut dx, mut dy) = load_cann_csv(path, mode);
            combined_dx.append(&mut dx);
            combined_dy.append(&mut dy);
        } else {
            println!("Figyelmeztetes: {} hianyzik! A gép itt kereste: {:?}", path, file_path.canonicalize());
        }
    }

    if !combined_dx.is_empty() {
        run_cann_test("CANN: Brain Tissue Combined", "CANN_Brain", combined_dx, combined_dy);
    } else {
        println!("Nem talalhato egyetlen CANN adatfajl sem a test_data/CANN mappaban.");
    }
}