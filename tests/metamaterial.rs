use rsr::api::*;
use rsr::engine::search::config::LossFunctionType;
use rsr::prelude::*;

mod common;

// A 3D (3x3) tenzorok beolvasásához a 18 oszlopos CSV-kből (9 elem az F-hez, 9 a P/sigma-hoz)
fn load_mat3_csv(path: &str) -> (Vec<Vec<f32>>, Vec<[f32; 9]>) {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .expect("Nem sikerült beolvasni a CSV-t");

    let mut dx = Vec::new();
    let mut dy = Vec::new();

    for result in rdr.records() {
        if let Ok(record) = result {
            // Az első 9 oszlop a deformációs gradiens (F)
            let f_vec = vec![
                record[0].parse().unwrap_or(0.0),
                record[1].parse().unwrap_or(0.0),
                record[2].parse().unwrap_or(0.0),
                record[3].parse().unwrap_or(0.0),
                record[4].parse().unwrap_or(0.0),
                record[5].parse().unwrap_or(0.0),
                record[6].parse().unwrap_or(0.0),
                record[7].parse().unwrap_or(0.0),
                record[8].parse().unwrap_or(0.0),
            ];
            
            // A következő 9 oszlop a feszültségtenzor (P vagy Cauchy szigma)[cite: 6]
            let p_arr = [
                record[9].parse().unwrap_or(0.0),
                record[10].parse().unwrap_or(0.0),
                record[11].parse().unwrap_or(0.0),
                record[12].parse().unwrap_or(0.0),
                record[13].parse().unwrap_or(0.0),
                record[14].parse().unwrap_or(0.0),
                record[15].parse().unwrap_or(0.0),
                record[16].parse().unwrap_or(0.0),
                record[17].parse().unwrap_or(0.0),
            ];

            dx.push(f_vec);
            dy.push(p_arr);
        }
    }
    (dx, dy)
}

#[test]
fn test_law_3_pig_sclera_biaxial() {
    let path = "test_data/biomechanical/1/pig 02_SR.csv";
    if !std::path::Path::new(path).exists() {
        println!("Fájl nem található: {}. Kérlek ellenőrizd az elérési utat!", path);
        return;
    }

    let (dx, dy) = load_mat3_csv(path);
    println!(">>> Betöltve {} adatpont a Pig Sclera adathalmazból <<<", dx.len());

    let dataset = Dataset::new_mat3(&dx, &dy, vec![ValueType::Mat3]);
    /*
    TODO: I4, I6
    */
    let mut config = Config::solid_incompressible_anisotropic();
   
    config.max_generations = 2000;
    config.loss_type = LossFunctionType::PlanarBiaxialMse;

    let regressor = SymbolicRegressor::new(config);

    println!(">>> RUNNING 3D Pig Sclera Soft Tissue Law Discovery <<<");
    let start_time = std::time::Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

     result.plot_solid(&dataset, PlotConfig {
        output_path: "pig.png".to_string(),
        mode: SolidPlotMode::BiaxialTension,
        max_points: 800,
    }).expect("Nem sikerült a diagramot kimenteni!");
    
    common::update_history("Tensor_Pig_Sclera", result.clear_mse, time_ms);
    println!("clear MSE = {:.8}, Time = {}ms\nEquation: {}\n", result.clear_mse, time_ms, result.equation);
    let mut i: i32 = 1;
    for pareto in result.pareto_front{
        println!("{}clear MSE = {:.8}, Compl = {}ms\nEquation: {}\n",i, pareto.clear_mse, pareto.complexity, pareto.equation);
        i += 1;
    }
}

#[test]
fn test_law_3_steel_x6cr17_uniaxial() {
    let path = "test_data/metal/1/steel_X6Cr17_SR.csv";
    if !std::path::Path::new(path).exists() {
        println!("Fájl nem található: {}. Kérlek ellenőrizd az elérési utat!", path);
        return;
    }

    let (dx, dy) = load_mat3_csv(path);
    println!(">>> Betöltve {} adatpont a Steel X6Cr17 adathalmazból <<<", dx.len());

    let dataset = Dataset::new_mat3(&dx, &dy, vec![ValueType::Mat3]);
    
    let mut config = Config::solid_elastoplastic_metals();
   
    config.max_generations = 1000;
    config.base_target_mse = 1.0;

    let regressor = SymbolicRegressor::new(config);

    println!(">>> RUNNING 3D Steel Plasticity Cauchy Stress Discovery <<<");
    let start_time = std::time::Instant::now();
    let result = regressor.fit(&dataset);
    let time_ms = start_time.elapsed().as_millis() as u64;

    result.plot_solid(&dataset, PlotConfig {
        output_path: "metal.png".to_string(),
        mode: SolidPlotMode::UniaxialTension,
        max_points: 800,
    }).expect("Nem sikerült a diagramot kimenteni!");
    
    common::update_history("Tensor_Steel_X6Cr17", result.mse, time_ms);
    println!("MSE = {:.8}, Time = {}ms\nEquation: {}\n", result.mse, time_ms, result.equation);
     let mut i: i32 = 1;
    for pareto in result.pareto_front{
        println!("{}clear MSE = {:.8}, Compl = {}ms\nEquation: {}\n",i, pareto.clear_mse, pareto.complexity, pareto.equation);
        i += 1;
    }
}