use rsr::api::*;
use rsr::prelude::*;

mod common;

fn run_solid_test(name: &str, config_fn: fn() -> Config, test_files: &[&str]) {
    let schema = Schema::new(vec![ValueType::Mat3]).with_target_type(ValueType::Mat3);
    
    for path in test_files {
        if !std::path::Path::new(path).exists() {
            println!("Teszt adat nem található (kihagyva): {}", path);
            continue;
        }

        println!(">>> RUNNING: {} - Fájl: {} <<<", name, path);
        let dataset = Dataset::from_csv(path, &schema).expect("Hibás CSV formátum");
        let mut config = config_fn();
        config.max_generations = 1000;
        let regressor = SymbolicRegressor::new(config);

        let start_time = std::time::Instant::now();
        let result = regressor.fit(&dataset);
        let time_ms = start_time.elapsed().as_millis() as u64;

        println!("MSE = {:.8}, Bonyolultság: {}, Time = {}ms\nKiválasztott egyenlet: {}\n", 
                 result.mse, result.complexity, time_ms, result.equation);
    }
}

#[test]
fn test_solid_incompressible_isotropic() {
    let files = [
        "test_data/solid/incompressible/isotropic/treloar/treloar_Isotropic_FP.csv",
    ];
    run_solid_test(
        "Solid Incompressible Isotropic", 
        || Config::solid_incompressible_isotropic(false), 
        &files
    );
}

#[test]
fn test_solid_incompressible_anisotropic() {
    let files = [
        "test_data/solid/incompressible/anisotropic/CANN_biaxial_deli_meat/AC_animal_chicken_FP.csv",
        "test_data/solid/incompressible/anisotropic/CANN_biaxial_deli_meat/AH_animal_ham_FP.csv",
        "test_data/solid/incompressible/anisotropic/CANN_biaxial_deli_meat/AP_animal_prosciutto_FP.csv",
        "test_data/solid/incompressible/anisotropic/CANN_biaxial_deli_meat/AT_animal_turkey_FP.csv",
        "test_data/solid/incompressible/anisotropic/CANN_biaxial_deli_meat/PP_plant_prosciutto_FP.csv",
        "test_data/solid/incompressible/anisotropic/CANN_biaxial_deli_meat/TA_tofurky_ham_FP.csv",
        "test_data/solid/incompressible/anisotropic/CANN_biaxial_deli_meat/TI_tofurky_hickory_FP.csv",
        "test_data/solid/incompressible/anisotropic/CANN_biaxial_deli_meat/TT_tofurky_turkey_FP.csv",

        "test_data/solid/incompressible/anisotropic/CANN_heart/HeartTissue_FP.csv",
    ];
    run_solid_test(
        "Solid Incompressible Anisotropic", 
        Config::solid_incompressible_anisotropic, 
        &files
    );
}

#[test]
fn test_solid_compressible_isotropic() {
    let files = [//TODO
    ];
    run_solid_test(
        "Solid Compressible Isotropic", 
        Config::solid_compressible_isotropic, 
        &files
    );
}

#[test]
fn test_solid_compressible_anisotropic() {
    let files = [//TODO
    ];
    run_solid_test(
        "Solid Compressible Anisotropic", 
        Config::solid_compressible_anisotropic, 
        &files
    );
}