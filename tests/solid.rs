use rsr::api::*;
use rsr::prelude::*;

mod common;

fn run_solid_test(name: &str, config_fn: fn() -> Config, test_files: &[&str], plot_mode: SolidPlotMode) {
    let schema = Schema::new(vec![ValueType::Mat3]).with_target_type(ValueType::Mat3);
    
    for path in test_files {
        if !std::path::Path::new(path).exists() {
            println!("Teszt adat nem található (kihagyva): {}", path);
            continue;
        }

        println!(">>> RUNNING: {} - Fájl: {} <<<", name, path);
        let dataset = Dataset::from_csv(path, &schema).expect("Hibás CSV formátum");
        let mut config = config_fn();
        config.max_generations = 2000;
        let regressor = SymbolicRegressor::new(config);

        let start_time = std::time::Instant::now();
        let result = regressor.fit(&dataset);
        let time_ms = start_time.elapsed().as_millis() as u64;

        println!("MSE = {:.8}, Bonyolultság: {}, Time = {}ms\nLegjobb egyenlet: {}\n", 
                 result.mse, result.complexity, time_ms, result.equation);
                 
        println!("--- Pareto Front ---");
        for (i, entry) in result.pareto_front.iter().enumerate() {
            println!("{:2}. MSE = {:.8}, Bonyolultság = {:2} | Egyenlet: {}", 
                     i + 1, entry.mse, entry.complexity, entry.equation);
        }
        println!("--------------------\n");

        // Fájlnév generálása a CSV nevéből, pl.: treloar_Isotropic_FP_plot.png
        let file_stem = std::path::Path::new(path).file_stem().unwrap().to_string_lossy();
        let out_path = format!("{}_plot.png", file_stem);

        result.plot_solid(&dataset, PlotConfig {
            output_path: out_path.clone(),
            mode: plot_mode,
            max_points: 800,
        }).expect("Nem sikerült a diagramot kimenteni!");
        
        println!("Diagram sikeresen kimentve: {}\n\n", out_path);
    }
}

#[test]
fn test_solid_incompressible_isotropic() {
    let files = [
        "test_data/solid/incompressible/isotropic/treloar/treloar_Isotropic_FP.csv",
        //"test_data/solid/incompressible/isotropic/Sideways and Forward Soft Fracture/"
    ];
    run_solid_test(
        "Solid Incompressible Isotropic", 
        || Config::solid_incompressible_isotropic(false), 
        &files,
        SolidPlotMode::UniaxialTension // A gumiknál klasszikus az egytengelyű húzás
    );
}

#[test]
fn test_solid_incompressible_anisotropic_dispersive() {
    let files = [
        "test_data/solid/incompressible/anisotropic/CANN_heart/HeartTissue_FP.csv",
        //"test_data/solid/incompressible/anisotropic/CANN_skin"
    ];
    run_solid_test(
        "Solid Incompressible Anisotropic", 
        Config::solid_incompressible_anisotropic_dispersive, 
        &files,
        SolidPlotMode::BiaxialTension // Biológiai szöveteknél a Biaxial a standard (2 tengelyes plot)
    );
}

#[test]
fn test_solid_compressible_isotropic() {
    let files = [
        "test_data/solid/compressible/isotropic/foam/foam_FP.csv",
        "test_data/solid/compressible/isotropic/sponge/sponge_FP.csv"
    ];
    run_solid_test(
        "Solid Compressible Isotropic", 
        Config::solid_compressible_isotropic, 
        &files,
        SolidPlotMode::Volumetric // Haboknál és szivacsoknál a térfogat-nyomás viszony a legfontosabb
    );
}

#[test]
fn test_solid_compressible_anisotropic() {
    let files = [
        "test_data/solid/compressible/anisotropic/lattice_a/lattice_A_FP.csv",
        "test_data/solid/compressible/anisotropic/lattice_b/lattice_B_FP.csv"
    ];
    run_solid_test(
        "Solid Compressible Anisotropic", 
        Config::solid_compressible_anisotropic, 
        &files,
        SolidPlotMode::UniaxialTension // A mechanikai metamateriálokat/rácsokat általában uniaxialisan nyomják össze
    );
}