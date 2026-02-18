use rsr::{EvolutionConfig, SimdDataset};

use std::fs::{OpenOptions, File};
use std::io::Write;
use serde::Serialize;

#[derive(Serialize)]
struct BenchResult {
    name: String,
    unit: String,
    value: f64,
}

pub fn append_benchmark_result(name: &str, mse: f64, duration_ms: u128) {
    let new_results = vec![
        BenchResult {
            name: format!("{} - MSE", name),
            unit: "MSE".to_string(),
            value: mse,
        },
        BenchResult {
            name: format!("{} - Time", name),
            unit: "ms".to_string(),
            value: duration_ms as f64,
        }
    ];

    let file_path = "benchmark_output.json";
    
    let mut current_data: Vec<BenchResult> = if let Ok(file) = File::open(file_path) {
        let reader = BufReader::new(file);
        serde_json::from_reader(reader).unwrap_or_else(|_| Vec::new())
    } else {
        Vec::new()
    };

    current_data.extend(new_results);

    let max_entries = 600; 
    
    if current_data.len() > max_entries {
        let to_remove = current_data.len() - max_entries;
        current_data.drain(0..to_remove);
    }

    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(file_path)
        .unwrap();
        
    serde_json::to_writer_pretty(file, &current_data).unwrap();
}

pub fn get_basic_config() -> EvolutionConfig {
    EvolutionConfig {
        num_islands: 2,
        island_size: 100,
        max_generations: 100,
        crossover_rate: 0.85,
        tournament_size: 3,
        migration_interval: 10,
        parsimony_penalty: 0.01,
        
        opt_prob: 0.1,
        opt_iterations: 5,
        
        stagnation_threshold: 10,
        min_improvement: 1e-5,
        target_mse: 1e-7,

        random_injection_rate: 0.05,
        min_random_injection: 1,
        verbose : false,
    }
}

pub fn create_linear_data() -> (SimdDataset, usize) {
    // Cél: y = 2 * X0 + 5
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        let val = i as f64;
        x.push(vec![val]);
        y.push(2.0 * val + 5.0);
    }
    (SimdDataset::new(&x, &y, 1), 1)
}

pub fn create_quadratic_data() -> (SimdDataset, usize) {
    // Cél: y = X0^2 - 10
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        let val = (i as f64) / 5.0;
        x.push(vec![val]);
        y.push(val * val - 10.0);
    }
    (SimdDataset::new(&x, &y, 1), 1)
}

// Cél: y = 2.5 * sin(3.0 * X0)
pub fn create_sine_wave_data() -> (SimdDataset, usize) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        let val = (i as f64) * 0.15; 
        x.push(vec![val]);
        y.push(2.5 * (3.0 * val).sin());
    }
    (SimdDataset::new(&x, &y, 1), 1)
}

// Cél: y = exp(0.5 * X0)
pub fn create_exponential_data() -> (SimdDataset, usize) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..50 {
        let val = (i as f64) * 0.1;
        x.push(vec![val]);
        y.push((0.5 * val).exp());
    }
    (SimdDataset::new(&x, &y, 1), 1)
}