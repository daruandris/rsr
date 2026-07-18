#![allow(dead_code)]
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::time::{SystemTime, UNIX_EPOCH};

use rsr::prelude::*; // Az új prelude!

const MAX_ENTRIES: usize = 30;
const TIME_LIMIT: u64 = 43200; // Fél nap

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct BenchmarkEntry {
    pub timestamp: u64,
    pub commit_hash: String,

    pub basic1_mse: Option<f32>,
    pub basic1_time_ms: Option<u64>,
    pub basic2_mse: Option<f32>,
    pub basic2_time_ms: Option<u64>,
    pub basic3_mse: Option<f32>,
    pub basic3_time_ms: Option<u64>,
    pub basic4_mse: Option<f32>,
    pub basic4_time_ms: Option<u64>,
    pub basic5_mse: Option<f32>,
    pub basic5_time_ms: Option<u64>,
    pub basic6_mse: Option<f32>,
    pub basic6_time_ms: Option<u64>,
    pub basic7_mse: Option<f32>,
    pub basic7_time_ms: Option<u64>,
    pub basic8_mse: Option<f32>,
    pub basic8_time_ms: Option<u64>,
    pub basic9_mse: Option<f32>,
    pub basic9_time_ms: Option<u64>,
    pub basic10_mse: Option<f32>,
    pub basic10_time_ms: Option<u64>,

    pub linalg1_mse: Option<f32>,
    pub linalg1_time_ms: Option<u64>,
    pub linalg2_mse: Option<f32>,
    pub linalg2_time_ms: Option<u64>,
    pub linalg3_mse: Option<f32>,
    pub linalg3_time_ms: Option<u64>,
    pub linalg4_mse: Option<f32>,
    pub linalg4_time_ms: Option<u64>,
}

pub fn update_history(category: &str, mut mse: f32, time_ms: u64) {
    if mse == 0.0 {
        mse = 1e-8;
    }
    let file_path = "benchmark_history.js";
    let prefix = "const benchmarkHistory = ";
    let suffix = ";";

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(file_path)
        .expect("Failed to open benchmark file");

    file.lock_exclusive().expect("Failed to lock file");

    let mut content = String::new();
    file.read_to_string(&mut content).unwrap_or_default();

    let mut history: Vec<BenchmarkEntry> = if content.trim().is_empty() {
        Vec::new()
    } else {
        let json_part = content
            .trim()
            .strip_prefix(prefix)
            .and_then(|s| s.strip_suffix(suffix))
            .unwrap_or("[]");
        serde_json::from_str(json_part)
            .expect("Corrupted benchmark history file! Fix or delete it manually.")
    };

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let needs_new_entry = if let Some(last) = history.last() {
        if (now - last.timestamp) > TIME_LIMIT {
            true
        } else {
            match category {
                "Basic1" => last.basic1_mse.is_some(),
                "Basic2" => last.basic2_mse.is_some(),
                "Basic3" => last.basic3_mse.is_some(),
                "Basic4" => last.basic4_mse.is_some(),
                "Basic5" => last.basic5_mse.is_some(),
                "Basic6" => last.basic6_mse.is_some(),
                "Basic7" => last.basic7_mse.is_some(),
                "Basic8" => last.basic8_mse.is_some(),
                "Basic9" => last.basic9_mse.is_some(),
                "Basic10" => last.basic10_mse.is_some(),
                "Linalg1" => last.linalg1_mse.is_some(),
                "Linalg2" => last.linalg2_mse.is_some(),
                "Linalg3" => last.linalg3_mse.is_some(),
                "Linalg4" => last.linalg4_mse.is_some(),
                _ => true,
            }
        }
    } else {
        true
    };

    if needs_new_entry {
        let mut new_entry = BenchmarkEntry::default();
        new_entry.timestamp = now;
        new_entry.commit_hash = env!("GIT_HASH").to_string();
        history.push(new_entry);
    }

    if let Some(entry) = history.last_mut() {
        let mse_opt = Some(mse);
        let time_opt = Some(time_ms);
        match category {
            "Basic1" => {
                entry.basic1_mse = mse_opt;
                entry.basic1_time_ms = time_opt;
            }
            "Basic2" => {
                entry.basic2_mse = mse_opt;
                entry.basic2_time_ms = time_opt;
            }
            "Basic3" => {
                entry.basic3_mse = mse_opt;
                entry.basic3_time_ms = time_opt;
            }
            "Basic4" => {
                entry.basic4_mse = mse_opt;
                entry.basic4_time_ms = time_opt;
            }
            "Basic5" => {
                entry.basic5_mse = mse_opt;
                entry.basic5_time_ms = time_opt;
            }
            "Basic6" => {
                entry.basic6_mse = mse_opt;
                entry.basic6_time_ms = time_opt;
            }
            "Basic7" => {
                entry.basic7_mse = mse_opt;
                entry.basic7_time_ms = time_opt;
            }
            "Basic8" => {
                entry.basic8_mse = mse_opt;
                entry.basic8_time_ms = time_opt;
            }
            "Basic9" => {
                entry.basic9_mse = mse_opt;
                entry.basic9_time_ms = time_opt;
            }
            "Basic10" => {
                entry.basic10_mse = mse_opt;
                entry.basic10_time_ms = time_opt;
            }
            "Linalg1" => {
                entry.linalg1_mse = mse_opt;
                entry.linalg1_time_ms = time_opt;
            }
            "Linalg2" => {
                entry.linalg2_mse = mse_opt;
                entry.linalg2_time_ms = time_opt;
            }
            "Linalg3" => {
                entry.linalg3_mse = mse_opt;
                entry.linalg3_time_ms = time_opt;
            }
            "Linalg4" => {
                entry.linalg4_mse = mse_opt;
                entry.linalg4_time_ms = time_opt;
            }
            _ => {}
        }
    }

    if history.len() > MAX_ENTRIES {
        history.drain(0..(history.len() - MAX_ENTRIES));
    }

    let json_data = serde_json::to_string_pretty(&history).unwrap();
    let js_content = format!("{}{}{}", prefix, json_data, suffix);

    file.seek(SeekFrom::Start(0)).unwrap();
    file.set_len(0).unwrap();
    file.write_all(js_content.as_bytes()).unwrap();
    file.unlock().unwrap();
}

pub fn get_test_config(modules: Vec<OpModule>) -> Config {
    Config {
        num_islands: 24,
        island_size: 25,
        max_generations: 10000,
        crossover_rate: 0.10,
        tournament_size: 2,
        migration_interval: 25,
        parsimony_penalty: 0.000005,
        opt_prob: 0.2,
        opt_iterations: 100,
        final_opt_iterations: 5000,
        stagnation_threshold: 1000,
        target_mse: 1e-7,
        min_improvement: 1e-6,
        random_injection_rate: 0.10,
        min_random_injection: 2,
        max_tree_size: 32,
        mutation_max_depth: 4,
        mutation_cycles: 5,
        verbose: true,
        allowed_modules: modules,
        custom_ops: vec![],
        excluded_ops: vec![],
    }
}
