#![allow(dead_code)]
use std::fs::{OpenOptions, File};
use std::io::{Read};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH};

// Ez a struktúra tárolja EGY futtatás teljes eredményét
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BenchmarkEntry {
    pub timestamp: u64,
    pub commit_hash: String,
    
    pub biology_mse: Option<f64>,
    pub biology_time_ms: Option<u64>,
    
    pub physics_mse: Option<f64>,
    pub physics_time_ms: Option<u64>,
    
    pub stats_mse: Option<f64>,
    pub stats_time_ms: Option<u64>,
}

pub fn update_history(
    category: &str,
    mse: f64,
    time_ms: u64
) {
    let file_path = "benchmark_history.json";

    let mut history: Vec<BenchmarkEntry> = if let Ok(mut file) = File::open(file_path) {
        let mut content = String::new();
        file.read_to_string(&mut content).unwrap_or_default();
        serde_json::from_str(&content).unwrap_or_else(|_| Vec::new())
    } else {
        Vec::new()
    };

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    
    let needs_new_entry = if let Some(last) = history.last() {
        (now - last.timestamp) > 60
    } else {
        true
    };

    if needs_new_entry {
        history.push(BenchmarkEntry {
            timestamp: now,
            commit_hash: "unknown".to_string(),
            biology_mse: None, biology_time_ms: None,
            physics_mse: None, physics_time_ms: None,
            stats_mse: None, stats_time_ms: None,
        });
    }

    if let Some(entry) = history.last_mut() {
        match category {
            "Biology" => {
                entry.biology_mse = Some(mse);
                entry.biology_time_ms = Some(time_ms);
            },
            "Physics" => {
                entry.physics_mse = Some(mse);
                entry.physics_time_ms = Some(time_ms);
            },
            "Statistics" => {
                entry.stats_mse = Some(mse);
                entry.stats_time_ms = Some(time_ms);
            },
            _ => {}
        }
    }

    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(file_path)
        .unwrap();
    serde_json::to_writer_pretty(file, &history).unwrap();
}