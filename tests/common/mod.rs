#![allow(dead_code)]
use std::fs::{OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, UNIX_EPOCH};
use fs2::FileExt;

const MAX_ENTRIES: usize = 30;
const TIME_LIMIT: u64 = 43200;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BenchmarkEntry {
    pub timestamp: u64,
    pub commit_hash: String,
    pub biology_mse: Option<f32>,
    pub biology_time_ms: Option<u64>,
    pub physics_mse: Option<f32>,
    pub physics_time_ms: Option<u64>,
    pub stats_mse: Option<f32>,
    pub stats_time_ms: Option<u64>,
}

pub fn update_history(category: &str, mse: f32, time_ms: u64) {
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

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    
    let needs_new_entry = if let Some(last) = history.last() {
        if (now - last.timestamp) > TIME_LIMIT {
            true
        } else {
            match category {
                "Biology" => last.biology_mse.is_some(),
                "Physics" => last.physics_mse.is_some(),
                "Statistics" => last.stats_mse.is_some(),
                _ => true
            }
        }
    } else {
        true
    };

    if needs_new_entry {
        history.push(BenchmarkEntry {
            timestamp: now,
            commit_hash: "local".to_string(),
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

    let max_entries = MAX_ENTRIES;
    if history.len() > max_entries {
        let to_remove = history.len() - max_entries;
        history.drain(0..to_remove);
    }


    let json_data = serde_json::to_string_pretty(&history).unwrap();
    let js_content = format!("{}{}{}", prefix, json_data, suffix);

    file.seek(SeekFrom::Start(0)).unwrap();
    file.set_len(0).unwrap(); 
    file.write_all(js_content.as_bytes()).unwrap();

    file.unlock().unwrap();
}