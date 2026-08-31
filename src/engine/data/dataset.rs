//! SIMD-optimized dataset structures for ultra-fast evaluation.
use crate::engine::data::schema::Schema;
use crate::engine::eval::types::ValueType;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use wide::f32x8;

/// A heavily optimized data container designed for SIMD execution.
///
/// The `Dataset` takes raw tabular data (from arrays, CSV, or JSON) and restructures
/// it into a vectorized format (`f32x8`). This allows the genetic engine's Virtual Machine
/// to evaluate 4 data points simultaneously in a single CPU cycle.
///
/// It also handles transparent Z-score standardization (normalization) and denormalization.
#[derive(Clone)]
pub struct Dataset {
    /// Flattened and SIMD-aligned input features.
    pub feature_flat: Vec<f32x8>,
    /// SIMD-aligned target values (Y).
    pub target_batches: Vec<f32x8>,
    pub num_features: u8,
    pub num_batches: usize,
    pub num_samples: usize,

    pub feature_means: Vec<f32>,
    pub feature_std_devs: Vec<f32>,
    pub target_mean: f32,
    pub target_std_dev: f32,
    pub target_variance: f32,

    pub is_normalized: bool,
    pub feature_types: Vec<ValueType>,
}

impl Dataset {
    pub fn new(
        data_x: &[Vec<f32>],
        data_y: &[f32],
        feature_types: Vec<ValueType>,
        normalize: bool,
    ) -> Self {
        let num_samples = data_x.len();
        let mut num_features_usize = 0;

        for t in &feature_types {
            num_features_usize += match t {
                ValueType::Float => 1,
                ValueType::Vec2 => 2,
                ValueType::Vec3 => 3,
                ValueType::Mat2 => 4,
                ValueType::Mat3 => 9,
                _ => 1,
            };
        }
        let num_features = num_features_usize as u8;

        let mut feature_means = vec![0.0; num_features_usize];
        let mut feature_std_devs = vec![1.0; num_features_usize];
        let mut target_mean = 0.0;
        let mut target_std_dev = 1.0;

        let target_sum: f32 = data_y.iter().sum();
        let actual_target_mean = target_sum / num_samples as f32;
        let mut target_variance: f32 = data_y
            .iter()
            .map(|&y| (y - actual_target_mean).powi(2))
            .sum::<f32>()
            / num_samples as f32;
        if target_variance < 1e-9 {
            target_variance = 1.0;
        }

        if normalize {
            target_mean = actual_target_mean;
            target_std_dev = target_variance.sqrt();

            for f_idx in 0..num_features_usize {
                let sum: f32 = data_x.iter().map(|row| row[f_idx]).sum();
                let mean = sum / num_samples as f32;
                feature_means[f_idx] = mean;

                let variance: f32 = data_x
                    .iter()
                    .map(|row| (row[f_idx] - mean).powi(2))
                    .sum::<f32>()
                    / num_samples as f32;
                feature_std_devs[f_idx] = if variance < 1e-9 {
                    1.0
                } else {
                    variance.sqrt()
                };
            }
        }

        let simd_width = 8;
        let remainder = num_samples % simd_width;
        let padding = if remainder == 0 {
            0
        } else {
            simd_width - remainder
        };
        let padded_size = num_samples + padding;
        let num_batches = padded_size / simd_width;

        let mut feature_flat = Vec::with_capacity(num_batches * num_features_usize);
        let mut target_batches = Vec::with_capacity(num_batches);

        let get_norm_sample = |idx: usize, f_idx: usize| -> f32 {
            if idx < num_samples {
                let val = data_x[idx][f_idx];
                (val - feature_means[f_idx]) / feature_std_devs[f_idx]
            } else {
                0.0
            }
        };

        let get_norm_target = |idx: usize| -> f32 {
            if idx < num_samples {
                let val = data_y[idx];
                (val - target_mean) / target_std_dev
            } else {
                0.0
            }
        };

        for i in 0..num_batches {
            let start_idx = i * simd_width;
            for f_idx in 0..num_features_usize {
                let batch = f32x8::new([
                    get_norm_sample(start_idx, f_idx),
                    get_norm_sample(start_idx + 1, f_idx),
                    get_norm_sample(start_idx + 2, f_idx),
                    get_norm_sample(start_idx + 3, f_idx),
                    get_norm_sample(start_idx + 4, f_idx),
                    get_norm_sample(start_idx + 5, f_idx),
                    get_norm_sample(start_idx + 6, f_idx),
                    get_norm_sample(start_idx + 7, f_idx),
                ]);
                feature_flat.push(batch);
            }

            let target_batch = f32x8::new([
                get_norm_target(start_idx),
                get_norm_target(start_idx + 1),
                get_norm_target(start_idx + 2),
                get_norm_target(start_idx + 3),
                get_norm_target(start_idx + 4),
                get_norm_target(start_idx + 5),
                get_norm_target(start_idx + 6),
                get_norm_target(start_idx + 7),
            ]);
            target_batches.push(target_batch);
        }

        Self {
            feature_flat,
            target_batches,
            num_features,
            num_batches,
            num_samples,
            feature_means,
            feature_std_devs,
            target_mean,
            target_std_dev,
            target_variance,
            is_normalized: normalize,
            feature_types,
        }
    }

    #[inline(always)]
    pub fn denormalize_target(&self, normalized_val: f32) -> f32 {
        if self.is_normalized {
            normalized_val * self.target_std_dev + self.target_mean
        } else {
            normalized_val
        }
    }

    #[inline(always)]
    pub fn denormalize_feature(&self, normalized_val: f32, feature_idx: usize) -> f32 {
        if self.is_normalized && feature_idx < self.feature_means.len() {
            normalized_val * self.feature_std_devs[feature_idx] + self.feature_means[feature_idx]
        } else {
            normalized_val
        }
    }

    #[inline(always)]
    pub fn denormalize_target_simd(&self, normalized_batch: f32x8) -> f32x8 {
        if self.is_normalized {
            let std = f32x8::splat(self.target_std_dev);
            let mean = f32x8::splat(self.target_mean);
            normalized_batch * std + mean
        } else {
            normalized_batch
        }
    }

    pub fn get_variable_registry(&self) -> Vec<(ValueType, u8)> {
        let mut registry = Vec::new();
        let mut current_idx = 0;

        for &t in &self.feature_types {
            registry.push((t, current_idx));
            let size = match t {
                ValueType::Float => 1,
                ValueType::Vec2 => 2,
                ValueType::Vec3 => 3,
                ValueType::Mat2 => 4,
                ValueType::Mat3 => 9,
                _ => 1,
            };
            if size > 1 {
                for offset in 0..size {
                    registry.push((ValueType::Float, current_idx + offset));
                }
            }
            current_idx += size;
        }
        registry
    }

    /// Loads a dataset directly from in-memory Rust arrays.
    ///
    /// # Arguments
    ///
    /// * `data_x` - A slice of vectors, where each vector is a row of input features.
    /// * `data_y` - A slice containing the target values for each row.
    /// * `schema` - The [`Schema`] defining data types and preprocessing rules.
    pub fn from_arrays(data_x: &[Vec<f32>], data_y: &[f32], schema: &Schema) -> Self {
        Self::new(
            data_x,
            data_y,
            schema.feature_types.clone(),
            schema.normalize,
        )
    }

    /// Loads a dataset from a JSON file based on the provided schema.
    ///
    /// The JSON file must contain a top-level array of objects. Each object must
    /// explicitly define an `"x"` field (an array of floating-point numbers representing the features)
    /// and a `"y"` field (a single floating-point number representing the target).
    ///
    /// # Expected JSON Format
    ///
    /// ```json
    /// [
    ///     { "x": [1.5, 2.0, -1.0], "y": 4.5 },
    ///     { "x": [3.0, 0.5, 2.1], "y": 8.0 }
    /// ]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the JSON file. Can be a string, `Path`, or `PathBuf`.
    /// * `schema` - The [`Schema`] defining data types and preprocessing rules.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * The file does not exist or cannot be read (I/O error).
    /// * The JSON structure is invalid or does not match the expected `[{x: [...], y: ...}]` format.
    pub fn from_json<P: AsRef<Path>>(path: P, schema: &Schema) -> Result<Self, Box<dyn Error>> {
        #[derive(serde::Deserialize)]
        struct Record {
            x: Vec<f32>,
            y: f32,
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let records: Vec<Record> = serde_json::from_reader(reader)?;

        let mut data_x = Vec::with_capacity(records.len());
        let mut data_y = Vec::with_capacity(records.len());

        for rec in records {
            data_x.push(rec.x);
            data_y.push(rec.y);
        }

        Ok(Self::new(
            &data_x,
            &data_y,
            schema.feature_types.clone(),
            schema.normalize,
        ))
    }

    /// Loads a dataset from a CSV file based on the provided schema.
    ///
    /// The CSV must have a header row. If `target_col_index` is not set in the schema,
    /// the last column is assumed to be the target variable.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read, parsed, or if a row has missing columns.
    pub fn from_csv<P: AsRef<Path>>(path: P, schema: &Schema) -> Result<Self, Box<dyn Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)?;

        let expected_floats: usize = schema
            .feature_types
            .iter()
            .map(|t| match t {
                ValueType::Float => 1,
                ValueType::Vec2 => 2,
                ValueType::Vec3 => 3,
                ValueType::Mat2 => 4,
                ValueType::Mat3 => 9,
                _ => 1,
            })
            .sum();

        let mut data_x = Vec::new();
        let mut data_y = Vec::new();

        for (line_idx, result) in rdr.records().enumerate() {
            let record = result?;
            let target_idx = schema.target_col_index.unwrap_or(record.len() - 1);

            if record.len() < expected_floats + 1 {
                return Err(format!(
                    "Error at line {}. : not enough column! Expected: {}, actual: {}",
                    line_idx + 1,
                    expected_floats + 1,
                    record.len()
                )
                .into());
            }

            let y_val: f32 = record.get(target_idx).unwrap().parse()?;
            let mut x_row = Vec::with_capacity(expected_floats);
            let mut current_col = 0;

            for _ in 0..expected_floats {
                if current_col == target_idx {
                    current_col += 1;
                }
                let val: f32 = record.get(current_col).unwrap().parse()?;
                x_row.push(val);
                current_col += 1;
            }

            data_x.push(x_row);
            data_y.push(y_val);
        }

        Ok(Self::new(
            &data_x,
            &data_y,
            schema.feature_types.clone(),
            schema.normalize,
        ))
    }

    /// Get the subset of the data for better performance
    pub fn subset(&self, target_samples: usize) -> Self {
        let samples = target_samples.min(self.num_samples);
        if samples == self.num_samples {
            return self.clone();
        }

        let simd_width = 8;
        let padding = if samples.is_multiple_of(simd_width) {
            0
        } else {
            simd_width - (samples % simd_width)
        };
        let num_batches = (samples + padding) / simd_width;

        let num_features_usize = self.num_features as usize;
        let mut feature_flat = Vec::with_capacity(num_batches * num_features_usize);
        let mut target_batches = Vec::with_capacity(num_batches);

        let step = (self.num_samples as f64 - 1.0) / (samples as f64 - 1.0).max(1.0);

        let get_feature_val = |orig_idx: usize, f_idx: usize| -> f32 {
            if orig_idx >= self.num_samples {
                return 0.0;
            }
            let batch_idx = orig_idx / 8;
            let lane_idx = orig_idx % 8;

            let vec_val = self.feature_flat[batch_idx * num_features_usize + f_idx];
            let arr: &[f32; 8] = unsafe { &*(&vec_val as *const _ as *const [f32; 8]) };
            arr[lane_idx]
        };

        let get_target_val = |orig_idx: usize| -> f32 {
            if orig_idx >= self.num_samples {
                return 0.0;
            }
            let batch_idx = orig_idx / 8;
            let lane_idx = orig_idx % 8;

            let vec_val = self.target_batches[batch_idx];
            let arr: &[f32; 8] = unsafe { &*(&vec_val as *const _ as *const [f32; 8]) };
            arr[lane_idx]
        };

        for i in 0..num_batches {
            let start_idx = i * simd_width;

            for f_idx in 0..num_features_usize {
                let batch = wide::f32x8::new([
                    get_feature_val(((start_idx as f64) * step).round() as usize, f_idx),
                    get_feature_val((((start_idx + 1) as f64) * step).round() as usize, f_idx),
                    get_feature_val((((start_idx + 2) as f64) * step).round() as usize, f_idx),
                    get_feature_val((((start_idx + 3) as f64) * step).round() as usize, f_idx),
                    get_feature_val((((start_idx + 4) as f64) * step).round() as usize, f_idx),
                    get_feature_val((((start_idx + 5) as f64) * step).round() as usize, f_idx),
                    get_feature_val((((start_idx + 6) as f64) * step).round() as usize, f_idx),
                    get_feature_val((((start_idx + 7) as f64) * step).round() as usize, f_idx),
                ]);
                feature_flat.push(batch);
            }

            let target_batch = wide::f32x8::new([
                get_target_val(((start_idx as f64) * step).round() as usize),
                get_target_val((((start_idx + 1) as f64) * step).round() as usize),
                get_target_val((((start_idx + 2) as f64) * step).round() as usize),
                get_target_val((((start_idx + 3) as f64) * step).round() as usize),
                get_target_val((((start_idx + 4) as f64) * step).round() as usize),
                get_target_val((((start_idx + 5) as f64) * step).round() as usize),
                get_target_val((((start_idx + 6) as f64) * step).round() as usize),
                get_target_val((((start_idx + 7) as f64) * step).round() as usize),
            ]);
            target_batches.push(target_batch);
        }

        Self {
            feature_flat,
            target_batches,
            num_features: self.num_features,
            num_batches,
            num_samples: samples,
            feature_means: self.feature_means.clone(),
            feature_std_devs: self.feature_std_devs.clone(),
            target_mean: self.target_mean,
            target_std_dev: self.target_std_dev,
            target_variance: self.target_variance,
            is_normalized: self.is_normalized,
            feature_types: self.feature_types.clone(),
        }
    }
}
