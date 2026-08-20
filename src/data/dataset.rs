use crate::eval::types::ValueType;
use wide::f32x4;
use std::error::Error;
use std::path::Path;

pub struct Dataset {
    pub feature_flat: Vec<f32x4>,
    pub target_batches: Vec<f32x4>,
    pub num_features: u8,
    pub num_batches: usize,
    pub num_samples: usize,

    pub feature_means: Vec<f32>,
    pub feature_std_devs: Vec<f32>,
    pub target_mean: f32,
    pub target_std_dev: f32,

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

        if normalize {
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

            let target_sum: f32 = data_y.iter().sum();
            target_mean = target_sum / num_samples as f32;

            let target_variance: f32 = data_y
                .iter()
                .map(|&y| (y - target_mean).powi(2))
                .sum::<f32>()
                / num_samples as f32;
            target_std_dev = if target_variance < 1e-9 {
                1.0
            } else {
                target_variance.sqrt()
            };
        }

        let simd_width = 4;
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
                let batch = f32x4::new([
                    get_norm_sample(start_idx, f_idx),
                    get_norm_sample(start_idx + 1, f_idx),
                    get_norm_sample(start_idx + 2, f_idx),
                    get_norm_sample(start_idx + 3, f_idx),
                ]);
                feature_flat.push(batch);
            }

            let target_batch = f32x4::new([
                get_norm_target(start_idx),
                get_norm_target(start_idx + 1),
                get_norm_target(start_idx + 2),
                get_norm_target(start_idx + 3),
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
    pub fn denormalize_target_simd(&self, normalized_batch: f32x4) -> f32x4 {
        if self.is_normalized {
            let std = f32x4::splat(self.target_std_dev);
            let mean = f32x4::splat(self.target_mean);
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

    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        feature_types: Vec<ValueType>,
        target_col_index: Option<usize>,
        normalize: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(path)?;
        let expected_floats: usize = feature_types.iter().map(|t| match t {
            ValueType::Float => 1,
            ValueType::Vec2 => 2,
            ValueType::Vec3 => 3,
            ValueType::Mat2 => 4,
            ValueType::Mat3 => 9,
            _ => 1,
        }).sum();

        let mut data_x = Vec::new();
        let mut data_y = Vec::new();

        for (line_idx, result) in rdr.records().enumerate() {
            let record = result?;
            let target_idx = target_col_index.unwrap_or(record.len() - 1);

            if record.len() < expected_floats + 1 {
                return Err(format!(
                    "Error at line {}. : not enough column! Expected: {}, actual: {}", 
                    line_idx + 1, expected_floats + 1, record.len()
                ).into());
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

        Ok(Dataset::new(&data_x, &data_y, feature_types, normalize))
    }
}
