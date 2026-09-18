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
    pub target_mat2_batches: Option<Vec<[f32x8; 4]>>,
    pub target_mat3_batches: Option<Vec<[f32x8; 9]>>,
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
    pub extract_scalars: bool,
}

impl Dataset {
    pub fn new(
        data_x: &[Vec<f32>],
        data_y: &[Vec<f32>],
        schema: &Schema,
    ) -> Self {
        let num_samples = data_x.len();
        let mut num_features_usize = 0;

        for t in &schema.feature_types {
            num_features_usize += match t {
                ValueType::Float => 1, ValueType::Vec2 => 2, ValueType::Vec3 => 3,
                ValueType::Mat2 => 4, ValueType::Mat3 => 9, _ => 1,
            };
        }

        let simd_width = 8;
        let padding = if num_samples % simd_width == 0 { 0 } else { simd_width - (num_samples % simd_width) };
        let num_batches = (num_samples + padding) / simd_width;

        let mut feature_flat = Vec::with_capacity(num_batches * num_features_usize);
        let mut target_batches = Vec::new();
        let mut target_mat2_batches = None;
        let mut target_mat3_batches = None;

        // X Bemenetek SIMD feltöltése
        for i in 0..num_batches {
            let start = i * simd_width;
            for f_idx in 0..num_features_usize {
                feature_flat.push(f32x8::new([
                    if start < num_samples { data_x[start][f_idx] } else { 0.0 },
                    if start+1 < num_samples { data_x[start+1][f_idx] } else { 0.0 },
                    if start+2 < num_samples { data_x[start+2][f_idx] } else { 0.0 },
                    if start+3 < num_samples { data_x[start+3][f_idx] } else { 0.0 },
                    if start+4 < num_samples { data_x[start+4][f_idx] } else { 0.0 },
                    if start+5 < num_samples { data_x[start+5][f_idx] } else { 0.0 },
                    if start+6 < num_samples { data_x[start+6][f_idx] } else { 0.0 },
                    if start+7 < num_samples { data_x[start+7][f_idx] } else { 0.0 },
                ]));
            }

            // Y Célváltozók SIMD feltöltése a típus alapján
            match schema.target_type {
                ValueType::Float => {
                    target_batches.push(f32x8::new([
                        if start < num_samples { data_y[start][0] } else { 0.0 },
                        if start+1 < num_samples { data_y[start+1][0] } else { 0.0 },
                        if start+2 < num_samples { data_y[start+2][0] } else { 0.0 },
                        if start+3 < num_samples { data_y[start+3][0] } else { 0.0 },
                        if start+4 < num_samples { data_y[start+4][0] } else { 0.0 },
                        if start+5 < num_samples { data_y[start+5][0] } else { 0.0 },
                        if start+6 < num_samples { data_y[start+6][0] } else { 0.0 },
                        if start+7 < num_samples { data_y[start+7][0] } else { 0.0 },
                    ]));
                },
                ValueType::Mat2 => {
                    let mut batch_target = [f32x8::splat(0.0); 4];
                    for dim in 0..4 {
                        batch_target[dim] = f32x8::new([
                            if start < num_samples { data_y[start][dim] } else { 0.0 },
                            if start+1 < num_samples { data_y[start+1][dim] } else { 0.0 },
                            if start+2 < num_samples { data_y[start+2][dim] } else { 0.0 },
                            if start+3 < num_samples { data_y[start+3][dim] } else { 0.0 },
                            if start+4 < num_samples { data_y[start+4][dim] } else { 0.0 },
                            if start+5 < num_samples { data_y[start+5][dim] } else { 0.0 },
                            if start+6 < num_samples { data_y[start+6][dim] } else { 0.0 },
                            if start+7 < num_samples { data_y[start+7][dim] } else { 0.0 },
                        ]);
                    }
                    if target_mat2_batches.is_none() { target_mat2_batches = Some(Vec::new()); }
                    target_mat2_batches.as_mut().unwrap().push(batch_target);
                },
                ValueType::Mat3 => {
                    let mut batch_target = [f32x8::splat(0.0); 9];
                    for dim in 0..9 {
                        batch_target[dim] = f32x8::new([
                            if start < num_samples { data_y[start][dim] } else { 0.0 },
                            if start+1 < num_samples { data_y[start+1][dim] } else { 0.0 },
                            if start+2 < num_samples { data_y[start+2][dim] } else { 0.0 },
                            if start+3 < num_samples { data_y[start+3][dim] } else { 0.0 },
                            if start+4 < num_samples { data_y[start+4][dim] } else { 0.0 },
                            if start+5 < num_samples { data_y[start+5][dim] } else { 0.0 },
                            if start+6 < num_samples { data_y[start+6][dim] } else { 0.0 },
                            if start+7 < num_samples { data_y[start+7][dim] } else { 0.0 },
                        ]);
                    }
                    if target_mat3_batches.is_none() { target_mat3_batches = Some(Vec::new()); }
                    target_mat3_batches.as_mut().unwrap().push(batch_target);
                },
                _ => {}
            }
        }

        Self {
            feature_flat,
            target_batches,
            target_mat2_batches,
            target_mat3_batches,
            num_features: num_features_usize as u8,
            num_batches,
            num_samples,
            feature_means: vec![0.0; num_features_usize],
            feature_std_devs: vec![1.0; num_features_usize],
            target_mean: 0.0,
            target_std_dev: 1.0,
            target_variance: 1.0,
            is_normalized: false,
            feature_types: schema.feature_types.clone(),
            extract_scalars: true,
        }
    }

    pub fn with_scalar_extraction(mut self, extract: bool) -> Self {
        self.extract_scalars = extract;
        self
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
            if size > 1 && self.extract_scalars {
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
    pub fn from_arrays(data_x: &[Vec<f32>], data_y: &[Vec<f32>], schema: &Schema) -> Self {
        Self::new(data_x, data_y, schema)
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
            data_y.push(vec![rec.y]);
        }

        Ok(Self::new(&data_x, &data_y, schema))
    }

    /// Loads a dataset from a CSV file based on the provided schema.
    ///
    /// The CSV must have a header row. If `target_col_index` is not set in the schema,
    /// the last column is assumed to be the target variable.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read, parsed, or if a row has missing columns.
    pub fn from_csv<P: AsRef<Path>>(path: P, schema: &Schema) -> Result<Self, Box<dyn Error>> {
        let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(path)?;

        let expected_floats: usize = schema.feature_types.iter().map(|t| match t {
            ValueType::Vec2 => 2, ValueType::Vec3 => 3, ValueType::Mat2 => 4, ValueType::Mat3 => 9, _ => 1,
        }).sum();

        let target_dim = match schema.target_type {
            ValueType::Vec2 => 2, ValueType::Vec3 => 3, ValueType::Mat2 => 4, ValueType::Mat3 => 9, _ => 1,
        };

        let mut data_x = Vec::new();
        let mut data_y = Vec::new();

        for result in rdr.records() {
            let record = result?;
            let target_start = schema.target_col_index.unwrap_or(record.len() - target_dim);

            let mut x_row = Vec::with_capacity(expected_floats);
            let mut current_col = 0;
            for _ in 0..expected_floats {
                if current_col == target_start { current_col += target_dim; }
                x_row.push(record.get(current_col).unwrap_or("0.0").parse()?);
                current_col += 1;
            }

            let mut y_row = Vec::with_capacity(target_dim);
            for d in 0..target_dim {
                y_row.push(record.get(target_start + d).unwrap_or("0.0").parse()?);
            }

            data_x.push(x_row);
            data_y.push(y_row);
        }

        Ok(Self::new(&data_x, &data_y, schema))
    }

    /// Get the subset of the data for better performance
    pub fn subset(&self, target_samples: usize) -> Self {
        let samples = target_samples.min(self.num_samples);
        if samples == self.num_samples { return self.clone(); }

        let simd_width = 8;
        let padding = if samples % simd_width == 0 { 0 } else { simd_width - (samples % simd_width) };
        let num_batches = (samples + padding) / simd_width;

        let num_features_usize = self.num_features as usize;
        let mut feature_flat = Vec::with_capacity(num_batches * num_features_usize);
        
        let mut target_batches = if !self.target_batches.is_empty() { Vec::with_capacity(num_batches) } else { vec![] };
        let mut target_mat2_batches = if self.target_mat2_batches.is_some() { Some(Vec::with_capacity(num_batches)) } else { None };
        let mut target_mat3_batches = if self.target_mat3_batches.is_some() { Some(Vec::with_capacity(num_batches)) } else { None };

        let step = (self.num_samples as f64 - 1.0) / (samples as f64 - 1.0).max(1.0);

        for i in 0..num_batches {
            let start_idx = i * simd_width;

            // ... IDE JÖN A FEATURE KISZEDÉS ...
            for f_idx in 0..num_features_usize {
                let mut batch_arr = [0.0; 8];
                for lane in 0..8 {
                    let orig_idx = (((start_idx + lane) as f64) * step).round() as usize;
                    if orig_idx < self.num_samples {
                        let b_idx = orig_idx / 8;
                        let l_idx = orig_idx % 8;
                        let vec_val = self.feature_flat[b_idx * num_features_usize + f_idx];
                        batch_arr[lane] = unsafe { (*(&vec_val as *const _ as *const [f32; 8]))[l_idx] };
                    }
                }
                feature_flat.push(wide::f32x8::new(batch_arr));
            }

            if !self.target_batches.is_empty() {
                let mut batch_arr = [0.0; 8];
                for lane in 0..8 {
                    let orig_idx = (((start_idx + lane) as f64) * step).round() as usize;
                    if orig_idx < self.num_samples {
                        let b_idx = orig_idx / 8;
                        let l_idx = orig_idx % 8;
                        let vec_val = self.target_batches[b_idx];
                        batch_arr[lane] = unsafe { (*(&vec_val as *const _ as *const [f32; 8]))[l_idx] };
                    }
                }
                target_batches.push(wide::f32x8::new(batch_arr));
            }

            if let Some(tb) = &self.target_mat2_batches {
                let mut mat2_batch = [wide::f32x8::splat(0.0); 4];
                for dim in 0..4 {
                    let mut batch_arr = [0.0; 8];
                    for lane in 0..8 {
                        let orig_idx = (((start_idx + lane) as f64) * step).round() as usize;
                        if orig_idx < self.num_samples {
                            let b_idx = orig_idx / 8;
                            let l_idx = orig_idx % 8;
                            let vec_val = tb[b_idx][dim];
                            batch_arr[lane] = unsafe { (*(&vec_val as *const _ as *const [f32; 8]))[l_idx] };
                        }
                    }
                    mat2_batch[dim] = wide::f32x8::new(batch_arr);
                }
                target_mat2_batches.as_mut().unwrap().push(mat2_batch);
            }

            if let Some(tb) = &self.target_mat3_batches {
                let mut mat3_batch = [wide::f32x8::splat(0.0); 9];
                for dim in 0..9 {
                    let mut batch_arr = [0.0; 8];
                    for lane in 0..8 {
                        let orig_idx = (((start_idx + lane) as f64) * step).round() as usize;
                        if orig_idx < self.num_samples {
                            let b_idx = orig_idx / 8;
                            let l_idx = orig_idx % 8;
                            let vec_val = tb[b_idx][dim];
                            batch_arr[lane] = unsafe { (*(&vec_val as *const _ as *const [f32; 8]))[l_idx] };
                        }
                    }
                    mat3_batch[dim] = wide::f32x8::new(batch_arr);
                }
                target_mat3_batches.as_mut().unwrap().push(mat3_batch);
            }
        }

        Self {
            feature_flat, target_batches, target_mat2_batches, target_mat3_batches,
            num_features: self.num_features, num_batches, num_samples: samples,
            feature_means: self.feature_means.clone(), feature_std_devs: self.feature_std_devs.clone(),
            target_mean: self.target_mean, target_std_dev: self.target_std_dev,
            target_variance: self.target_variance, is_normalized: self.is_normalized,
            feature_types: self.feature_types.clone(), extract_scalars: self.extract_scalars,
        }
    }
}
