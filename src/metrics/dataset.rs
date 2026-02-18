use wide::f32x4;


pub struct SimdDataset {
    pub feature_flat: Vec<f32x4>,
    pub target_batches: Vec<f32x4>,
    pub num_features: u8,
    pub num_batches: usize,
    pub num_samples: usize,
    pub feature_means: Vec<f32>,
    pub feature_std_devs: Vec<f32>,
    pub target_mean: f32,
    pub target_std_dev: f32,
}

impl SimdDataset {
    pub fn new(data_x: &[Vec<f32>], data_y: &[f32], num_features: u8) -> Self {
        let num_samples = data_x.len();
        let num_features_usize = num_features as usize;        
        let mut feature_means = vec![0.0; num_features_usize];
        let mut feature_std_devs = vec![1.0; num_features_usize];

        for f_idx in 0..num_features_usize {
            let sum: f32 = data_x.iter().map(|row| row[f_idx]).sum();
            let mean = sum / num_samples as f32;
            feature_means[f_idx] = mean;

            let variance: f32 = data_x.iter()
                .map(|row| (row[f_idx] - mean).powi(2))
                .sum::<f32>() / num_samples as f32;
            
            feature_std_devs[f_idx] = if variance < 1e-9 { 1.0 } else { variance.sqrt() };
        }

        let target_sum: f32 = data_y.iter().sum();
        let target_mean = target_sum / num_samples as f32;
        
        let target_variance: f32 = data_y.iter()
            .map(|&y| (y - target_mean).powi(2))
            .sum::<f32>() / num_samples as f32;
            
        let target_std_dev = if target_variance < 1e-9 { 1.0 } else { target_variance.sqrt() };

        let simd_width = 4;
        let remainder = num_samples % simd_width;
        let padding = if remainder == 0 { 0 } else { simd_width - remainder };
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
        }
    }


    /// Képlet: y_orig = y_norm * std + mean
    #[inline(always)]
    pub fn denormalize_target(&self, normalized_val: f32) -> f32 {
        normalized_val * self.target_std_dev + self.target_mean
    }

    #[inline(always)]
    pub fn denormalize_feature(&self, normalized_val: f32, feature_idx: usize) -> f32 {
        if feature_idx < self.feature_means.len() {
            normalized_val * self.feature_std_devs[feature_idx] + self.feature_means[feature_idx]
        } else {
            normalized_val
        }
    }
    
    #[inline(always)]
    pub fn denormalize_target_simd(&self, normalized_batch: f32x4) -> f32x4 {
        let std = f32x4::splat(self.target_std_dev);
        let mean = f32x4::splat(self.target_mean);
        normalized_batch * std + mean
    }
}

