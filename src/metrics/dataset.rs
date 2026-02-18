use wide::f32x4;


pub struct SimdDataset {
    pub feature_flat: Vec<f32x4>,
    pub target_batches: Vec<f32x4>,
    pub num_features: u8,
    pub num_batches: usize,
    pub num_samples: usize,
}

impl SimdDataset {
    pub fn new(data_x: &[Vec<f32>], data_y: &[f32], num_features: u8) -> Self {
        let num_samples = data_x.len();
        let num_features_usize = num_features as usize;
        let simd_width = 4;
        let remainder = num_samples % simd_width;
        let padding = if remainder == 0 { 0 } else { simd_width - remainder };
        let padded_size = num_samples + padding;
        let num_batches = padded_size / simd_width;

        let mut feature_flat = Vec::with_capacity(num_batches * num_features_usize);
        let mut target_batches = Vec::with_capacity(num_batches);

        let get_sample = |idx: usize, feature_idx: usize| -> f32 {
            if idx < num_samples {
                data_x[idx][feature_idx]
            } else {
                0.0
            }
        };

        let get_target = |idx: usize| -> f32 {
            if idx < num_samples {
                data_y[idx]
            } else {
                0.0
            }
        };

        for i in 0..num_batches {
            let start_idx = i * simd_width;
            
            for f_idx in 0..num_features_usize {
                let batch = f32x4::new([
                    get_sample(start_idx, f_idx),
                    get_sample(start_idx + 1, f_idx),
                    get_sample(start_idx + 2, f_idx),
                    get_sample(start_idx + 3, f_idx),
                ]);
                feature_flat.push(batch);
            }

            let target_batch = f32x4::new([
                get_target(start_idx),
                get_target(start_idx + 1),
                get_target(start_idx + 2),
                get_target(start_idx + 3),
            ]);
            target_batches.push(target_batch);
        }

        Self {
            feature_flat,
            target_batches,
            num_features: num_features,
            num_batches,
            num_samples,
        }
    }
}

