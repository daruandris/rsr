use crate::ast::bytecode::Program;
use crate::metrics::dataset::SimdDataset;

pub fn calculate_mse_simd(program: &Program, dataset: &SimdDataset) -> f32 {
    let mut sum_squared_error = 0.0;
    let num_features = dataset.num_features as usize;
    let flat_features = &dataset.feature_flat;
    let targets = &dataset.target_batches;

    for i in 0..dataset.num_batches {
        let start = i * num_features;
        let input_batch = unsafe {
            flat_features.get_unchecked(start..start + num_features)
        };

        let prediction = program.eval_simd(input_batch);
        
        let target = unsafe { *targets.get_unchecked(i) };

        let diff = prediction - target;
        let sqr = diff * diff;

        sum_squared_error += sqr.reduce_add();
    }

    if !sum_squared_error.is_finite() {
        return f32::MAX;
    }
    sum_squared_error / (dataset.num_samples as f32)
}