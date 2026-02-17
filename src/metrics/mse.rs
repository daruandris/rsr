use crate::ast::eval::{evaluate_ast, evaluate_ast_simd};
use crate::ast::node::Node;
use crate::metrics::dataset::SimdDataset;

pub fn calculate_mse(nodes: &[Node], dataset: &SimdDataset) -> f64 {
    let mut sum_error = 0.0;

    for i in 0..dataset.x_batches.len() {
        let pred_batch = evaluate_ast_simd(nodes, &dataset.x_batches[i]);
        let target_batch = dataset.y_batches[i];
        
        let diff = pred_batch - target_batch;
        let squared_error = diff * diff;
        
        sum_error += squared_error.reduce_add(); 
    }

    for i in 0..dataset.remainder_x.len() {
        let pred = evaluate_ast(nodes, &dataset.remainder_x[i]);
        let diff = pred - dataset.remainder_y[i];
        sum_error += diff * diff;
    }

    if sum_error.is_nan() || sum_error.is_infinite() {
        return f64::MAX;
    }

    sum_error / (dataset.total_samples() as f64)
}