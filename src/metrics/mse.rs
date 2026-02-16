use crate::ast::eval::evaluate_ast;
use crate::ast::node::Node;

pub fn calculate_mse(nodes: &[Node], data_x: &[Vec<f64>], data_y: &[f64]) -> f64 {
    let mut sum_error = 0.0;
    for (i, row) in data_x.iter().enumerate() {
        let pred = evaluate_ast(nodes, row);
        let diff = pred - data_y[i];
        sum_error += diff * diff;
    }
    sum_error / (data_x.len() as f64)
}