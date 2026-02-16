use crate::evolution::individual::Individual;

pub fn optimize_individual_constants(
    ind: &mut Individual, 
    data_x: &[Vec<f64>], 
    data_y: &[f64],
    iterations: usize,
    lr: f64,
    epsilon: f64
) {
    let mut consts = ind.get_constants();
    if consts.is_empty() { return; }

    for _ in 0..iterations {
        let current_mse = ind.calculate_mse(data_x, data_y);
        let mut gradients = vec![0.0; consts.len()];
        
        for i in 0..consts.len() {
            let original_val = consts[i];
            consts[i] = original_val + epsilon;
            ind.set_constants(&consts);
            let plus_mse = ind.calculate_mse(data_x, data_y);
            gradients[i] = (plus_mse - current_mse) / epsilon;
            
            consts[i] = original_val;
        }
        
        let mut improved = false;
        for i in 0..consts.len() {
            let grad = gradients[i].clamp(-10.0, 10.0);
            consts[i] -= lr * grad;
            
            if grad.abs() > 1e-6 { improved = true; }
        }

        ind.set_constants(&consts);
        if !improved { break; }
    }
}