use crate::engine::individual::Individual;
use crate::metrics::dataset::SimdDataset;
use crate::domain::Domain;
use crate::domain::universal::UniversalScalar;

const MAX_PARAMS: usize = 32;
const M: usize = 6;

pub fn run_lbfgs<D: Domain<ScalarValue = UniversalScalar>>(
    ind: &mut Individual<D>, 
    dataset: &SimdDataset, 
    max_iterations: usize
) {
    if ind.program.is_none() { 
        ind.compile(); 
    }
    let program = match ind.program.as_mut() { Some(p) => p, None => return, };

    let mut x = [0.0f32; MAX_PARAMS];
    let mut n = 0;
    
    for c in &program.constants {
        n += match c {
            UniversalScalar::Float(_) => 1,
            UniversalScalar::Vec2(_) => 2,
            UniversalScalar::Vec3(_) => 3,
            UniversalScalar::Mat2(_) => 4,
            UniversalScalar::Mat3(_) => 9,
            _ => 0,
        };
    }
    n = n.min(MAX_PARAMS);
    if n == 0 { return; }

    let mut ptr = 0;
    for c in &program.constants {
        match c {
            UniversalScalar::Float(f) => { if ptr < n { x[ptr] = *f; ptr += 1; } },
            UniversalScalar::Vec2(v) => { for i in 0..2 { if ptr < n { x[ptr] = v[i]; ptr += 1; } } },
            UniversalScalar::Vec3(v) => { for i in 0..3 { if ptr < n { x[ptr] = v[i]; ptr += 1; } } },
            UniversalScalar::Mat2(m) => { for i in 0..4 { if ptr < n { x[ptr] = m[i]; ptr += 1; } } },
            UniversalScalar::Mat3(m) => { for i in 0..9 { if ptr < n { x[ptr] = m[i]; ptr += 1; } } },
            _ => {}
        }
    }

    let update_constants = |prog_consts: &mut Vec<UniversalScalar>, flat_vals: &[f32; MAX_PARAMS]| {
        let mut p = 0;
        for c in prog_consts.iter_mut() {
            match c {
                UniversalScalar::Float(f) => { if p < n { *f = flat_vals[p]; p += 1; } },
                UniversalScalar::Vec2(v) => { for i in 0..2 { if p < n { v[i] = flat_vals[p]; p += 1; } } },
                UniversalScalar::Vec3(v) => { for i in 0..3 { if p < n { v[i] = flat_vals[p]; p += 1; } } },
                UniversalScalar::Mat2(m) => { for i in 0..4 { if p < n { m[i] = flat_vals[p]; p += 1; } } },
                UniversalScalar::Mat3(m) => { for i in 0..9 { if p < n { m[i] = flat_vals[p]; p += 1; } } },
                _ => {}
            }
        }
    };

    let mut s = [[0.0f32; MAX_PARAMS]; M];
    let mut y = [[0.0f32; MAX_PARAMS]; M];
    let mut rho = [0.0f32; M];
    let mut alpha = [0.0f32; M];
    
    let mut q = [0.0f32; MAX_PARAMS];
    
    let (mut current_mse, mut current_grad) = D::compute_mse_with_gradient(&program.code, &program.constants, dataset);
    
    let mut history_size = 0;
    let mut head = 0;

    for _iter in 0..max_iterations {
        let mut grad_norm_sq: f32 = 0.0;
        for i in 0..n { grad_norm_sq += current_grad[i] * current_grad[i]; }
        if grad_norm_sq.sqrt() < 1e-5 { break; }

        for i in 0..n { q[i] = current_grad[i]; }
        
        let mut curr_idx = head;
        for _ in 0..history_size {
            curr_idx = if curr_idx == 0 { M - 1 } else { curr_idx - 1 };
            let mut dot_sq = 0.0;
            for j in 0..n { dot_sq += s[curr_idx][j] * q[j]; }
            alpha[curr_idx] = rho[curr_idx] * dot_sq;
            
            for j in 0..n { q[j] -= alpha[curr_idx] * y[curr_idx][j]; }
        }

        let mut gamma = 1.0;
        if history_size > 0 {
            let last_idx = if head == 0 { M - 1 } else { head - 1 };
            let mut dot_sy = 0.0;
            let mut dot_yy = 0.0;
            for j in 0..n {
                dot_sy += s[last_idx][j] * y[last_idx][j];
                dot_yy += y[last_idx][j] * y[last_idx][j];
            }
            if dot_yy > 1e-10 { gamma = dot_sy / dot_yy; }
        }

        for j in 0..n { q[j] *= gamma; }

        curr_idx = if history_size < M { 0 } else { head };
        for _ in 0..history_size {
            let mut dot_yq = 0.0;
            for j in 0..n { dot_yq += y[curr_idx][j] * q[j]; }
            let beta = rho[curr_idx] * dot_yq;
            
            for j in 0..n { q[j] += s[curr_idx][j] * (alpha[curr_idx] - beta); }
            curr_idx = (curr_idx + 1) % M;
        }

        let mut p = [0.0f32; MAX_PARAMS];
        let mut dir_dot_grad = 0.0;
        for j in 0..n { 
            p[j] = -q[j]; 
            dir_dot_grad += p[j] * current_grad[j];
        }

        if dir_dot_grad > -1e-8 {
            for j in 0..n { p[j] = -current_grad[j]; }
            dir_dot_grad = -grad_norm_sq;
            history_size = 0;
        }

        let mut step_size = 1.0f32;
        let c1 = 1e-4;
        let mut next_x = [0.0f32; MAX_PARAMS];
        
        let mut ls_iters = 0;
        let next_mse = loop {
            for j in 0..n { next_x[j] = x[j] + step_size * p[j]; }
            update_constants(&mut program.constants, &next_x);
            
            let mse = D::compute_mse(&program.code, &program.constants, dataset);
            
            if mse <= current_mse + c1 * step_size * dir_dot_grad || ls_iters > 10 {
                break mse;
            }
            step_size *= 0.5;
            ls_iters += 1;
        };
        let (_, next_grad) = D::compute_mse_with_gradient(&program.code, &program.constants, dataset);

        let mut s_new = [0.0f32; MAX_PARAMS];
        let mut y_new = [0.0f32; MAX_PARAMS];
        let mut dot_sy = 0.0;

        for j in 0..n {
            s_new[j] = next_x[j] - x[j];
            y_new[j] = next_grad[j] - current_grad[j];
            dot_sy += s_new[j] * y_new[j];
        }

        if dot_sy > 1e-10 {
            for j in 0..n {
                s[head][j] = s_new[j];
                y[head][j] = y_new[j];
            }
            rho[head] = 1.0 / dot_sy;
            
            head = (head + 1) % M;
            if history_size < M { history_size += 1; }
        }

        for j in 0..n { x[j] = next_x[j]; }
        current_mse = next_mse;
        current_grad = next_grad;
    }

    update_constants(&mut program.constants, &x);
    ind.fitness = current_mse;
    
    let mut const_idx = 0;
    for node in ind.nodes.iter_mut() {
        if let crate::ast::node::Node::Constant(val, _) = node {
            if const_idx < program.constants.len() {
                *val = program.constants[const_idx];
                const_idx += 1;
            }
        }
    }
}