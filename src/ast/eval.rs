use wide::{f32x4, CmpLt};

use crate::ast::node::{Node, Op};

pub fn evaluate_ast(nodes: &[Node], features: &[f32]) -> f32 {
    let mut stack = Vec::with_capacity(32);

    for node in nodes {
        match node {
            Node::Constant(c) => stack.push(*c),
            Node::Variable(v) => stack.push(features[*v as usize]),
            Node::Operator(op) => apply_operator(*op, &mut stack),
        }
    }
    stack.pop().unwrap_or(f32::NAN)
}

#[inline(always)]
fn apply_operator(op: Op, stack: &mut Vec<f32>) {
    match op {
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            if let (Some(b), Some(a)) = (stack.pop(), stack.pop()) {
                let result = match op {
                    Op::Add => a + b,
                    Op::Sub => a - b,
                    Op::Mul => a * b,
                    Op::Div => {
                        if b.abs() < 1e-9 { 1.0 } else { a / b }
                    },
                    _ => unreachable!()
                };
                stack.push(result);
            }
        },
        Op::Sin | Op::Cos | Op::Exp | Op::Sqr => {
            if let Some(a) = stack.pop() {
                let result = match op {
                    Op::Sin => a.sin(),
                    Op::Cos => a.cos(),
                    Op::Exp => {
                        let val = a.exp();
                        if val.is_finite() { val } else { f32::MAX }
                    },
                    Op::Sqr => a * a,
                    _ => unreachable!(),
                };
                stack.push(result);
            }
        }
    }
}

pub fn evaluate_ast_simd(nodes: &[Node], features: &[f32x4]) -> f32x4 {
    let mut stack: Vec<f32x4> = Vec::with_capacity(32);

    for node in nodes {
        match node {
            Node::Constant(c) => stack.push(f32x4::splat(*c)),
            Node::Variable(v) => stack.push(features[*v as usize]),
            Node::Operator(op) => apply_operator_simd(*op, &mut stack),
        }
    }
    stack.pop().unwrap_or_else(|| f32x4::splat(f32::NAN))
}

#[inline(always)]
fn apply_operator_simd(op: Op, stack: &mut Vec<f32x4>) {
    match op {
        Op::Add | Op::Sub | Op::Mul | Op::Div => {
            if let (Some(b), Some(a)) = (stack.pop(), stack.pop()) {
                let result = match op {
                    Op::Add => a + b,
                    Op::Sub => a - b,
                    Op::Mul => a * b,
                    Op::Div => {
                        let epsilon = f32x4::splat(1e-9);
                        let ones = f32x4::splat(1.0);
                        let b_abs = b.abs();
                        
                        let is_zero_mask = b_abs.simd_lt(epsilon);                    
                        let safe_b = is_zero_mask.blend(ones, b);
                        a / safe_b
                    },
                    _ => unreachable!()
                };
                stack.push(result);
            }
        },
        Op::Sin | Op::Cos | Op::Exp | Op::Sqr => {
            if let Some(a) = stack.pop() {
                let result = match op {
                    Op::Sin => a.sin(),
                    Op::Cos => a.cos(),
                    Op::Exp => a.exp(),
                    // (Az exp infinity check-et itt most a sebesség miatt elhagyjuk, 
                    // de a MSE számításnál a f64::NAN / INF úgyis f64::MAX büntetést kap)
                    Op::Sqr => a * a,
                    _ => unreachable!(),
                };
                stack.push(result);
            }
        }
    }
}

#[allow(dead_code)]
fn fallback_simd_example(a: f32x4) -> f32x4 {
    let arr = a.to_array(); 
    
    let res0 = custom_slow_function(arr[0]);
    let res1 = custom_slow_function(arr[1]);
    let res2 = custom_slow_function(arr[2]);
    let res3 = custom_slow_function(arr[3]);
    f32x4::new([res0, res1, res2, res3]) 
}

fn custom_slow_function(x: f32) -> f32 {
    x.powf(1.5) // Csak egy példa
}