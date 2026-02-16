use crate::ast::node::{Node, Op};

pub fn evaluate_ast(nodes: &[Node], features: &[f64]) -> f64 {
    let mut stack = Vec::with_capacity(32);

    for node in nodes {
        match node {
            Node::Constant(c) => stack.push(*c),
            Node::Variable(v) => stack.push(features[*v]),
            Node::Operator(op) => apply_operator(*op, &mut stack),
        }
    }
    stack.pop().unwrap_or(f64::NAN)
}

#[inline(always)]
fn apply_operator(op: Op, stack: &mut Vec<f64>) {
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
        Op::Sin | Op::Cos | Op::Exp => {
            if let Some(a) = stack.pop() {
                let result = match op {
                    Op::Sin => a.sin(),
                    Op::Cos => a.cos(),
                    Op::Exp => {
                        let val = a.exp();
                        if val.is_finite() { val } else { f64::MAX }
                    },
                    _ => unreachable!(),
                };
                stack.push(result);
            }
        }
    }
}