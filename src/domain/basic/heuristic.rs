use crate::ast::node::Node;
use crate::domain::basic::{BasicDomain, BasicOp};

#[derive(Clone, Copy, Debug)]
struct ExprInfo {
    start_idx: usize,
    const_val: Option<f32>,
}

const EPSILON: f32 = 1e-9;

pub fn simplify_ast(nodes: &[Node<BasicDomain>]) -> Vec<Node<BasicDomain>> {
    if nodes.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(nodes.len());
    let mut stack: Vec<ExprInfo> = Vec::with_capacity(32);

    for &node in nodes {
        match node {
            Node::Constant(val) => {
                let start_idx = output.len();
                output.push(node);
                stack.push(ExprInfo {
                    start_idx,
                    const_val: Some(val),
                });
            }
            Node::Variable(_) => {
                let start_idx = output.len();
                output.push(node);
                stack.push(ExprInfo {
                    start_idx,
                    const_val: None,
                });
            }
            Node::Operator(op) => {
                let arity = node.arity();

                if arity == 1 {
                    handle_unary(op, &mut stack, &mut output);
                } else if arity == 2 {
                    handle_binary(op, &mut stack, &mut output);
                }
            }
        }
    }
    output
}

#[inline(always)]
fn handle_unary(op: BasicOp, stack: &mut Vec<ExprInfo>, output: &mut Vec<Node<BasicDomain>>) {
    if let Some(arg) = stack.pop() {
        // Konstans folding (pl. sin(0) -> 0)
        if let Some(val) = arg.const_val {
            let res = match op {
                BasicOp::Sin => val.sin(),
                BasicOp::Cos => val.cos(),
                BasicOp::Exp => val.exp(),
                BasicOp::Sqr => val * val,
                BasicOp::Sqrt => val.abs().sqrt(),
                BasicOp::Ln => (val.abs() + EPSILON).ln(),
                _ => f32::NAN, 
            };

            if res.is_finite() {
                output.truncate(arg.start_idx);
                let new_start = output.len();
                output.push(Node::Constant(res));
                stack.push(ExprInfo { start_idx: new_start, const_val: Some(res) });
                return;
            }
        }
        output.push(Node::Operator(op));
        stack.push(ExprInfo { start_idx: arg.start_idx, const_val: None });
    }
}

#[inline(always)]
fn handle_binary(op: BasicOp, stack: &mut Vec<ExprInfo>, output: &mut Vec<Node<BasicDomain>>) {
    if let (Some(b), Some(a)) = (stack.pop(), stack.pop()) {
        
        // 2 konstans foldingja
        if let (Some(val_a), Some(val_b)) = (a.const_val, b.const_val) {
            let res = match op {
                BasicOp::Add => val_a + val_b,
                BasicOp::Sub => val_a - val_b,
                BasicOp::Mul => val_a * val_b,
                BasicOp::Div => if val_b.abs() < EPSILON { f32::NAN } else { val_a / val_b },
                _ => f32::NAN,
            };

            if res.is_finite() {
                output.truncate(a.start_idx);
                let new_start = output.len();
                output.push(Node::Constant(res));
                stack.push(ExprInfo { start_idx: new_start, const_val: Some(res) });
                return;
            }
        }

        // Algebrai egyszerűsítések
        match op {
            BasicOp::Add => {
                // x + 0 = x
                if is_zero(b.const_val) {
                    output.truncate(b.start_idx);
                    stack.push(a);
                    return;
                }
            },
            BasicOp::Sub => {
                // x - 0 = x
                if is_zero(b.const_val) {
                    output.truncate(b.start_idx);
                    stack.push(a);
                    return;
                }
                // x - x = 0
                if nodes_are_equal(output, a.start_idx, b.start_idx) {
                    output.truncate(a.start_idx);
                    let new_start = output.len();
                    output.push(Node::Constant(0.0));
                    stack.push(ExprInfo { start_idx: new_start, const_val: Some(0.0) });
                    return;
                }
            },
            BasicOp::Mul => {
                // x * 0 = 0 vagy 0 * x = 0
                if is_zero(a.const_val) || is_zero(b.const_val) {
                    output.truncate(a.start_idx);
                    let new_start = output.len();
                    output.push(Node::Constant(0.0));
                    stack.push(ExprInfo { start_idx: new_start, const_val: Some(0.0) });
                    return;
                }
                // x * 1 = x
                if is_one(b.const_val) {
                    output.truncate(b.start_idx);
                    stack.push(a);
                    return;
                }
            },
            BasicOp::Div => {
                // 0 / x = 0
                if is_zero(a.const_val) {
                    output.truncate(a.start_idx);
                    let new_start = output.len();
                    output.push(Node::Constant(0.0));
                    stack.push(ExprInfo { start_idx: new_start, const_val: Some(0.0) });
                    return;
                }
                // x / 1 = x
                if is_one(b.const_val) {
                    output.truncate(b.start_idx);
                    stack.push(a);
                    return;
                }
                // x / x = 1
                if nodes_are_equal(output, a.start_idx, b.start_idx) {
                    output.truncate(a.start_idx);
                    let new_start = output.len();
                    output.push(Node::Constant(1.0));
                    stack.push(ExprInfo { start_idx: new_start, const_val: Some(1.0) });
                    return;
                }
            },
            _ => {}
        }

        output.push(Node::Operator(op));
        stack.push(ExprInfo { start_idx: a.start_idx, const_val: None });
    }
}

#[inline(always)]
fn is_zero(val: Option<f32>) -> bool {
    val.map_or(false, |v| v.abs() < EPSILON)
}

#[inline(always)]
fn is_one(val: Option<f32>) -> bool {
    val.map_or(false, |v| (v - 1.0).abs() < EPSILON)
}

fn nodes_are_equal(buffer: &[Node<BasicDomain>], a_start: usize, b_start: usize) -> bool {
    let a_slice = &buffer[a_start..b_start];
    let b_slice = &buffer[b_start..];
    
    a_slice == b_slice
}