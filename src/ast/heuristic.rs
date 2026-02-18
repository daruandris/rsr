use crate::ast::node::{Node, Op};

#[derive(Clone, Copy, Debug)]
struct ExprInfo {
    start_idx: usize,
    const_val: Option<f32>,
}

const EPSILON: f32 = 1e-9;

pub fn simplify_ast(nodes: &[Node]) -> Vec<Node> {
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
fn handle_unary(op: Op, stack: &mut Vec<ExprInfo>, output: &mut Vec<Node>) {
    if let Some(arg) = stack.pop() {
        // Konstans folding (pl. sin(0) -> 0)
        if let Some(val) = arg.const_val {
            let res = match op {
                Op::Sin => val.sin(),
                Op::Cos => val.cos(),
                Op::Exp => val.exp(),
                Op::Sqr => val * val,
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
fn handle_binary(op: Op, stack: &mut Vec<ExprInfo>, output: &mut Vec<Node>) {
    if let (Some(b), Some(a)) = (stack.pop(), stack.pop()) {
        
        // 2 konstans
        if let (Some(val_a), Some(val_b)) = (a.const_val, b.const_val) {
            let res = match op {
                Op::Add => val_a + val_b,
                Op::Sub => val_a - val_b,
                Op::Mul => val_a * val_b,
                Op::Div => if val_b.abs() < EPSILON { f32::NAN } else { val_a / val_b },
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

        //algebrai egyszerűsytések
        match op {
            Op::Add => {
                // x + 0 = x
                if is_zero(b.const_val) {
                    output.truncate(b.start_idx);
                    stack.push(a);
                    return;
                }
                // 0 + x = x
                if is_zero(a.const_val) {
                    // Ez trükkös RPN-nél: [0, x, +] -> [x]
                    // A 0-t törölni kell, az x-et előre kell mozgatni? 
                    // Mivel ez drága (memmove), RPN-ben egyszerűbb, ha hagyjuk,
                    // VAGY ha a 0 nagyon az elején van, rewrite.
                    // Optimalizáció: inkább hagyjuk most, vagy swap? 
                    // Mivel a remove lassú, hagyjuk a '0 + x'-et, majd a kövi pass kiszedi,
                    // vagy: ha 'x' rövid, átmásoljuk.
                }
            },
            Op::Sub => {
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
            Op::Mul => {
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
                // 1 * x = x 
            },
            Op::Div => {
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

fn nodes_are_equal(buffer: &[Node], a_start: usize, b_start: usize) -> bool {
    let a_slice = &buffer[a_start..b_start];
    let b_slice = &buffer[b_start..];
    
    a_slice == b_slice
}