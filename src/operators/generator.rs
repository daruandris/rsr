use crate::ast::node::{Node, Op};
use rand::RngExt;

#[inline(always)]
pub fn random_node_of_arity(arity: usize, rng: &mut impl RngExt, num_features: u8) -> Node {
    match arity {
        0 => {  
            if rng.random::<bool>() {
                Node::Variable(rng.random_range(0..num_features))
            } else {
                Node::Constant(rng.random_range(-5.0..5.0))
            }
        },
        1 => {
            let ops = [Op::Sin, Op::Cos, Op::Exp, Op::Sqr];
            Node::Operator(ops[rng.random_range(0..ops.len())])
        },
        2 => {
            let ops = [Op::Add, Op::Sub, Op::Mul, Op::Div];
            Node::Operator(ops[rng.random_range(0..ops.len())])
        },
        _ => unreachable!(),
    }
}

pub fn generate_random_ast(max_depth: usize, rng: &mut impl RngExt, num_features: u8) -> Vec<Node> {
    let cap = 1 << (max_depth.min(6)); 
    let mut nodes = Vec::with_capacity(cap);
    build_ast_recursive(&mut nodes, 0, max_depth, rng, num_features);
    nodes
}

fn build_ast_recursive(nodes: &mut Vec<Node>, current_depth: usize, max_depth: usize, rng: &mut impl RngExt, num_features: u8) {
    let is_terminal = current_depth >= max_depth || (current_depth > 0 && rng.random::<f64>() < 0.2);
    if is_terminal {
        let leaf = random_node_of_arity(0, rng, num_features);
        nodes.push(leaf);
    } else {
        let arity = if rng.random::<bool>() { 2 } else { 1 };
        for _ in 0..arity {
            build_ast_recursive(nodes, current_depth+1, max_depth, rng, num_features);
        }
        let op = random_node_of_arity(arity, rng, num_features);
        nodes.push(op);
    }
}