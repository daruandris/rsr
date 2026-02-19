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
    build_ast_recursive(&mut nodes, 0, max_depth, rng, num_features, None);
    nodes
}

fn build_ast_recursive(
    nodes: &mut Vec<Node>, 
    current_depth: usize, 
    max_depth: usize, 
    rng: &mut impl RngExt, 
    num_features: u8
    , parent_op: Option<Op>
) {
    let is_terminal = current_depth >= max_depth || (current_depth > 0 && rng.random::<f32>() < 0.2);
    if is_terminal {
        if rng.random::<bool>() {
            nodes.push(Node::Variable(rng.random_range(0..num_features)));
        } else {
            nodes.push(Node::Constant(rng.random_range(-5.0..5.0)));
        }
    } else {
        let arity = if rng.random::<bool>() { 2 } else { 1 };
        let chosen_op = random_op_for_parent(parent_op, arity, rng);
        for _ in 0..arity {
            build_ast_recursive(nodes, current_depth+1, max_depth, rng, num_features, Some(chosen_op));
        }
        nodes.push(Node::Operator(chosen_op));
    }
}

pub fn random_op_for_parent(parent_op: Option<Op>, arity: usize, rng: &mut impl RngExt) -> Op {
    let all_ops_arity1 = [Op::Sin, Op::Cos, Op::Exp, Op::Sqr];
    let all_ops_arity2 = [Op::Add, Op::Sub, Op::Mul, Op::Div];

    let candidates: &[Op] = if arity == 1 { &all_ops_arity1 } else { &all_ops_arity2 };

    let forbidden = match parent_op {
        Some(op) => op.forbidden_children(),
        None => &[],
    };

    let allowed: Vec<Op> = candidates.iter()
        .filter(|op| !forbidden.contains(op))
        .cloned()
        .collect();

    if allowed.is_empty() {
        candidates[rng.random_range(0..candidates.len())]
    } else {
        allowed[rng.random_range(0..allowed.len())]
    }
}