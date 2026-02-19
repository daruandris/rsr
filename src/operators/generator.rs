use crate::ast::node::Node;
use crate::domain::Domain;
use rand::RngExt;

#[inline(always)]
pub fn random_node_of_arity<D: Domain>(arity: usize, rng: &mut impl RngExt, num_features: u8) -> Node<D> {
    match arity {
        0 => {  
            if rng.random::<bool>() {
                Node::Variable(rng.random_range(0..num_features))
            } else {
                Node::Constant(D::random_constant(rng))
            }
        },
        1 | 2 => {
            Node::Operator(D::random_operator(arity, None, rng))
        },
        _ => unreachable!(),
    }
}

pub fn generate_random_ast<D: Domain>(max_depth: usize, rng: &mut impl RngExt, num_features: u8) -> Vec<Node<D>> {
    let cap = 1 << (max_depth.min(6)); 
    let mut nodes = Vec::with_capacity(cap);
    build_ast_recursive::<D>(&mut nodes, 0, max_depth, rng, num_features, None);
    nodes
}

fn build_ast_recursive<D: Domain>(
    nodes: &mut Vec<Node<D>>, 
    current_depth: usize, 
    max_depth: usize, 
    rng: &mut impl RngExt, 
    num_features: u8, 
    parent_op: Option<D::Operator>
) {
    let is_terminal = current_depth >= max_depth || (current_depth > 0 && rng.random::<f32>() < 0.2);
    if is_terminal {
        if rng.random::<bool>() {
            nodes.push(Node::Variable(rng.random_range(0..num_features)));
        } else {
            nodes.push(Node::Constant(D::random_constant(rng)));
        }
    } else {
        let arity = if rng.random::<bool>() { 2 } else { 1 };
        let chosen_op = D::random_operator(arity, parent_op.clone(), rng);
        for _ in 0..arity {
            build_ast_recursive::<D>(nodes, current_depth+1, max_depth, rng, num_features, Some(chosen_op.clone()));
        }
        nodes.push(Node::Operator(chosen_op));
    }
}