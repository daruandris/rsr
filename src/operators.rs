use rand::{RngExt};
use rand_distr::{Distribution, Normal};

use crate::{individual::Individual};
use crate::node::{Node, Op};

pub fn crossover(
    parent_a: &Individual, 
    parent_b: &Individual, 
    rng: &mut impl RngExt
) -> Individual {
    if parent_a.nodes.is_empty() || parent_b.nodes.is_empty() {
        return parent_a.clone();
    }
    let root_a = rng.random_range(0..parent_a.nodes.len());
    let root_b = rng.random_range(0..parent_b.nodes.len());

    let (start_a, end_a) = parent_a.get_subtree_bounds(root_a);
    let (start_b, end_b) = parent_b.get_subtree_bounds(root_b);

    let mut child_nodes = Vec::with_capacity(parent_a.nodes.len() + (end_b - start_b + 1));
    child_nodes.extend_from_slice(&parent_a.nodes[..start_a]);
    child_nodes.extend_from_slice(&parent_b.nodes[start_b..=end_b]);
    child_nodes.extend_from_slice(&parent_a.nodes[end_a + 1..]);

    if child_nodes.len() > 50 { return parent_a.clone(); }

    Individual { nodes: child_nodes, fitness: f64::MAX }
}

pub fn tournament_selection<'a>(
    population: &'a [Individual],
    k: usize,
    rng: &mut impl RngExt
) -> &'a Individual {
    let mut best = &population[rng.random_range(0..population.len())];
    for _ in 0..k {
        let contender = &population[rng.random_range(0..population.len())];
        if contender.fitness < best.fitness {
            best = contender;
        }
    }

    best
}

fn random_node_of_arity(arity: usize, rng: &mut impl RngExt, num_features: usize) -> Node {
    match arity {
        0 => {  
            if rng.random::<bool>() {
                Node::Variable(rng.random_range(0..num_features))
            }
            else{
                Node::Constant(rng.random_range(-5.0..5.0))
            }
        },
        1 => {
            let ops = [Op::Sin, Op::Cos, Op::Exp];
            Node::Operator(ops[rng.random_range(0..ops.len())])
        },
        2 => {
            let ops = [Op::Add, Op::Sub, Op::Mul, Op::Div];
            Node::Operator(ops[rng.random_range(0..ops.len())])
        },
        _ => unreachable!(),
    }
}

pub fn point_mutation(
    ind: &mut Individual,
    rng: &mut impl RngExt,
    num_features: usize
)
{
    if ind.nodes.is_empty(){ return; }
    let idx = rng.random_range(0..ind.nodes.len());
    let target_arity = ind.nodes[idx].arity();
    ind.nodes[idx] = random_node_of_arity(target_arity, rng, num_features);
    ind.fitness = f64::MAX;
}

pub fn constant_perturbation(ind: &mut Individual, rng: &mut impl RngExt){
    let const_indices: Vec<usize> = ind.nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| if let Node::Constant(_) = node {Some(i)} else {None})
        .collect();
    if const_indices.is_empty(){ return; }
    let idx = const_indices[rng.random_range(0..const_indices.len())];
    if let Node::Constant(ref mut val) = ind.nodes[idx] {
        let normal = Normal::new(0.0, 0.1).unwrap();
        *val += normal.sample(rng);
    }

    ind.fitness = f64::MAX;
}

pub fn subtree_mutation(
    ind: &mut Individual,
    rng: &mut impl RngExt,
    num_features: usize
)
{
    if ind.nodes.is_empty() { return; }
    let root_idx = rng.random_range(0..ind.nodes.len());
    let (start, end) = ind.get_subtree_bounds(root_idx);
    let new_subtree = generate_random_ast(7, rng, num_features);
    ind.nodes.splice(start..=end, new_subtree);
    ind.fitness = f64::MAX;
}

pub fn generate_random_ast(
    max_depth: usize, 
    rng: &mut impl RngExt, 
    num_features: usize
) -> Vec<Node>{
    let mut nodes = Vec::new();
    build_ast_recursive(&mut nodes, 0, max_depth, rng, num_features);
    nodes
}

fn build_ast_recursive(
    nodes: &mut Vec<Node>,
    current_depth: usize,
    max_depth: usize,
    rng: &mut impl RngExt,
    num_features: usize
){
    let is_terminal = current_depth >= max_depth || (current_depth > 0 && rng.random::<f64>() < 0.2);
    if is_terminal{
        let leaf = random_node_of_arity(0, rng, num_features);
        nodes.push(leaf);
    }
    else{
        let arity = if rng.random::<bool>() {2} else{1};
        for _ in 0..arity {
            build_ast_recursive(nodes, current_depth+1, max_depth, rng, num_features);
        }

        let op = random_node_of_arity(arity, rng, num_features);
        nodes.push(op);
    }
}