use crate::ast::node::Node;
use crate::evolution::individual::Individual;
use crate::operators::generator::{generate_random_ast, random_node_of_arity};
use rand::RngExt;
use rand_distr::{Distribution, Normal};

pub fn point_mutation(ind: &mut Individual, rng: &mut impl RngExt, num_features: usize) {
    if ind.nodes.is_empty() { return; }
    let idx = rng.random_range(0..ind.nodes.len());
    let target_arity = ind.nodes[idx].arity();
    ind.nodes[idx] = random_node_of_arity(target_arity, rng, num_features);
    ind.fitness = f64::MAX;
}

pub fn constant_perturbation(ind: &mut Individual, rng: &mut impl RngExt) {
    let const_indices: Vec<usize> = ind.nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| if let Node::Constant(_) = node { Some(i) } else { None })
        .collect();
        
    if const_indices.is_empty() { return; }
    
    let idx = const_indices[rng.random_range(0..const_indices.len())];
    if let Node::Constant(ref mut val) = ind.nodes[idx] {
        let normal = Normal::new(0.0, 0.1).unwrap();
        *val += normal.sample(rng);
    }

    ind.fitness = f64::MAX;
}

pub fn subtree_mutation(ind: &mut Individual, rng: &mut impl RngExt, num_features: usize) {
    if ind.nodes.is_empty() { return; }
    let root_idx = rng.random_range(0..ind.nodes.len());
    let (start, end) = ind.get_subtree_bounds(root_idx);
    let new_subtree = generate_random_ast(7, rng, num_features);
    ind.nodes.splice(start..=end, new_subtree);
    ind.fitness = f64::MAX;
}