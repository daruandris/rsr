use crate::ast::node::Node;
use crate::evolution::individual::Individual;
use crate::operators::generator::{generate_random_ast, random_node_of_arity};
use rand::RngExt;
use rand_distr::{Distribution, Normal};

pub fn point_mutation(ind: &mut Individual, rng: &mut impl RngExt, num_features: u8) {
    if ind.nodes.is_empty() { return; }
    let idx = rng.random_range(0..ind.nodes.len());
    let target_arity = ind.nodes[idx].arity();
    ind.nodes[idx] = random_node_of_arity(target_arity, rng, num_features);
    ind.invalidate();
}

pub fn constant_perturbation(ind: &mut Individual, rng: &mut impl RngExt) {
    let mut target_idx = None;
    let mut count = 0;

    for (i, node) in ind.nodes.iter().enumerate() {
        if let Node::Constant(_) = node {
            count += 1;
            if rng.random_range(0..count) == 0 {
                target_idx = Some(i);
            }
        }
    }

    if let Some(idx) = target_idx {
        if let Node::Constant(ref mut val) = ind.nodes[idx] {
            let normal = Normal::new(0.0, 0.1).unwrap(); 
            *val += normal.sample(rng);
            
            ind.invalidate();
        }
    }
}

pub fn subtree_mutation(ind: &mut Individual, rng: &mut impl RngExt, num_features: u8) {
    if ind.nodes.is_empty() { return; }
    let root_idx = rng.random_range(0..ind.nodes.len());
    let (start, end) = ind.get_subtree_bounds(root_idx);
    let new_subtree = generate_random_ast(7, rng, num_features);
    ind.nodes.splice(start..=end, new_subtree);
    ind.invalidate();
}