use crate::ast::node::Node;
use crate::engine::individual::Individual;
use crate::domain::Domain;
use crate::operators::generator::{generate_random_ast, random_node_of_arity};
use rand::RngExt;

pub fn point_mutation<D: Domain>(ind: &mut Individual<D>, rng: &mut impl RngExt, num_features: u8) {
    if ind.nodes.is_empty() { return; }
    let idx = rng.random_range(0..ind.nodes.len());
    let target_arity = ind.nodes[idx].arity();
    ind.nodes[idx] = random_node_of_arity::<D>(target_arity, rng, num_features);
    ind.invalidate();
}

pub fn constant_perturbation<D: Domain>(ind: &mut Individual<D>, rng: &mut impl RngExt) {
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
            D::perturb_constant(val, rng); // Generikus hívás!
            ind.invalidate();
        }
    }
}

pub fn subtree_mutation<D: Domain>(
    ind: &mut Individual<D>, 
    rng: &mut impl RngExt, 
    num_features: u8, 
    max_size: usize,
    mutation_max_depth: usize
) {
    if ind.nodes.is_empty() { return; }
    let mutation_point = rng.random_range(0..ind.nodes.len());
    let (start, end) = ind.get_subtree_bounds(mutation_point);

    let removed_len = end - start + 1;
    let current_len = ind.nodes.len();
    let allowed_new_len = max_size.saturating_sub(current_len - removed_len);
    if allowed_new_len == 0 { return; }

    let new_subtree = generate_random_ast::<D>(mutation_max_depth, rng, num_features);

    if new_subtree.len() > allowed_new_len { return; }

    ind.nodes.splice(start..=end, new_subtree);
    ind.invalidate();
}