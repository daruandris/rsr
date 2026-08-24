use crate::engine::eval::types::ValueType;
use crate::engine::search::individual::Individual;
use rand::RngExt;

pub fn crossover(
    parent_a: &Individual,
    parent_b: &Individual,
    rng: &mut impl RngExt,
    max_size: usize,
) -> Individual {
    if parent_a.nodes.is_empty() || parent_b.nodes.is_empty() {
        return parent_a.clone();
    }

    let root_a = match select_node_index(parent_a, rng, None) {
        Some(idx) => idx,
        None => return parent_a.clone(),
    };

    let target_type = parent_a.nodes[root_a].get_type();
    let root_b_opt = select_node_index(parent_b, rng, Some(target_type));

    let root_b = match root_b_opt {
        Some(idx) => idx,
        None => return parent_a.clone(),
    };

    let (start_a, end_a) = parent_a.get_subtree_bounds(root_a);
    let (start_b, end_b) = parent_b.get_subtree_bounds(root_b);
    let new_len = start_a + (end_b - start_b + 1) + (parent_a.nodes.len() - end_a - 1);

    if new_len > max_size {
        return parent_a.clone();
    }

    let mut child_nodes = Vec::with_capacity(new_len);
    child_nodes.extend_from_slice(&parent_a.nodes[..start_a]);
    child_nodes.extend_from_slice(&parent_b.nodes[start_b..=end_b]);
    child_nodes.extend_from_slice(&parent_a.nodes[end_a + 1..]);

    Individual::new(child_nodes)
}

fn select_node_index(
    ind: &Individual,
    rng: &mut impl RngExt,
    required_type: Option<ValueType>,
) -> Option<usize> {
    let len = ind.nodes.len();
    if len == 0 {
        return None;
    }
    const MAX_NODES: usize = 64;
    let mut internal_indices = [0usize; MAX_NODES];
    let mut all_valid_indices = [0usize; MAX_NODES];
    let mut int_count = 0;
    let mut all_count = 0;

    for (i, node) in ind.nodes.iter().enumerate() {
        let is_type_match = required_type.is_none_or(|t| node.get_type() == t);
        if is_type_match {
            if all_count < MAX_NODES {
                all_valid_indices[all_count] = i;
                all_count += 1;
            }
            if node.arity() > 0 && int_count < MAX_NODES {
                internal_indices[int_count] = i;
                int_count += 1;
            }
        }
    }

    if all_count == 0 {
        return None;
    }

    if int_count > 0 && rng.random::<f32>() < 0.9 {
        let idx = rng.random_range(0..int_count);
        Some(internal_indices[idx])
    } else {
        let idx = rng.random_range(0..all_count);
        Some(all_valid_indices[idx])
    }
}
