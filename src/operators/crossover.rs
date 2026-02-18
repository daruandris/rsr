use crate::evolution::individual::Individual;
use rand::RngExt;

pub fn crossover(parent_a: &Individual, parent_b: &Individual, rng: &mut impl RngExt) -> Individual {
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

    Individual { nodes: child_nodes, fitness: f64::MAX, age: 0 }
}