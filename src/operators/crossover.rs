use crate::engine::individual::Individual;
use crate::domain::Domain;
use rand::RngExt;

pub fn crossover<D: Domain>(parent_a: &Individual<D>, parent_b: &Individual<D>, rng: &mut impl RngExt, max_size: usize) -> Individual<D> {
    if parent_a.nodes.is_empty() || parent_b.nodes.is_empty() {
        return parent_a.clone();
    }

    let root_a = select_node_index::<D>(parent_a, rng);
    let root_b = select_node_index::<D>(parent_b, rng);

    let (start_a, end_a) = parent_a.get_subtree_bounds(root_a);
    let (start_b, end_b) = parent_b.get_subtree_bounds(root_b);

    let new_len = start_a + (end_b - start_b + 1) + (parent_a.nodes.len() - end_a - 1);
    
    if new_len > max_size { return parent_a.clone(); }
    
    let mut child_nodes = Vec::with_capacity(new_len);
    child_nodes.extend_from_slice(&parent_a.nodes[..start_a]);
    child_nodes.extend_from_slice(&parent_b.nodes[start_b..=end_b]);
    child_nodes.extend_from_slice(&parent_a.nodes[end_a + 1..]);

    Individual::new(child_nodes)
}

fn select_node_index<D: Domain>(ind: &Individual<D>, rng: &mut impl RngExt) -> usize {
    let len = ind.nodes.len();
    if len < 2 { return 0; }

    let internal_indices: Vec<usize> = ind.nodes.iter().enumerate()
        .filter_map(|(i, n)| if n.arity() > 0 { Some(i) } else { None })
        .collect();

    if !internal_indices.is_empty() && rng.random::<f32>() < 0.9 {
        let idx = rng.random_range(0..internal_indices.len());
        internal_indices[idx]
    } else {
        rng.random_range(0..len)
    }
}