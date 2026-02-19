use crate::ast::node::Node;
use crate::evolution::individual::Individual;
use crate::operators::generator::{generate_random_ast, random_node_of_arity};
use rand::RngExt;

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
            // Multiplikatív perturbáció (Log-Normal szerű viselkedés)
            // 80% esély: kicsit szorozzuk meg (0.9 ... 1.1) - finomhangolás
            // 10% esély: additív zaj (ha esetleg 0 lenne az érték)
            // 10% esély: teljesen új random szám (menekülés lokális minimumból)
            
            let r = rng.random::<f32>();
            if r < 0.8 {
                let factor = rng.random_range(0.9..1.1); 
                *val *= factor;
            } else if r < 0.9 {
                let shift = rng.random_range(-0.1..0.1);
                *val += shift;
            } else {
                *val = rng.random_range(-5.0..5.0);
            }
            
            ind.invalidate();
        }
    }
}

pub fn subtree_mutation(ind: &mut Individual, rng: &mut impl RngExt, num_features: u8) {
    if ind.nodes.is_empty() { return; }
    let mutation_point = rng.random_range(0..ind.nodes.len());
    let (start, end) = ind.get_subtree_bounds(mutation_point);
    let new_subtree = generate_random_ast(5, rng, num_features);
    ind.nodes.splice(start..=end, new_subtree);
    ind.invalidate();
}