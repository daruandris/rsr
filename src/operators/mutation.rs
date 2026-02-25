use crate::ast::node::Node;
use crate::engine::individual::Individual;
use crate::domain::Domain;
use crate::operators::generator::generate_random_ast;
use rand::RngExt;

pub fn point_mutation<D: Domain>(ind: &mut Individual<D>, rng: &mut impl RngExt, variables: &[(D::TypeId, u8)], allowed_ops: &[D::Operator]) {
    if ind.nodes.is_empty() { return; }
    let idx = rng.random_range(0..ind.nodes.len());
    
    let target_node = &ind.nodes[idx];
    let target_arity = target_node.arity();
    let target_type = target_node.get_type();
    
    if target_arity == 0 {
        let valid_vars: Vec<_> = variables.iter().filter(|v| v.0 == target_type).collect();
        let is_var_valid = !valid_vars.is_empty();
        let maybe_const = D::random_constant(target_type, rng);
        
        match (is_var_valid, maybe_const) {
            (true, Some(c)) => {
                if rng.random::<bool>() {
                    let chosen = valid_vars[rng.random_range(0..valid_vars.len())];
                    ind.nodes[idx] = Node::Variable(chosen.1, target_type);
                } else {
                    ind.nodes[idx] = Node::Constant(c, target_type);
                }
            },
            (true, None) => {
                let chosen = valid_vars[rng.random_range(0..valid_vars.len())];
                ind.nodes[idx] = Node::Variable(chosen.1, target_type);
            },
            (false, Some(c)) => {
                ind.nodes[idx] = Node::Constant(c, target_type);
            },
            (false, None) => {
            }
        }
    } else {
        if let Some(new_op) = D::random_operator(target_type, allowed_ops, None, rng) {
            if let Node::Operator(old_op) = target_node {
                if D::expected_types(&new_op) == D::expected_types(old_op) {
                    ind.nodes[idx] = Node::Operator(new_op);
                }
            }
        }
    }
    ind.invalidate();
}

pub fn constant_perturbation<D: Domain>(ind: &mut Individual<D>, rng: &mut impl RngExt) {
    let mut target_idx = None;
    let mut count = 0;

    for (i, node) in ind.nodes.iter().enumerate() {
        if let Node::Constant(_, _) = node {
            count += 1;
            if rng.random_range(0..count) == 0 {
                target_idx = Some(i);
            }
        }
    }

    if let Some(idx) = target_idx {
        if let Node::Constant(ref mut val, _) = ind.nodes[idx] {
            D::perturb_constant(val, rng);
            ind.invalidate();
        }
    }
}

pub fn subtree_mutation<D: Domain>(
    ind: &mut Individual<D>, 
    rng: &mut impl RngExt, 
    variables: &[(D::TypeId, u8)],
    max_size: usize,
    mutation_max_depth: usize,
    allowed_ops: &[D::Operator],
) {
    if ind.nodes.is_empty() { return; }
    let mutation_point = rng.random_range(0..ind.nodes.len());
    let (start, end) = ind.get_subtree_bounds(mutation_point);

    let required_type = ind.nodes[end].get_type();
    
    let removed_len = end - start + 1;
    let current_len = ind.nodes.len();
    let allowed_new_len = max_size.saturating_sub(current_len - removed_len);
    
    if allowed_new_len == 0 { return; }

    let new_subtree = generate_random_ast::<D>(required_type, mutation_max_depth, rng, variables, allowed_ops);
    
    if new_subtree.len() > allowed_new_len { return; }

    ind.nodes.splice(start..=end, new_subtree);
    ind.invalidate();
}