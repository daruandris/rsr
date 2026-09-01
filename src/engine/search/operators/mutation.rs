use crate::Instruction;
use crate::engine::eval::scalar::Scalar;
use crate::engine::eval::types::ValueType;
use crate::engine::expr::node::Node;
use crate::engine::search::individual::Individual;
use crate::engine::search::operators::generator::{
    generate_random_ast, random_constant, random_operator,
};
use rand::RngExt;

pub fn point_mutation(
    ind: &mut Individual,
    rng: &mut impl RngExt,
    variables: &[(ValueType, u8)],
    allowed_ops: &[Instruction],
    disabled_constants: &[ValueType],
) {
    if ind.nodes.is_empty() {
        return;
    }
    let idx = rng.random_range(0..ind.nodes.len());

    let target_node = ind.nodes[idx];
    let target_arity = target_node.arity();
    let target_type = target_node.get_type();

    if target_arity == 0 {
        let valid_vars: Vec<_> = variables.iter().filter(|v| v.0 == target_type).collect();
        let is_var_valid = !valid_vars.is_empty();
        let maybe_const = random_constant(target_type, disabled_constants, rng);

        match (is_var_valid, maybe_const) {
            (true, Some(c)) => {
                if rng.random::<bool>() {
                    let chosen = valid_vars[rng.random_range(0..valid_vars.len())];
                    ind.nodes[idx] = Node::Variable(chosen.1, target_type);
                } else {
                    let c_idx = ind.constants.len() as u16;
                    ind.constants.push(c);
                    ind.nodes[idx] = Node::Constant(c_idx, target_type);
                }
            }
            (true, None) => {
                let chosen = valid_vars[rng.random_range(0..valid_vars.len())];
                ind.nodes[idx] = Node::Variable(chosen.1, target_type);
            }
            (false, Some(c)) => {
                let const_idx = ind.constants.len() as u16;
                ind.constants.push(c);
                ind.nodes[idx] = Node::Constant(const_idx, target_type);
            }
            (false, None) => {}
        }
    } else if let Some(new_op) = random_operator(target_type, allowed_ops, None, rng)
        && let Node::Operator(old_op) = target_node
        && new_op.expected_types() == old_op.expected_types()
    {
        ind.nodes[idx] = Node::Operator(new_op);
    }
    ind.invalidate();
}

pub fn constant_perturbation(ind: &mut Individual, rng: &mut impl RngExt) {
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

    if let Some(idx) = target_idx
        && let Node::Constant(c_idx, _) = ind.nodes[idx]
    {
        perturb_constant(&mut ind.constants[c_idx as usize], rng);
        ind.invalidate();
    }
}

fn perturb_constant(val: &mut Scalar, rng: &mut impl RngExt) {
    let r = rng.random::<f32>();
    match val {
        Scalar::Float(f) => {
            if r < 0.8 {
                *f *= rng.random_range(0.9..1.1);
            } else if r < 0.9 {
                *f += rng.random_range(-0.1..0.1);
            } else {
                *f = rng.random_range(-5.0..5.0);
            }
        }
        Scalar::Vec2(v) => {
            for item in v {
                *item += rng.random_range(-0.5..0.5);
            }
        }
        Scalar::Vec3(v) => {
            for item in v {
                *item += rng.random_range(-0.5..0.5);
            }
        }
        Scalar::Mat2(m) => {
            for item in m {
                *item += rng.random_range(-0.5..0.5);
            }
        }
        Scalar::Mat3(m) => {
            for item in m {
                *item += rng.random_range(-0.5..0.5);
            }
        }
        _ => {}
    }
}

pub fn subtree_mutation(
    ind: &mut Individual,
    rng: &mut impl RngExt,
    variables: &[(ValueType, u8)],
    max_size: usize,
    mutation_max_depth: usize,
    allowed_ops: &[Instruction],
    disabled_constants: &[ValueType],
) {
    if ind.nodes.is_empty() {
        return;
    }
    let mutation_point = rng.random_range(0..ind.nodes.len());
    let (start, end) = ind.get_subtree_bounds(mutation_point);

    let required_type = ind.nodes[end].get_type();
    let removed_len = end - start + 1;
    let current_len = ind.nodes.len();
    let allowed_new_len = max_size.saturating_sub(current_len - removed_len);

    if allowed_new_len == 0 {
        return;
    }

    let (mut new_subtree, new_consts) = generate_random_ast(
        required_type,
        mutation_max_depth,
        rng,
        variables,
        allowed_ops,
        disabled_constants
    );

    let const_offset = ind.constants.len() as u16;
    for node in &mut new_subtree {
        if let Node::Constant(idx, _) = node {
            *idx += const_offset;
        }
    }

    ind.constants.extend(new_consts);
    ind.nodes.splice(start..=end, new_subtree);
    ind.invalidate();
}
