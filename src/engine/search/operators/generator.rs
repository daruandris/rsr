// src/engine/search/operators/generator.rs

use crate::Instruction;
use crate::engine::eval::scalar::Scalar;
use crate::engine::eval::types::ValueType;
use crate::engine::expr::node::Node;
use rand::RngExt;

pub fn generate_random_ast(
    target_type: ValueType,
    max_depth: usize,
    rng: &mut impl RngExt,
    variables: &[(ValueType, u8)],
    allowed_ops: &[Instruction],
) -> (Vec<Node>, Vec<Scalar>) {
    let cap = 1 << (max_depth.min(6));
    let mut nodes = Vec::with_capacity(cap);
    let mut constants = Vec::new();

    build_ast_recursive(
        &mut nodes,
        &mut constants,
        target_type,
        0,
        max_depth,
        rng,
        variables,
        allowed_ops,
        None,
    );

    (nodes, constants)
}

fn build_ast_recursive(
    nodes: &mut Vec<Node>,
    constants: &mut Vec<Scalar>,
    target_type: ValueType,
    current_depth: usize,
    max_depth: usize,
    rng: &mut impl RngExt,
    variables: &[(ValueType, u8)],
    allowed_ops: &[Instruction],
    parent_op: Option<Instruction>,
) {
    let is_terminal =
        current_depth >= max_depth || (current_depth > 0 && rng.random::<f32>() < 0.2);

    if is_terminal {
        let valid_vars: Vec<_> = variables.iter().filter(|v| v.0 == target_type).collect();
        let is_var_valid = !valid_vars.is_empty();
        let maybe_const = random_constant(target_type, rng);

        match (is_var_valid, maybe_const) {
            (true, Some(c)) => {
                if rng.random::<bool>() {
                    let chosen = valid_vars[rng.random_range(0..valid_vars.len())];
                    nodes.push(Node::Variable(chosen.1, target_type));
                } else {
                    let idx = constants.len() as u16;
                    constants.push(c);
                    nodes.push(Node::Constant(idx, target_type));
                }
            }
            (true, None) => {
                let chosen = valid_vars[rng.random_range(0..valid_vars.len())];
                nodes.push(Node::Variable(chosen.1, target_type));
            }
            (false, Some(c)) => {
                let idx = constants.len() as u16;
                constants.push(c);
                nodes.push(Node::Constant(idx, target_type));
            }
            (false, None) => {
                add_operator_node(
                    nodes,
                    constants,
                    target_type,
                    current_depth,
                    max_depth,
                    rng,
                    variables,
                    allowed_ops,
                    parent_op,
                );
            }
        }
    } else {
        add_operator_node(
            nodes,
            constants,
            target_type,
            current_depth,
            max_depth,
            rng,
            variables,
            allowed_ops,
            parent_op,
        );
    }
}

fn add_operator_node(
    nodes: &mut Vec<Node>,
    constants: &mut Vec<Scalar>,
    target_type: ValueType,
    current_depth: usize,
    max_depth: usize,
    rng: &mut impl RngExt,
    variables: &[(ValueType, u8)],
    allowed_ops: &[Instruction],
    parent_op: Option<Instruction>,
) {
    if let Some(chosen_op) = random_operator(target_type, allowed_ops, parent_op, rng) {
        let expected_children_types = chosen_op.expected_types();
        for &child_type in expected_children_types {
            build_ast_recursive(
                nodes,
                constants,
                child_type,
                current_depth + 1,
                max_depth,
                rng,
                variables,
                allowed_ops,
                Some(chosen_op),
            );
        }
        nodes.push(Node::Operator(chosen_op));
    } else {
        let valid_vars: Vec<_> = variables.iter().filter(|v| v.0 == target_type).collect();
        let maybe_const = random_constant(target_type, rng);

        match (!valid_vars.is_empty(), maybe_const) {
            (true, Some(c)) => {
                if rng.random::<bool>() {
                    let chosen = valid_vars[rng.random_range(0..valid_vars.len())];
                    nodes.push(Node::Variable(chosen.1, target_type));
                } else {
                    let idx = constants.len() as u16;
                    constants.push(c);
                    nodes.push(Node::Constant(idx, target_type));
                }
            }
            (true, None) => {
                let chosen = valid_vars[rng.random_range(0..valid_vars.len())];
                nodes.push(Node::Variable(chosen.1, target_type));
            }
            (false, Some(c)) => {
                let idx = constants.len() as u16;
                constants.push(c);
                nodes.push(Node::Constant(idx, target_type));
            }
            (false, None) => {
                panic!(
                    "Error: No operator, variable or constant for {:?} type!",
                    target_type
                );
            }
        }
    }
}

pub fn random_operator(
    target_type: ValueType,
    allowed_ops: &[Instruction],
    parent_op: Option<Instruction>,
    rng: &mut impl RngExt,
) -> Option<Instruction> {
    let is_valid = |op: &Instruction| -> bool {
        if op.return_type() != target_type {
            return false;
        }
        if let Some(p) = parent_op
            && p.is_forbidden_child(op)
        {
            return false;
        }
        true
    };

    let valid_count = allowed_ops.iter().copied().filter(is_valid).count();

    if valid_count == 0 {
        return None;
    }
    let chosen_idx = rng.random_range(0..valid_count);
    allowed_ops.iter().copied().filter(is_valid).nth(chosen_idx)
}

pub fn random_constant(target_type: ValueType, rng: &mut impl RngExt) -> Option<Scalar> {
    match target_type {
        ValueType::Float => Some(Scalar::Float(rng.random_range(-5.0..5.0))),
        ValueType::Vec2 => Some(Scalar::Vec2([
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
        ])),
        ValueType::Vec3 => Some(Scalar::Vec3([
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
        ])),
        ValueType::Mat2 => Some(Scalar::Mat2([
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
        ])),
        ValueType::Mat3 => Some(Scalar::Mat3([
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
        ])),
        ValueType::Bool => Some(Scalar::Bool(rng.random::<bool>())),
        ValueType::Int => Some(Scalar::Int(rng.random_range(-10..10))),
    }
}
