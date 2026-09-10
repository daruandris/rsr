use crate::Instruction;
use crate::engine::eval::scalar::Scalar;
use crate::engine::eval::types::ValueType;
use crate::engine::expr::node::Node;
use rand::RngExt;

#[derive(Clone, Copy)]
enum Choice {
    Var(u8),
    Const,
    Op(Instruction),
}

pub fn generate_random_ast(
    target_type: ValueType,
    max_depth: usize,
    rng: &mut impl RngExt,
    variables: &[(ValueType, u8)],
    allowed_ops: &[Instruction],
    disabled_constants: &[ValueType],
) -> (Vec<Node>, Vec<Scalar>) {
    let cap = 1 << (max_depth.min(6));
    let mut nodes = Vec::with_capacity(cap);
    let mut constants = Vec::new();
    let is_solid = allowed_ops.iter().any(|op| matches!(op, Instruction::Solid(_)));

    for _ in 0..10 {
        nodes.clear();
        constants.clear();
        if build_ast_recursive(
            &mut nodes,
            &mut constants,
            target_type,
            0,
            max_depth,
            rng,
            variables,
            allowed_ops,
            None,
            disabled_constants,
            is_solid
        ) {
            return (nodes, constants);
        }
    }

    nodes.clear();
    constants.clear();
    let valid_vars: Vec<_> = variables.iter().filter(|v| v.0 == target_type).collect();
    if !valid_vars.is_empty() {
        nodes.push(Node::Variable(valid_vars[0].1, target_type));
        return (nodes, constants);
    }
    if !disabled_constants.contains(&target_type) {
        if let Some(c) = random_constant(target_type, disabled_constants, is_solid, rng) {
            constants.push(c);
            nodes.push(Node::Constant(0, target_type));
            return (nodes, constants);
        }
    }
    panic!("Nem sikerült AST-t generálni a {:?} típushoz. Nincs érvényes operátor, változó, és a konstansok is tiltva vannak!", target_type);
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
    disabled_constants: &[ValueType],
    is_solid: bool,
) -> bool {
    let prefer_terminal = current_depth >= max_depth || (current_depth > 0 && rng.random::<f32>() < 0.2);

    let mut terminals = Vec::new();
    if !disabled_constants.contains(&target_type) {
        terminals.push(Choice::Const);
    }
    for v in variables.iter().filter(|v| v.0 == target_type) {
        terminals.push(Choice::Var(v.1));
    }

    let mut ops = Vec::new();
    if current_depth < max_depth {
        for op in allowed_ops {
            if op.return_type() == target_type {
                let forbidden = parent_op.is_some_and(|p| p.is_forbidden_child(op));
                if !forbidden {
                    ops.push(Choice::Op(*op));
                }
            }
        }
    }

    let mut shuffle = |vec: &mut Vec<Choice>| {
        if vec.is_empty() { return; }
        for i in (1..vec.len()).rev() {
            let j = rng.random_range(0..=i);
            vec.swap(i, j);
        }
    };

    shuffle(&mut terminals);
    shuffle(&mut ops);

    let mut choices = Vec::with_capacity(terminals.len() + ops.len());
    if prefer_terminal && !terminals.is_empty() {
        choices.extend(terminals);
        choices.extend(ops);
    } else {
        choices.extend(ops);
        choices.extend(terminals);
    }

    let saved_nodes_len = nodes.len();
    let saved_consts_len = constants.len();

    for choice in choices {
        match choice {
            Choice::Var(idx) => {
                nodes.push(Node::Variable(idx, target_type));
                return true;
            }
            Choice::Const => {
                if let Some(c) = random_constant(target_type, disabled_constants, is_solid, rng) {
                    let idx = constants.len() as u16;
                    constants.push(c);
                    nodes.push(Node::Constant(idx, target_type));
                    return true;
                }
            }
            Choice::Op(op) => {
                let expected_types = op.expected_types();
                let mut success = true;

                for &child_type in expected_types {
                    if !build_ast_recursive(
                        nodes, constants, child_type, current_depth + 1, max_depth,
                        rng, variables, allowed_ops, Some(op), disabled_constants, is_solid
                    ) {
                        success = false;
                        break;
                    }
                }

                if success {
                    nodes.push(Node::Operator(op));
                    return true;
                } else {
                    nodes.truncate(saved_nodes_len);
                    constants.truncate(saved_consts_len);
                }
            }
        }
    }
    false
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

pub fn random_constant(target_type: ValueType, disabled_constants: &[ValueType], is_solid: bool, rng: &mut impl RngExt) -> Option<Scalar> {
    if disabled_constants.contains(&target_type) {
        return None;
    }
    match target_type {
        ValueType::Float => Some(Scalar::Float(rng.random_range(-5.0..5.0))),
        ValueType::Vec2 => Some(Scalar::Vec2([
            rng.random_range(-5.0..5.0),
            rng.random_range(-5.0..5.0),
        ])),
        // only for now, for solid tests
        ValueType::Vec3 => {
            if is_solid {
                let r = rng.random::<f32>();
                if r < 0.30 {
                    Some(Scalar::Vec3([1.0, 0.0, 0.0])) // X tengely (rostirány)
                } else if r < 0.60 {
                    Some(Scalar::Vec3([0.0, 1.0, 0.0])) // Y tengely
                } else if r < 0.80 {
                    Some(Scalar::Vec3([0.0, 0.0, 1.0])) // Z tengely
                } else {
                    Some(Scalar::Vec3([
                        rng.random_range(-2.0..2.0),
                        rng.random_range(-2.0..2.0),
                        rng.random_range(-2.0..2.0),
                    ]))
                }
            } else {
                Some(Scalar::Vec3([
                    rng.random_range(-5.0..5.0),
                    rng.random_range(-5.0..5.0),
                    rng.random_range(-5.0..5.0),
                ]))
            }
        },
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
