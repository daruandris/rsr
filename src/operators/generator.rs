use crate::ast::node::Node;
use crate::domain::Domain;
use rand::RngExt;

pub fn generate_random_ast<D: Domain>(
    target_type: D::TypeId, 
    max_depth: usize, 
    rng: &mut impl RngExt, 
    variables: &[(D::TypeId, u8)],
    allowed_ops: &[D::Operator] 
) -> Vec<Node<D>> {
    let cap = 1 << (max_depth.min(6));
    let mut nodes = Vec::with_capacity(cap);
    // Induláskor nincs szülő (None)
    build_ast_recursive::<D>(&mut nodes, target_type, 0, max_depth, rng, variables, allowed_ops, None);
    nodes
}

fn build_ast_recursive<D: Domain>(
    nodes: &mut Vec<Node<D>>,
    target_type: D::TypeId,
    current_depth: usize, 
    max_depth: usize, 
    rng: &mut impl RngExt, 
    variables: &[(D::TypeId, u8)],
    allowed_ops: &[D::Operator],
    parent_op: Option<D::Operator>
) {
    let is_terminal = current_depth >= max_depth || (current_depth > 0 && rng.random::<f32>() < 0.2);

    if is_terminal {
        let valid_vars: Vec<_> = variables.iter().filter(|v| v.0 == target_type).collect();
        let is_var_valid = !valid_vars.is_empty();
        let maybe_const = D::random_constant(target_type, rng);

        match (is_var_valid, maybe_const) {
            (true, Some(c)) => {
                if rng.random::<bool>() {
                    let chosen = valid_vars[rng.random_range(0..valid_vars.len())];
                    nodes.push(Node::Variable(chosen.1, target_type));
                } else {
                    nodes.push(Node::Constant(c, target_type));
                }
            },
            (true, None) => {
                let chosen = valid_vars[rng.random_range(0..valid_vars.len())];
                nodes.push(Node::Variable(chosen.1, target_type));
            },
            (false, Some(c)) => nodes.push(Node::Constant(c, target_type)),
            (false, None) => {
                add_operator_node(nodes, target_type, current_depth, max_depth, rng, variables, allowed_ops, parent_op);
            }
        }
    } else {
        add_operator_node(nodes, target_type, current_depth, max_depth, rng, variables, allowed_ops, parent_op);
    }
}

fn add_operator_node<D: Domain>(
    nodes: &mut Vec<Node<D>>, 
    target_type: D::TypeId, 
    current_depth: usize, 
    max_depth: usize, 
    rng: &mut impl RngExt, 
    variables: &[(D::TypeId, u8)],
    allowed_ops: &[D::Operator],
    parent_op: Option<D::Operator>
) {
    // Átadjuk a szülőt a sorsolónak!
    if let Some(chosen_op) = D::random_operator(target_type, allowed_ops, parent_op, rng) {
        let expected_children_types = D::expected_types(&chosen_op);
        
        for &child_type in expected_children_types {
            // A rekurzióban a most kisorsolt operátor (chosen_op) lesz az új szülő!
            build_ast_recursive::<D>(nodes, child_type, current_depth + 1, max_depth, rng, variables, allowed_ops, Some(chosen_op));
        }
        
        nodes.push(Node::Operator(chosen_op));
    } else {
        panic!("Nyelvtani hiba: Nincs érvényes operátor a {:?} típushoz!", target_type);
    }
}