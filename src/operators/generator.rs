use crate::ast::node::Node;
use crate::domain::Domain;
use rand::RngExt;

pub fn generate_random_ast<D: Domain>(target_type: D::TypeId, max_depth: usize, rng: &mut impl RngExt, num_features: u8) -> Vec<Node<D>> {
    let cap = 1 << (max_depth.min(6)); 
    let mut nodes = Vec::with_capacity(cap);
    build_ast_recursive::<D>(&mut nodes, target_type, 0, max_depth, rng, num_features);
    nodes
}

fn build_ast_recursive<D: Domain>(
    nodes: &mut Vec<Node<D>>,
    target_type: D::TypeId,
    current_depth: usize, 
    max_depth: usize, 
    rng: &mut impl RngExt, 
    num_features: u8, 
) {
    let is_terminal = current_depth >= max_depth || (current_depth > 0 && rng.random::<f32>() < 0.2);
    if is_terminal {
        let is_var_valid = D::variable_type() == target_type;
        let is_const_valid = D::constant_type() == target_type;

        if is_var_valid && is_const_valid {
            if rng.random::<bool>() {
                nodes.push(Node::Variable(rng.random_range(0..num_features)));
            } else {
                if let Some(c) = D::random_constant(target_type, rng) {
                    nodes.push(Node::Constant(c));
                } else {
                    nodes.push(Node::Variable(rng.random_range(0..num_features)));
                }
            }
        } else if is_var_valid {
            nodes.push(Node::Variable(rng.random_range(0..num_features)));
        } else if is_const_valid {
            if let Some(c) = D::random_constant(target_type, rng) {
                nodes.push(Node::Constant(c));
            } else {
                panic!("Nem sikerült konstanst generálni a kért típushoz!");
            }
        } else {
            add_operator_node(nodes, target_type, current_depth, max_depth, rng, num_features);
        }
    } else {
        add_operator_node(nodes, target_type, current_depth, max_depth, rng, num_features);
    }
}

fn add_operator_node<D: Domain>(
    nodes: &mut Vec<Node<D>>, 
    target_type: D::TypeId, 
    current_depth: usize, 
    max_depth: usize, 
    rng: &mut impl RngExt, 
    num_features: u8,
) {
    if let Some(chosen_op) = D::random_operator(target_type, rng) {
        let expected_children_types = D::expected_types(&chosen_op);
        
        // STGP Varázslat: Minden gyereknek megmondjuk, milyen típust KELL visszaadnia!
        for child_type in expected_children_types {
            build_ast_recursive::<D>(nodes, child_type, current_depth + 1, max_depth, rng, num_features);
        }
        nodes.push(Node::Operator(chosen_op));
    } else {
        // Fallback, ha nincs olyan operátor, ami ezt a típust adná (hibás nyelvtani definíció)
        panic!("A nyelvtanban nincs olyan operátor, ami a kért típust adná vissza!");
    }
}