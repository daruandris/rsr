use crate::ast::node::Node;
use crate::domain::Domain;
use rand::RngExt;

pub fn generate_random_ast<D: Domain>(
    target_type: D::TypeId, 
    max_depth: usize, 
    rng: &mut impl RngExt, 
    num_features: u8,
    allowed_ops: &[D::Operator]
) -> Vec<Node<D>> {
    let cap = 1 << (max_depth.min(6));
    let mut nodes = Vec::with_capacity(cap);
    build_ast_recursive::<D>(&mut nodes, target_type, 0, max_depth, rng, num_features, allowed_ops);
    nodes
}

fn build_ast_recursive<D: Domain>(
    nodes: &mut Vec<Node<D>>,
    target_type: D::TypeId,
    current_depth: usize, 
    max_depth: usize, 
    rng: &mut impl RngExt, 
    num_features: u8,
    allowed_ops: &[D::Operator],
) {
    // Terminál feltétel: elértük a max mélységet, vagy véletlenszerűen leállunk
    let is_terminal = current_depth >= max_depth || (current_depth > 0 && rng.random::<f32>() < 0.2);

    if is_terminal {
        // Megvizsgáljuk, milyen terminálok érhetők el a kért típushoz
        let is_var_valid = D::variable_type() == target_type; 
        let maybe_const = D::random_constant(target_type, rng);

        match (is_var_valid, maybe_const) {
            (true, Some(c)) => {
                // Változó és Konstans is generálható ebből a típusból -> Sorsolunk
                if rng.random::<bool>() {
                    // ÚJ: Átadjuk a target_type-ot is a Node-nak!
                    nodes.push(Node::Variable(rng.random_range(0..num_features), target_type));
                } else {
                    nodes.push(Node::Constant(c, target_type));
                }
            },
            (true, None) => {
                // Csak változó van (pl. ha a konstans generátor nem támogatja ezt a típust)
                nodes.push(Node::Variable(rng.random_range(0..num_features), target_type));
            },
            (false, Some(c)) => {
                // Csak konstans van (pl. Int típus, de a bemeneti datasetben csak Float feature-ök vannak)
                nodes.push(Node::Constant(c, target_type));
            },
            (false, None) => {
                // Ha se változó, se konstans nincs a kért típusból, kénytelenek vagyunk 
                // operátort behívni, hogy feloldja ezt a típust (kivétel a max mélység alól)
                add_operator_node(nodes, target_type, current_depth, max_depth, rng, num_features, allowed_ops);
            }
        }
    } else {
        add_operator_node(nodes, target_type, current_depth, max_depth, rng, num_features, allowed_ops);
    }
}

fn add_operator_node<D: Domain>(
    nodes: &mut Vec<Node<D>>, 
    target_type: D::TypeId, 
    current_depth: usize, 
    max_depth: usize, 
    rng: &mut impl RngExt, 
    num_features: u8,
    allowed_ops: &[D::Operator],
) {
    // Maszkolás varázslat: Csak olyan operátort kapunk, aminek a kimenete 'target_type'
    if let Some(chosen_op) = D::random_operator(target_type, allowed_ops, rng) {
        let expected_children_types = D::expected_types(&chosen_op);
        
        // STGP: Gyerekek legenerálása a specifikált típusok alapján [cite: 362, 363]
        for child_type in expected_children_types {
            build_ast_recursive::<D>(nodes, child_type, current_depth + 1, max_depth, rng, num_features, allowed_ops);
        }
        
        nodes.push(Node::Operator(chosen_op));
    } else {
        // Fallback: Ha a nyelvtan hibás (nincs se terminál, se operátor a kért típushoz)
        panic!("Nyelvtani hiba: Nincs érvényes operátor vagy terminál a {:?} típushoz!", target_type);
    }
}