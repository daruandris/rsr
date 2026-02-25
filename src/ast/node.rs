use crate::domain::Domain;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Node<D: Domain> {
    Operator(D::Operator),
    Variable(u8, D::TypeId),
    Constant(D::ScalarValue, D::TypeId),
}

impl<D: Domain> Node<D> {
    pub fn arity(&self) -> usize {
        match self {
            Node::Constant(_, _) | Node::Variable(_, _) => 0,
            Node::Operator(op) => D::operator_arity(op),
        }
    }

    pub fn weight(&self) -> usize {
        match self {
            Node::Variable(_, _) => 1, 
            Node::Constant(_, type_id) => D::type_weight(type_id),
            Node::Operator(op) => D::operator_weight(op), 
        }
    }

    pub fn get_type(&self) -> D::TypeId {
        match self {
            Node::Operator(op) => D::return_type(op),
            Node::Variable(_, type_id) => *type_id,
            Node::Constant(_, type_id) => *type_id,
        }
    }
}