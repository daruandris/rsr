use crate::Instruction;
use crate::engine::eval::scalar::Scalar;
use crate::engine::eval::types::ValueType;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Node {
    Operator(Instruction),
    Variable(u8, ValueType),
    Constant(Scalar, ValueType),
}

impl Node {
    #[inline]
    pub fn arity(&self) -> usize {
        match self {
            Node::Constant(_, _) | Node::Variable(_, _) => 0,
            Node::Operator(op) => op.arity(),
        }
    }

    #[inline]
    pub fn weight(&self) -> usize {
        match self {
            Node::Variable(_, _) => 1,
            Node::Constant(_, type_id) => match type_id {
                ValueType::Float | ValueType::Int | ValueType::Bool => 1,
                ValueType::Vec2 => 2,
                ValueType::Vec3 => 3,
                ValueType::Mat2 => 4,
                ValueType::Mat3 => 9,
            },
            Node::Operator(op) => op.weight(),
        }
    }

    #[inline]
    pub fn get_type(&self) -> ValueType {
        match self {
            Node::Operator(op) => op.return_type(),
            Node::Variable(_, type_id) => *type_id,
            Node::Constant(_, type_id) => *type_id,
        }
    }
}
