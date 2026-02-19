use crate::domain::Domain;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Add, Sub, Mul, Div, Sin, Cos, Exp, Sqr
}

impl Op {
    pub fn weight(&self) -> usize {
        match self {
            Op::Add | Op::Sub | Op::Mul => 1,
            Op::Sqr => 2,
            Op::Div => 3,
            Op::Cos | Op::Sin => 4,
            Op::Exp => 3,
        }
    }

    pub fn forbidden_children(&self) -> &'static [Op] {
        match self {
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Sqr => &[],
            Op::Sin | Op::Cos => &[Op::Sin, Op::Cos, Op::Exp],
            Op::Exp => &[Op::Exp, Op::Sin, Op::Cos, Op::Sqr],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Node<D: Domain> {
    Operator(D::Operator),
    Variable(u8),
    Constant(D::ScalarValue),
}

impl<D: Domain> Node<D> {
    pub fn arity(&self) -> usize {
        match self {
            Node::Constant(_) | Node::Variable(_) => 0,
            Node::Operator(op) => D::operator_arity(op),
        }
    }

    pub fn weight(&self) -> usize {
        match self {
            Node::Constant(_) | Node::Variable(_) => 1,
            Node::Operator(op) => D::operator_weight(op),
        }
    }
}