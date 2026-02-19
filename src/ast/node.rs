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
pub enum Node {
    Operator(Op),
    Variable(u8),
    Constant(f32),
}

impl Node {
    pub fn arity(&self) -> usize {
        match self {
            Node::Constant(_) | Node::Variable(_) => 0,
            Node::Operator(op) => match op {
                Op::Sin | Op::Cos | Op::Exp | Op::Sqr => 1,
                Op::Add | Op::Sub | Op::Mul | Op::Div => 2,
            },
        }
    }

    pub fn weight(&self) -> usize {
        match self {
            Node::Constant(_) | Node::Variable(_) => 1,
            Node::Operator(op) => op.weight(),
        }
        
    }
}