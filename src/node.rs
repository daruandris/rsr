pub enum Op {
    Add, Sub, Mul, Div, Sin, Cos, Exp
}

pub enum Node {
    Operator(Op),
    Variable(usize),
    Constant(f64),
}

impl Node {
    pub fn arity(&self) -> usize{
        match self {
            Node::Constant(_) | Node::Variable(_) => 0,
            Node::Operator(op) => match op {
                Op::Sin | Op::Cos | Op::Exp => 1,
                Op::Add | Op::Sub | Op::Mul | Op::Div => 2,
            },
        }
    }
}