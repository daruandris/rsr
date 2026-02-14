pub enum Op {
    Add, Sub, Mul, Div, Sin, Cos, Exp
}

pub enum Node {
    Operator(Op),
    Variable(usize),
    Constant(f64),
}