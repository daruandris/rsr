use crate::ast::node::{Node, Op};

pub fn format_ast(nodes: &[Node]) -> String {
    let mut stack: Vec<String> = Vec::with_capacity(32);

    for node in nodes {
        match node {
            Node::Constant(c) => stack.push(format!("{:.3}", c)),
            Node::Variable(v) => stack.push(format!("X{}", v)),
            Node::Operator(op) => match op {
                Op::Add | Op::Sub | Op::Mul | Op::Div => {
                    if let (Some(b), Some(a)) = (stack.pop(), stack.pop()) {
                        let sym = match op {
                            Op::Add => "+",
                            Op::Sub => "-",
                            Op::Mul => "*",
                            Op::Div => "/",
                            _ => unreachable!(),
                        };
                        stack.push(format!("({} {} {})", a, sym, b));
                    }
                },
                Op::Sin | Op::Cos | Op::Exp => {
                    if let Some(a) = stack.pop() {
                        let sym = match op {
                            Op::Sin => "sin",
                            Op::Cos => "cos",
                            Op::Exp => "exp",
                            Op::Sqr => "sqr",
                            _ => unreachable!(),
                        };
                        stack.push(format!("{}({})", sym, a));
                    }
                },
                Op::Sqr => {
                    if let Some(a) = stack.pop() {
                        stack.push(format!("({})^2", a));
                    }
                }
            }
        }
    }
    
    stack.pop().unwrap_or_else(|| "Üres_kifejezés".to_string())
}