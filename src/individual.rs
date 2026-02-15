use crate::node::{Node, Op};

#[derive(Clone)]
pub struct Individual {
    pub nodes: Vec<Node>,
    pub fitness: f64,
}

impl Individual {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self {
            nodes,
            fitness: f64::MAX,
        }
    }

    pub fn evaluate(&self, features: &[f64]) -> f64 {
        let mut stack = Vec::with_capacity(32);

        for node in &self.nodes{
            match node {
                Node::Constant(c) => stack.push(*c),
                Node::Variable(v) => stack.push(features[*v]),
                Node::Operator(op) => match op {
                    Op::Add | Op::Sub | Op::Mul | Op::Div => {
                        if let (Some(a), Some(b)) = (stack.pop(), stack.pop()){
                            let result = match op {
                                Op::Add => a + b,
                                Op::Sub => a - b,
                                Op::Mul => a * b,
                                Op::Div => {
                                    if b.abs() < 1e-9 { 1.0} else { a / b }
                                },
                                _ => unreachable!()
                            };
                            stack.push(result);
                        }
                    },

                    Op::Sin | Op::Cos | Op::Exp => {
                        if let Some(a) = stack.pop() {
                            let result = match op {
                                Op::Sin => a.sin(),
                                Op::Cos => a.cos(),
                                Op::Exp => {
                                    let val = a.exp();
                                    if val.is_finite() { val } else { f64::MAX }
                                },
                                _ => unreachable!(),
                            };
                            stack.push(result);
                        }
                    }
                }
            }
        }
        stack.pop().unwrap_or(f64::NAN)
    }

    pub fn get_subtree_bounds(&self, root_idx: usize) -> (usize, usize) {
        let mut needed = 1;
        let mut current_idx = root_idx;
        loop {
            needed += self.nodes[current_idx].arity() - 1;
            if needed == 0{
                return (current_idx, root_idx);
            }

            if current_idx == 0 {
                break ;
            }
            current_idx -= 1;
        }

        (0, root_idx)
    }

}