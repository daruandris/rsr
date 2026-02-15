use std::fmt;

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
                        if let (Some(b), Some(a)) = (stack.pop(), stack.pop()){
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
            needed = needed + self.nodes[current_idx].arity() - 1;
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

    pub fn get_constants(&self) -> Vec<f64> {
        self.nodes.iter().filter_map(|node| {
            if let Node::Constant(c) = node {Some(*c)} else { None }
        }).collect()
    }

    pub fn set_constants(&mut self, new_constants: &[f64]) {
        let mut const_idx = 0;
        for node in self.nodes.iter_mut() {
            if let Node::Constant(c) = node {
                if const_idx < new_constants.len() {
                    *c = new_constants[const_idx];
                    const_idx += 1;
                }
            }
        }
    }

    pub fn calculate_mse(&self, data_x: &[Vec<f64>], data_y: &[f64]) -> f64 {
        let mut sum_error = 0.0;
        for (i, row) in data_x.iter().enumerate() {
            let pred = self.evaluate(row);
            let diff = pred - data_y[i];
            sum_error += diff * diff;
        }
        sum_error / (data_x.len() as f64)
    }

    pub fn optimize_constants(
        &mut self, 
        data_x: &[Vec<f64>], 
        data_y: &[f64],
        iterations: usize,
        lr: f64,
        epsilon: f64)
        {
        let mut consts = self.get_constants();
        if consts.is_empty() { return; }

        for _ in 0..iterations {
            let current_mse = self.calculate_mse(data_x, data_y);
            let mut gradients = vec![0.0; consts.len()];
            for i in 0..consts.len() {
                let original_val = consts[i];
                consts[i] = original_val + epsilon;
                self.set_constants(&consts);
                let plus_mse = self.calculate_mse(data_x, data_y);
                gradients[i] = (plus_mse - current_mse) / epsilon;
                
                consts[i] = original_val;
            }
            let mut improved = false;
            for i in 0..consts.len() {
                let grad = gradients[i].clamp(-10.0, 10.0);
                consts[i] -= lr * grad;
                
                if grad.abs() > 1e-6 { improved = true; }
            }

            self.set_constants(&consts);
            if !improved { break; }
        }
    }

    pub fn simplify(&mut self) {
        if self.nodes.is_empty() { return; }

        struct SubTree {
            nodes: Vec<Node>,
            is_const: bool,
            val: f64,
        }

        let mut stack: Vec<SubTree> = Vec::with_capacity(32);

        for node in &self.nodes {
            match node {
                Node::Constant(c) => {
                    stack.push(SubTree { nodes: vec![Node::Constant(*c)], is_const: true, val: *c });
                },
                Node::Variable(v) => {
                    stack.push(SubTree { nodes: vec![Node::Variable(*v)], is_const: false, val: 0.0 });
                },
                Node::Operator(op) => {
                    let arity = node.arity();

                    if arity == 1{
                        if let Some(a) = stack.pop() {
                            if a.is_const {
                                let res = match op {
                                    Op::Sin => a.val.sin(),
                                    Op::Cos => a.val.cos(),
                                    Op::Exp => a.val.exp(),
                                    _ => unreachable!(),
                                };
                                if res.is_finite(){
                                    stack.push(SubTree { nodes: vec![Node::Constant(res)], is_const: true, val: res });
                                    continue;
                                };
                            }

                            let mut new_nodes = a.nodes;
                            new_nodes.push(*node);
                            stack.push(SubTree { nodes: new_nodes, is_const: false, val: 0.0 });
                        }
                    }
                    else if arity == 2 {
                        if let (Some(b), Some(a)) = (stack.pop(), stack.pop()) {
                            if a.is_const && b.is_const {
                                let res = match op {
                                    Op::Add => a.val + b.val,
                                    Op::Sub => a.val - b.val,
                                    Op::Mul => a.val * b.val,
                                    Op::Div => if b.val.abs() < 1e-9 { 1.0 } else { a.val / b.val },
                                    _ => unreachable!(),
                                };
                                if res.is_finite() {
                                    stack.push(SubTree { nodes: vec![Node::Constant(res)], is_const: true, val: res });
                                    continue;
                                }
                            }

                            if a.nodes == b.nodes {
                                if let Op::Sub = op {
                                    // A - A = 0
                                    stack.push(SubTree { nodes: vec![Node::Constant(0.0)], is_const: true, val: 0.0 });
                                    continue;
                                }
                                if let Op::Div = op {
                                    // A / A = 1 (feltételezve, hogy A nem nulla)
                                    stack.push(SubTree { nodes: vec![Node::Constant(1.0)], is_const: true, val: 1.0 });
                                    continue;
                                }
                            }
                            
                            // 2. Szorzás nullával (0 * A = 0, vagy A * 0 = 0)
                            if let Op::Mul = op {
                                if (a.is_const && a.val.abs() < 1e-9) || (b.is_const && b.val.abs() < 1e-9) {
                                    stack.push(SubTree { nodes: vec![Node::Constant(0.0)], is_const: true, val: 0.0 });
                                    continue;
                                }
                            }

                            let mut new_nodes = Vec::with_capacity(a.nodes.len() + b.nodes.len() + 1);
                            new_nodes.extend(a.nodes);
                            new_nodes.extend(b.nodes);
                            new_nodes.push(*node);
                            stack.push(SubTree { nodes: new_nodes, is_const: false, val: 0.0 });
                        }
                    }
                },
            }
        }
        if let Some(final_tree) = stack.pop() {
            self.nodes = final_tree.nodes;
            self.fitness = f64::MAX;
        }
    }

}

impl fmt::Display for Individual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut stack: Vec<String> = Vec::with_capacity(32);

        for node in &self.nodes {
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
                                _ => unreachable!(),
                            };
                            stack.push(format!("{}({})", sym, a));
                        }
                    }
                }
            }
        }
        
        let expr_str = stack.pop().unwrap_or_else(|| "Üres_kifejezés".to_string());
        write!(f, "{}", expr_str)
    }
}