use crate::ast::node::{Node, Op};

struct SubTree {
    nodes: Vec<Node>,
    is_const: bool,
    val: f64,
}

pub fn simplify_ast(original_nodes: &[Node]) -> Vec<Node> {
    if original_nodes.is_empty() { return Vec::new(); }

    let mut stack: Vec<SubTree> = Vec::with_capacity(32);

    for node in original_nodes {
        match node {
            Node::Constant(c) => {
                stack.push(SubTree { nodes: vec![Node::Constant(*c)], is_const: true, val: *c });
            },
            Node::Variable(v) => {
                stack.push(SubTree { nodes: vec![Node::Variable(*v)], is_const: false, val: 0.0 });
            },
            Node::Operator(op) => {
                let arity = node.arity();

                if arity == 1 {
                    if let Some(a) = stack.pop() {
                        // Konstans egyszerűsítés
                        if a.is_const {
                            let res = match op {
                                Op::Sin => a.val.sin(),
                                Op::Cos => a.val.cos(),
                                Op::Exp => a.val.exp(),
                                Op::Sqr => a.val * a.val,
                                _ => unreachable!(),
                            };
                            if res.is_finite() {
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
                        // két konstans
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
                        // azonos ágak
                        if a.nodes == b.nodes {
                            if let Op::Sub = op {
                                stack.push(SubTree { nodes: vec![Node::Constant(0.0)], is_const: true, val: 0.0 });
                                continue;
                            }
                            if let Op::Div = op {
                                stack.push(SubTree { nodes: vec![Node::Constant(1.0)], is_const: true, val: 1.0 });
                                continue;
                            }
                            if let Op::Mul = op {
                                let mut new_nodes = a.nodes;
                                new_nodes.push(Node::Operator(Op::Sqr)); 
                                stack.push(SubTree { nodes: new_nodes, is_const: false, val: 0.0 });
                                continue;
                            }
                        }
                        
                        // szorzás 0-val
                        if let Op::Mul = op {
                            if (a.is_const && a.val.abs() < 1e-9) || (b.is_const && b.val.abs() < 1e-9) {
                                stack.push(SubTree { nodes: vec![Node::Constant(0.0)], is_const: true, val: 0.0 });
                                continue;
                            }
                        }

                        // A - (-B) = A + B
                        if let Op::Sub = op {
                            if b.nodes.len() >= 2 {
                                if let Some(Node::Operator(Op::Sub)) = b.nodes.last() {
                                    if let Some(Node::Constant(c)) = b.nodes.get(b.nodes.len() - 2) {
                                        if c.abs() < 1e-9 {
                                            let x_nodes = &b.nodes[0..b.nodes.len()-2];
                                            if a.is_const && a.val.abs() < 1e-9 {
                                                stack.push(SubTree { 
                                                    nodes: x_nodes.to_vec(), 
                                                    is_const: false,
                                                    val: 0.0 
                                                });
                                                continue;
                                            }
                                            let mut new_nodes = Vec::with_capacity(a.nodes.len() + x_nodes.len() + 1);
                                            new_nodes.extend(a.nodes);
                                            new_nodes.extend_from_slice(x_nodes);
                                            new_nodes.push(Node::Operator(Op::Add));
                                            
                                            stack.push(SubTree { nodes: new_nodes, is_const: false, val: 0.0 });
                                            continue;
                                        }
                                    }
                                }
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
        final_tree.nodes
    } else {
        Vec::new()
    }
}