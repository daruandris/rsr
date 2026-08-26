use super::node::Node;
use crate::engine::eval::scalar::Scalar;

pub fn format_ast(nodes: &[Node], constants: &[Scalar]) -> String {
    let mut stack: Vec<String> = Vec::with_capacity(32);

    for node in nodes {
        match node {
            Node::Constant(c, _) => {
                let idx = *c as usize;
                if idx < constants.len() {
                    stack.push(format!("{}", constants[idx]));
                } else {
                    stack.push(format!("C[{}]", idx));
                }
            }
            Node::Variable(v, type_id) => stack.push(format!("X{}_{:?}", v, type_id)),
            Node::Operator(op) => {
                let arity = op.arity();
                let mut args = Vec::with_capacity(arity);

                for _ in 0..arity {
                    if let Some(arg) = stack.pop() {
                        args.push(arg);
                    } else {
                        args.push("?".to_string());
                    }
                }
                args.reverse();
                stack.push(op.format_op(&args));
            }
        }
    }
    stack
        .pop()
        .unwrap_or_else(|| "Empty expression".to_string())
}