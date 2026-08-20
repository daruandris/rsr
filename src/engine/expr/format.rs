use super::node::Node;

pub fn format_ast(nodes: &[Node]) -> String {
    let mut stack: Vec<String> = Vec::with_capacity(32);

    for node in nodes {
        match node {
            Node::Constant(c, _) => stack.push(format!("{}", c)),
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
    stack.pop().unwrap_or_else(|| "Empty expression".to_string())
}