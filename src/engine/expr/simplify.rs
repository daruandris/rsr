use super::node::Node;
use crate::Instruction::Basic;
use crate::domains::basic::BasicOpCode::{ExpF, LnF, SqrF, SqrtF};
use crate::engine::domain::SimplifyAction;
use crate::engine::eval::scalar::Scalar;

#[derive(Clone, Copy, Debug)]
struct ExprInfo {
    start_idx: usize,
    const_val: Option<Scalar>,
}

pub fn simplify_ast(nodes: &[Node]) -> Vec<Node> {
    if nodes.is_empty() {
        return vec![];
    }

    let mut output = Vec::with_capacity(nodes.len());
    let mut stack: Vec<ExprInfo> = Vec::with_capacity(32);

    for &node in nodes {
        match node {
            Node::Constant(val, _type_id) => {
                let start_idx = output.len();
                output.push(node);
                stack.push(ExprInfo {
                    start_idx,
                    const_val: Some(val),
                });
            }
            Node::Variable(_, _) => {
                let start_idx = output.len();
                output.push(node);
                stack.push(ExprInfo {
                    start_idx,
                    const_val: None,
                });
            }
            Node::Operator(op) => {
                let arity = op.arity();
                if stack.len() < arity {
                    let start_idx = output.len();
                    output.push(node);
                    stack.push(ExprInfo {
                        start_idx,
                        const_val: None,
                    });
                    continue;
                }

                let mut args = Vec::with_capacity(arity);
                for _ in 0..arity {
                    args.push(stack.pop().unwrap());
                }
                args.reverse();

                let mut args_equal = false;
                if arity == 2 {
                    let a = &args[0];
                    let b = &args[1];
                    if a.start_idx < b.start_idx && b.start_idx <= output.len() {
                        args_equal = output[a.start_idx..b.start_idx] == output[b.start_idx..];
                    }
                }

                let const_vals: Vec<Option<Scalar>> = args.iter().map(|a| a.const_val).collect();
                let action = crate::SymbolicEngine::try_simplify(op, &const_vals, args_equal);

                match action {
                    SimplifyAction::ReplaceWithConstant(val) => {
                        output.truncate(args[0].start_idx);
                        let new_start = output.len();
                        output.push(Node::Constant(val, op.return_type()));
                        stack.push(ExprInfo {
                            start_idx: new_start,
                            const_val: Some(val),
                        });
                    }
                    SimplifyAction::KeepArg(idx) => {
                        let target_arg = &args[idx];
                        let start_of_args = args[0].start_idx;

                        let target_end = if idx == arity - 1 {
                            output.len()
                        } else {
                            args[idx + 1].start_idx
                        };

                        let target_len = target_end - target_arg.start_idx;

                        if target_arg.start_idx > start_of_args {
                            output.copy_within(target_arg.start_idx..target_end, start_of_args);
                        }
                        output.truncate(start_of_args + target_len);

                        let mut kept_info = *target_arg;
                        kept_info.start_idx = start_of_args;
                        stack.push(kept_info);
                    }
                    SimplifyAction::None => {
                        // Speciális inverz függvények kiejtése (pl. ln(exp(x)) == x)
                        if arity == 1
                            && output.len() > args[0].start_idx
                            && let Node::Operator(child_op) = output[output.len() - 1]
                        {
                            match (op, child_op) {
                                (Basic(LnF), Basic(ExpF))
                                | (Basic(ExpF), Basic(LnF))
                                | (Basic(SqrtF), Basic(SqrF))
                                | (Basic(SqrF), Basic(SqrtF)) => {
                                    output.pop(); // Levesszük a belső operátort
                                    stack.push(ExprInfo {
                                        start_idx: args[0].start_idx,
                                        const_val: None,
                                    });
                                    continue;
                                }
                                _ => {}
                            }
                        }
                        output.push(node);
                        stack.push(ExprInfo {
                            start_idx: args[0].start_idx,
                            const_val: None,
                        });
                    }
                }
            }
        }
    }
    output
}
