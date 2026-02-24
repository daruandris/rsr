// src/ast/format.rs
use crate::ast::node::Node;
use crate::metrics::dataset::SimdDataset;
use crate::engine::individual::Individual;
use crate::domain::Domain;

pub fn format_ast<D: Domain>(nodes: &[Node<D>]) -> String {
    let mut stack: Vec<String> = Vec::with_capacity(32);

    for node in nodes {
        match node {
            // A D::ScalarValue-ra kikötöttük a traitben, hogy implementálja a Display-t
            Node::Constant(c,_) => stack.push(format!("{:.3}", c)),
            Node::Variable(v, type_id) => stack.push(format!("X{}_{:?}", v, type_id)),
            Node::Operator(op) => {
                let arity = D::operator_arity(op);
                let mut args = Vec::with_capacity(arity);
                
                // Mivel a stack-ről fordított sorrendben jönnek le a dolgok (LIFO),
                // először kivesszük őket...
                for _ in 0..arity {
                    if let Some(arg) = stack.pop() {
                        args.push(arg);
                    } else {
                        args.push("?".to_string()); // Biztonsági tartalék érvénytelen fákra
                    }
                }
                // ...majd megfordítjuk, hogy a bal argumentum legyen az args[0]
                args.reverse();
                
                // Rábízzuk a Domain-re a string összerakását
                stack.push(D::format_operator(op, &args));
            }
        }
    }
    
    stack.pop().unwrap_or_else(|| "Empty expression".to_string())
}

pub fn format_real_equation<D: Domain>(ind: &Individual<D>, dataset: &SimdDataset) -> String {
    let core_expr = format_ast(&ind.nodes);
    
    format!(
        "y = {:.4} * ( {} ) + {:.4}\n\t[Input normalization: X_norm = (X - mean) / std]",
        dataset.target_std_dev,
        core_expr,
        dataset.target_mean
    )
}