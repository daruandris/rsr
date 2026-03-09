use super::node::Node;
use crate::eval::op::Op;

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
                stack.push(format_op(*op, &args));
            }
        }
    }

    stack
        .pop()
        .unwrap_or_else(|| "Empty expression".to_string())
}

fn format_op(op: Op, args: &[String]) -> String {
    match op {
        // Alap matematikai műveletek
        Op::AddF | Op::AddV2 | Op::AddV3 | Op::AddM2 | Op::AddM3 => {
            format!("({} + {})", args[0], args[1])
        }
        Op::SubF | Op::SubV2 | Op::SubV3 | Op::SubM2 | Op::SubM3 => {
            format!("({} - {})", args[0], args[1])
        }
        Op::MulF => format!("({} * {})", args[0], args[1]),
        Op::DivF => format!("({} / {})", args[0], args[1]),
        Op::SinF => format!("sin({})", args[0]),
        Op::CosF => format!("cos({})", args[0]),
        Op::ExpF => format!("exp({})", args[0]),
        Op::SqrF => format!("({})^2", args[0]),
        Op::SqrtF => format!("sqrt({})", args[0]),
        Op::LnF => format!("ln({})", args[0]),

        // Vektor/Mátrix Létrehozás és Elemek lekérése
        Op::MakeVec2 => format!("({}, {})", args[0], args[1]),
        Op::MakeVec3 => format!("({}, {}, {})", args[0], args[1], args[2]),
        Op::GetXV2 | Op::GetXV3 => format!("{}.x", args[0]),
        Op::GetYV2 | Op::GetYV3 => format!("{}.y", args[0]),
        Op::GetZV3 => format!("{}.z", args[0]),

        // Vektor/Mátrix műveletek
        Op::DotV2 | Op::DotV3 => format!("<{} • {}>", args[0], args[1]),
        Op::CrossV3 => format!("({} x {})", args[0], args[1]),
        Op::NormV2 | Op::NormV3 => format!("||{}||", args[0]),
        Op::ScaleV2
        | Op::ScaleV3
        | Op::ScaleM2
        | Op::ScaleM3
        | Op::MulM2
        | Op::MulM3
        | Op::MulM2V2
        | Op::MulM3V3 => format!("({} * {})", args[0], args[1]),
        Op::InverseM2 | Op::InverseM3 => format!("{}^-1", args[0]),
        Op::TransposeM2 | Op::TransposeM3 => format!("{}^T", args[0]),
        Op::DetM2 | Op::DetM3 => format!("det({})", args[0]),
        Op::TraceM2 | Op::TraceM3 => format!("tr({})", args[0]),

        // Logika
        Op::IfElseF => format!("(if {} then {} else {})", args[0], args[1], args[2]),

        _ => format!("{:?}({})", op, args.join(", ")),
    }
}
