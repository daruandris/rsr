use super::node::Node;
use crate::Instruction;
use crate::eval::basic_domain::BasicOpCode;
use crate::eval::linalg_domain::LinalgOpCode;

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
    stack.pop().unwrap_or_else(|| "Empty expression".to_string())
}

fn format_op(op: Instruction, args: &[String]) -> String {
    match op {
        Instruction::Basic(b) => match b {
            BasicOpCode::AddF => format!("({} + {})", args[0], args[1]),
            BasicOpCode::SubF => format!("({} - {})", args[0], args[1]),
            BasicOpCode::MulF => format!("({} * {})", args[0], args[1]),
            BasicOpCode::DivF => format!("({} / {})", args[0], args[1]),
            BasicOpCode::SinF => format!("sin({})", args[0]),
            BasicOpCode::CosF => format!("cos({})", args[0]),
            BasicOpCode::ExpF => format!("exp({})", args[0]),
            BasicOpCode::SqrF => format!("({})^2", args[0]),
            BasicOpCode::SqrtF => format!("sqrt({})", args[0]),
            BasicOpCode::LnF => format!("ln({})", args[0]),
        },
        Instruction::Linalg(l) => match l {
            LinalgOpCode::MakeVec2 => format!("({}, {})", args[0], args[1]),
            LinalgOpCode::MakeVec3 => format!("({}, {}, {})", args[0], args[1], args[2]),
            LinalgOpCode::GetXV2 | LinalgOpCode::GetXV3 => format!("{}.x", args[0]),
            LinalgOpCode::GetYV2 | LinalgOpCode::GetYV3 => format!("{}.y", args[0]),
            LinalgOpCode::GetZV3 => format!("{}.z", args[0]),
            LinalgOpCode::DotV2 | LinalgOpCode::DotV3 => format!("<{} • {}>", args[0], args[1]),
            LinalgOpCode::CrossV3 => format!("({} x {})", args[0], args[1]),
            LinalgOpCode::NormV2 | LinalgOpCode::NormV3 => format!("||{}||", args[0]),
            LinalgOpCode::AddV2 | LinalgOpCode::AddV3 | LinalgOpCode::AddM2 | LinalgOpCode::AddM3 => format!("({} + {})", args[0], args[1]),
            LinalgOpCode::SubV2 | LinalgOpCode::SubV3 | LinalgOpCode::SubM2 | LinalgOpCode::SubM3 => format!("({} - {})", args[0], args[1]),
            LinalgOpCode::ScaleV2 | LinalgOpCode::ScaleV3 | LinalgOpCode::ScaleM2 | LinalgOpCode::ScaleM3 |
            LinalgOpCode::MulM2 | LinalgOpCode::MulM3 | LinalgOpCode::MulM2V2 | LinalgOpCode::MulM3V3 => format!("({} * {})", args[0], args[1]),
            LinalgOpCode::InverseM2 | LinalgOpCode::InverseM3 => format!("{}^-1", args[0]),
            LinalgOpCode::TransposeM2 | LinalgOpCode::TransposeM3 => format!("{}^T", args[0]),
            LinalgOpCode::DetM2 | LinalgOpCode::DetM3 => format!("det({})", args[0]),
            LinalgOpCode::TraceM2 | LinalgOpCode::TraceM3 => format!("tr({})", args[0]),
            LinalgOpCode::MakeMat2 => format!("({}, {})", args[0], args[1]),
            LinalgOpCode::MakeMat3 => format!("({}, {}, {})", args[0], args[1], args[2]),
        },
        _ => format!("{:?}({})", op, args.join(", ")),
    }
}