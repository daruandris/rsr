use super::node::Node;
use crate::eval::op::Op;
use crate::eval::scalar::Scalar;

enum SimplifyAction {
    ReplaceWithConstant(Scalar),
    KeepArg(usize),
    None,
}

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
                let action = try_simplify(op, &const_vals, args_equal);

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
                        if arity == 1 && output.len() > args[0].start_idx {
                            if let Node::Operator(child_op) = output[output.len() - 1] {
                                match (op, child_op) {
                                    (Op::LnF, Op::ExpF)
                                    | (Op::ExpF, Op::LnF)
                                    | (Op::SqrtF, Op::SqrF)
                                    | (Op::SqrF, Op::SqrtF) => {
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

fn try_simplify(op: Op, const_vals: &[Option<Scalar>], args_equal: bool) -> SimplifyAction {
    let all_const = const_vals.iter().all(|c| c.is_some());

    // 1. Konstansok összevonása (Constant folding)
    if all_const {
        let vals: Vec<Scalar> = const_vals.iter().map(|c| c.unwrap()).collect();
        if let Some(folded) = fold_constants(op, &vals) {
            if let Scalar::Float(f) = folded {
                if f.is_finite() {
                    return SimplifyAction::ReplaceWithConstant(folded);
                }
            } else {
                return SimplifyAction::ReplaceWithConstant(folded);
            }
        }
    }

    // 2. Algebrai egyszerűsítések (Identitások)
    if const_vals.len() == 2 {
        let a_is_zero = const_vals[0].as_ref().map_or(false, |c| c.is_zero());
        let b_is_zero = const_vals[1].as_ref().map_or(false, |c| c.is_zero());
        let a_is_one = const_vals[0].as_ref().map_or(false, |c| c.is_one());
        let b_is_one = const_vals[1].as_ref().map_or(false, |c| c.is_one());

        match op {
            // Skaláris
            Op::AddF => {
                if b_is_zero {
                    return SimplifyAction::KeepArg(0);
                }
                if a_is_zero {
                    return SimplifyAction::KeepArg(1);
                }
            }
            Op::SubF => {
                if b_is_zero {
                    return SimplifyAction::KeepArg(0);
                }
                if args_equal {
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(0.0));
                }
            }
            Op::MulF => {
                if b_is_one {
                    return SimplifyAction::KeepArg(0);
                }
                if a_is_one {
                    return SimplifyAction::KeepArg(1);
                }
                if a_is_zero || b_is_zero {
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(0.0));
                }
            }
            Op::DivF => {
                if b_is_one {
                    return SimplifyAction::KeepArg(0);
                }
                if a_is_zero && !b_is_zero {
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(0.0));
                }
                if args_equal {
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(1.0));
                }
            }

            // Linalg Add/Sub
            Op::AddV2 | Op::AddV3 | Op::AddM2 | Op::AddM3 => {
                if b_is_zero {
                    return SimplifyAction::KeepArg(0);
                }
                if a_is_zero {
                    return SimplifyAction::KeepArg(1);
                }
            }
            Op::SubV2 | Op::SubV3 | Op::SubM2 | Op::SubM3 => {
                if b_is_zero {
                    return SimplifyAction::KeepArg(0);
                }
                if args_equal {
                    return SimplifyAction::ReplaceWithConstant(zero_for_op(op));
                }
            }

            // Linalg Scale
            Op::ScaleV2 | Op::ScaleV3 | Op::ScaleM2 | Op::ScaleM3 => {
                if a_is_zero {
                    return SimplifyAction::ReplaceWithConstant(zero_for_op(op));
                }
                if a_is_one {
                    return SimplifyAction::KeepArg(1);
                }
            }

            // Linalg Dot/Cross
            Op::DotV2 | Op::DotV3 => {
                if a_is_zero || b_is_zero {
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(0.0));
                }
            }
            Op::CrossV3 => {
                if a_is_zero || b_is_zero || args_equal {
                    return SimplifyAction::ReplaceWithConstant(Scalar::Vec3([0.0; 3]));
                }
            }

            // Mátrix szorzások
            Op::MulM2 | Op::MulM3 => {
                if const_vals[0].as_ref().map_or(false, |c| c.is_identity()) {
                    return SimplifyAction::KeepArg(1);
                }
                if const_vals[1].as_ref().map_or(false, |c| c.is_identity()) {
                    return SimplifyAction::KeepArg(0);
                }
                if a_is_zero || b_is_zero {
                    return SimplifyAction::ReplaceWithConstant(zero_for_op(op));
                }
            }
            Op::MulM2V2 | Op::MulM3V3 => {
                if const_vals[0].as_ref().map_or(false, |c| c.is_identity()) {
                    return SimplifyAction::KeepArg(1);
                }
                if a_is_zero || b_is_zero {
                    return SimplifyAction::ReplaceWithConstant(zero_for_op(op));
                }
            }
            _ => {}
        }
    }

    SimplifyAction::None
}

// Segédfüggvény a megfelelő típusú 0 generálásához
fn zero_for_op(op: Op) -> Scalar {
    match op {
        Op::SubV2 | Op::ScaleV2 | Op::MulM2V2 => Scalar::Vec2([0.0; 2]),
        Op::SubV3 | Op::ScaleV3 | Op::CrossV3 | Op::MulM3V3 => Scalar::Vec3([0.0; 3]),
        Op::SubM2 | Op::ScaleM2 | Op::MulM2 => Scalar::Mat2([0.0; 4]),
        Op::SubM3 | Op::ScaleM3 | Op::MulM3 => Scalar::Mat3([0.0; 9]),
        _ => Scalar::Float(0.0),
    }
}

// Maga a konstans művelet elvégzője
fn fold_constants(op: Op, args: &[Scalar]) -> Option<Scalar> {
    use Scalar::*;
    match (op, args) {
        (Op::SinF, [Float(a)]) => Some(Float(a.sin())),
        (Op::CosF, [Float(a)]) => Some(Float(a.cos())),
        (Op::ExpF, [Float(a)]) => Some(Float(a.exp())),
        (Op::SqrF, [Float(a)]) => Some(Float(a * a)),
        (Op::SqrtF, [Float(a)]) => Some(Float(a.sqrt())),
        (Op::LnF, [Float(a)]) => Some(Float(a.ln())),

        (Op::AddF, [Float(a), Float(b)]) => Some(Float(a + b)),
        (Op::SubF, [Float(a), Float(b)]) => Some(Float(a - b)),
        (Op::MulF, [Float(a), Float(b)]) => Some(Float(a * b)),
        (Op::DivF, [Float(a), Float(b)]) => Some(Float(a / b)),

        (Op::MakeVec2, [Float(x), Float(y)]) => Some(Vec2([*x, *y])),
        (Op::MakeVec3, [Float(x), Float(y), Float(z)]) => Some(Vec3([*x, *y, *z])),
        (Op::GetXV2, [Vec2(v)]) => Some(Float(v[0])),
        (Op::GetYV2, [Vec2(v)]) => Some(Float(v[1])),
        (Op::GetXV3, [Vec3(v)]) => Some(Float(v[0])),
        (Op::GetYV3, [Vec3(v)]) => Some(Float(v[1])),
        (Op::GetZV3, [Vec3(v)]) => Some(Float(v[2])),

        (Op::AddV2, [Vec2(a), Vec2(b)]) => Some(Vec2([a[0] + b[0], a[1] + b[1]])),
        (Op::SubV2, [Vec2(a), Vec2(b)]) => Some(Vec2([a[0] - b[0], a[1] - b[1]])),
        (Op::ScaleV2, [Float(s), Vec2(v)]) => Some(Vec2([s * v[0], s * v[1]])),
        (Op::DotV2, [Vec2(a), Vec2(b)]) => Some(Float(a[0] * b[0] + a[1] * b[1])),
        (Op::NormV2, [Vec2(v)]) => Some(Float((v[0] * v[0] + v[1] * v[1]).sqrt())),

        (Op::AddV3, [Vec3(a), Vec3(b)]) => Some(Vec3([a[0] + b[0], a[1] + b[1], a[2] + b[2]])),
        (Op::SubV3, [Vec3(a), Vec3(b)]) => Some(Vec3([a[0] - b[0], a[1] - b[1], a[2] - b[2]])),
        (Op::ScaleV3, [Float(s), Vec3(v)]) => Some(Vec3([s * v[0], s * v[1], s * v[2]])),
        (Op::DotV3, [Vec3(a), Vec3(b)]) => Some(Float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])),
        (Op::NormV3, [Vec3(v)]) => Some(Float((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())),
        (Op::CrossV3, [Vec3(a), Vec3(b)]) => Some(Vec3([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])),

        (Op::MakeMat2, [Vec2(c0), Vec2(c1)]) => Some(Mat2([c0[0], c0[1], c1[0], c1[1]])),
        (Op::AddM2, [Mat2(a), Mat2(b)]) => {
            Some(Mat2([a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]))
        }
        (Op::SubM2, [Mat2(a), Mat2(b)]) => {
            Some(Mat2([a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]))
        }
        (Op::ScaleM2, [Float(s), Mat2(m)]) => Some(Mat2([s * m[0], s * m[1], s * m[2], s * m[3]])),
        (Op::MulM2, [Mat2(a), Mat2(b)]) => Some(Mat2([
            a[0] * b[0] + a[2] * b[1],
            a[1] * b[0] + a[3] * b[1],
            a[0] * b[2] + a[2] * b[3],
            a[1] * b[2] + a[3] * b[3],
        ])),
        (Op::MulM2V2, [Mat2(m), Vec2(v)]) => {
            Some(Vec2([m[0] * v[0] + m[2] * v[1], m[1] * v[0] + m[3] * v[1]]))
        }
        (Op::DetM2, [Mat2(m)]) => Some(Float(m[0] * m[3] - m[1] * m[2])),
        (Op::TraceM2, [Mat2(m)]) => Some(Float(m[0] + m[3])),
        (Op::TransposeM2, [Mat2(m)]) => Some(Mat2([m[0], m[2], m[1], m[3]])),
        (Op::InverseM2, [Mat2(m)]) => {
            let det = m[0] * m[3] - m[1] * m[2];
            if det.abs() > 1e-9 {
                Some(Mat2([m[3] / det, -m[1] / det, -m[2] / det, m[0] / det]))
            } else {
                None
            }
        }

        (Op::MakeMat3, [Vec3(c0), Vec3(c1), Vec3(c2)]) => Some(Mat3([
            c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2],
        ])),
        (Op::AddM3, [Mat3(a), Mat3(b)]) => Some(Mat3([
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
            a[3] + b[3],
            a[4] + b[4],
            a[5] + b[5],
            a[6] + b[6],
            a[7] + b[7],
            a[8] + b[8],
        ])),
        (Op::SubM3, [Mat3(a), Mat3(b)]) => Some(Mat3([
            a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2],
            a[3] - b[3],
            a[4] - b[4],
            a[5] - b[5],
            a[6] - b[6],
            a[7] - b[7],
            a[8] - b[8],
        ])),
        (Op::ScaleM3, [Float(s), Mat3(m)]) => Some(Mat3([
            s * m[0],
            s * m[1],
            s * m[2],
            s * m[3],
            s * m[4],
            s * m[5],
            s * m[6],
            s * m[7],
            s * m[8],
        ])),
        (Op::MulM3, [Mat3(a), Mat3(b)]) => Some(Mat3([
            a[0] * b[0] + a[3] * b[1] + a[6] * b[2],
            a[1] * b[0] + a[4] * b[1] + a[7] * b[2],
            a[2] * b[0] + a[5] * b[1] + a[8] * b[2],
            a[0] * b[3] + a[3] * b[4] + a[6] * b[5],
            a[1] * b[3] + a[4] * b[4] + a[7] * b[5],
            a[2] * b[3] + a[5] * b[4] + a[8] * b[5],
            a[0] * b[6] + a[3] * b[7] + a[6] * b[8],
            a[1] * b[6] + a[4] * b[7] + a[7] * b[8],
            a[2] * b[6] + a[5] * b[7] + a[8] * b[8],
        ])),
        (Op::MulM3V3, [Mat3(m), Vec3(v)]) => Some(Vec3([
            m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
            m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
            m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
        ])),
        (Op::TraceM3, [Mat3(m)]) => Some(Float(m[0] + m[4] + m[8])),
        (Op::TransposeM3, [Mat3(m)]) => {
            Some(Mat3([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]))
        }
        (Op::DetM3, [Mat3(m)]) => Some(Float(
            m[0] * (m[4] * m[8] - m[5] * m[7]) - m[3] * (m[1] * m[8] - m[2] * m[7])
                + m[6] * (m[1] * m[5] - m[2] * m[4]),
        )),
        (Op::InverseM3, [Mat3(m)]) => {
            let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[3] * (m[1] * m[8] - m[2] * m[7])
                + m[6] * (m[1] * m[5] - m[2] * m[4]);
            if det.abs() > 1e-9 {
                let inv_d = 1.0 / det;
                Some(Mat3([
                    (m[4] * m[8] - m[5] * m[7]) * inv_d,
                    -(m[1] * m[8] - m[2] * m[7]) * inv_d,
                    (m[1] * m[5] - m[2] * m[4]) * inv_d,
                    -(m[3] * m[8] - m[5] * m[6]) * inv_d,
                    (m[0] * m[8] - m[2] * m[6]) * inv_d,
                    -(m[0] * m[5] - m[2] * m[3]) * inv_d,
                    (m[3] * m[7] - m[4] * m[6]) * inv_d,
                    -(m[0] * m[7] - m[1] * m[6]) * inv_d,
                    (m[0] * m[4] - m[1] * m[3]) * inv_d,
                ]))
            } else {
                None
            }
        }
        _ => None,
    }
}
