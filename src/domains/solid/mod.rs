use crate::engine::domain::{Domain, SimplifyAction};
use crate::engine::eval::scalar::Scalar;
use crate::engine::eval::state::{DualVmState, VmState};
use crate::engine::eval::types::ValueType;

mod eval;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SolidOpCode {
    RightCauchyGreenM3,
    LeftCauchyGreenM3,
    Invariant2M3,
    CofactorM3,
    GreenLagrangeStrainM3,
    IsochoricInvariant1,
    IsochoricInvariant2,
    TraceSqrM3,
    DeviatoricM3,
    InvariantJ2M3,
    InvariantJ3M3,
}

pub struct SolidDomain;

impl Domain for SolidDomain {
    type OpCode = SolidOpCode;

    #[inline(always)]
    fn eval(op: Self::OpCode, ctx: &mut VmState) {
        unsafe {
            match op {
                SolidOpCode::RightCauchyGreenM3 => eval::eval_right_cauchy_green_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                SolidOpCode::LeftCauchyGreenM3 => eval::eval_left_cauchy_green_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                SolidOpCode::GreenLagrangeStrainM3 => eval::eval_green_lagrange_strain_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                SolidOpCode::IsochoricInvariant1 => eval::eval_isochoric_invariant1(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                SolidOpCode::IsochoricInvariant2 => eval::eval_isochoric_invariant2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                SolidOpCode::Invariant2M3 => eval::eval_invariant2_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                SolidOpCode::CofactorM3 => eval::eval_cofactor_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                SolidOpCode::TraceSqrM3 => eval::eval_trace_sqr_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                SolidOpCode::DeviatoricM3 => eval::eval_deviatoric_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                SolidOpCode::InvariantJ2M3 => eval::eval_invariant_j2_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                SolidOpCode::InvariantJ3M3 => eval::eval_invariant_j3_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
            }
        }
    }

    #[inline(always)]
    fn eval_dual(op: Self::OpCode, ctx: &mut DualVmState) {
        unsafe {
            match op {
                SolidOpCode::RightCauchyGreenM3 => eval::eval_dual_right_cauchy_green_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                SolidOpCode::LeftCauchyGreenM3 => eval::eval_dual_left_cauchy_green_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                SolidOpCode::GreenLagrangeStrainM3 => eval::eval_dual_green_lagrange_strain_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                SolidOpCode::IsochoricInvariant1 => eval::eval_dual_isochoric_invariant1(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                SolidOpCode::IsochoricInvariant2 => eval::eval_dual_isochoric_invariant2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                SolidOpCode::Invariant2M3 => eval::eval_dual_invariant2_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                SolidOpCode::CofactorM3 => eval::eval_dual_cofactor_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                SolidOpCode::TraceSqrM3 => eval::eval_dual_trace_sqr_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                SolidOpCode::DeviatoricM3 => eval::eval_dual_deviatoric_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                SolidOpCode::InvariantJ2M3 => eval::eval_dual_invariant_j2_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                SolidOpCode::InvariantJ3M3 => eval::eval_dual_invariant_j3_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
            }
        }
    }

    fn try_simplify(
        op: Self::OpCode,
        const_vals: &[Option<Scalar>],
        _args_equal: bool,
    ) -> SimplifyAction {
        if const_vals.len() == 1
            && let Some(Scalar::Mat3(m)) = const_vals[0]
        {
            match op {
                SolidOpCode::RightCauchyGreenM3 => {
                    let res = [
                        m[0] * m[0] + m[3] * m[3] + m[6] * m[6],
                        m[0] * m[1] + m[3] * m[4] + m[6] * m[7],
                        m[0] * m[2] + m[3] * m[5] + m[6] * m[8],
                        m[1] * m[0] + m[4] * m[3] + m[7] * m[6],
                        m[1] * m[1] + m[4] * m[4] + m[7] * m[7],
                        m[1] * m[2] + m[4] * m[5] + m[7] * m[8],
                        m[2] * m[0] + m[5] * m[3] + m[8] * m[6],
                        m[2] * m[1] + m[5] * m[4] + m[8] * m[7],
                        m[2] * m[2] + m[5] * m[5] + m[8] * m[8],
                    ];
                    return SimplifyAction::ReplaceWithConstant(Scalar::Mat3(res));
                }
                SolidOpCode::LeftCauchyGreenM3 => {
                    let res = [
                        m[0] * m[0] + m[1] * m[1] + m[2] * m[2],
                        m[0] * m[3] + m[1] * m[4] + m[2] * m[5],
                        m[0] * m[6] + m[1] * m[7] + m[2] * m[8],
                        m[3] * m[0] + m[4] * m[1] + m[5] * m[2],
                        m[3] * m[3] + m[4] * m[4] + m[5] * m[5],
                        m[3] * m[6] + m[4] * m[7] + m[5] * m[8],
                        m[6] * m[0] + m[7] * m[1] + m[8] * m[2],
                        m[6] * m[3] + m[7] * m[4] + m[8] * m[5],
                        m[6] * m[6] + m[7] * m[7] + m[8] * m[8],
                    ];
                    return SimplifyAction::ReplaceWithConstant(Scalar::Mat3(res));
                }
                SolidOpCode::GreenLagrangeStrainM3 => {
                    let res = [
                        0.5 * (m[0] * m[0] + m[3] * m[3] + m[6] * m[6] - 1.0),
                        0.5 * (m[0] * m[1] + m[3] * m[4] + m[6] * m[7]),
                        0.5 * (m[0] * m[2] + m[3] * m[5] + m[6] * m[8]),
                        0.5 * (m[1] * m[0] + m[4] * m[3] + m[7] * m[6]),
                        0.5 * (m[1] * m[1] + m[4] * m[4] + m[7] * m[7] - 1.0),
                        0.5 * (m[1] * m[2] + m[4] * m[5] + m[7] * m[8]),
                        0.5 * (m[2] * m[0] + m[5] * m[3] + m[8] * m[6]),
                        0.5 * (m[2] * m[1] + m[5] * m[4] + m[8] * m[7]),
                        0.5 * (m[2] * m[2] + m[5] * m[5] + m[8] * m[8] - 1.0),
                    ];
                    return SimplifyAction::ReplaceWithConstant(Scalar::Mat3(res));
                }
                SolidOpCode::IsochoricInvariant1 => {
                    let det = m[0] * (m[4] * m[8] - m[5] * m[7])
                        - m[3] * (m[1] * m[8] - m[2] * m[7])
                        + m[6] * (m[1] * m[5] - m[2] * m[4]);
                    let j_pow = det.powf(-2.0 / 3.0);
                    let i1 = m[0] * m[0]
                        + m[1] * m[1]
                        + m[2] * m[2]
                        + m[3] * m[3]
                        + m[4] * m[4]
                        + m[5] * m[5]
                        + m[6] * m[6]
                        + m[7] * m[7]
                        + m[8] * m[8];
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(j_pow * i1));
                }
                SolidOpCode::IsochoricInvariant2 => {
                    let det = m[0] * (m[4] * m[8] - m[5] * m[7])
                        - m[3] * (m[1] * m[8] - m[2] * m[7])
                        + m[6] * (m[1] * m[5] - m[2] * m[4]);
                    let j_pow = det.powf(-4.0 / 3.0);

                    let i1 = m[0] * m[0]
                        + m[1] * m[1]
                        + m[2] * m[2]
                        + m[3] * m[3]
                        + m[4] * m[4]
                        + m[5] * m[5]
                        + m[6] * m[6]
                        + m[7] * m[7]
                        + m[8] * m[8];

                    let c00 = m[0] * m[0] + m[3] * m[3] + m[6] * m[6];
                    let c01 = m[0] * m[1] + m[3] * m[4] + m[6] * m[7];
                    let c02 = m[0] * m[2] + m[3] * m[5] + m[6] * m[8];
                    let c10 = m[1] * m[0] + m[4] * m[3] + m[7] * m[6];
                    let c11 = m[1] * m[1] + m[4] * m[4] + m[7] * m[7];
                    let c12 = m[1] * m[2] + m[4] * m[5] + m[7] * m[8];
                    let c20 = m[2] * m[0] + m[5] * m[3] + m[8] * m[6];
                    let c21 = m[2] * m[1] + m[5] * m[4] + m[8] * m[7];
                    let c22 = m[2] * m[2] + m[5] * m[5] + m[8] * m[8];

                    let tr_c2 = c00 * c00
                        + c01 * c10
                        + c02 * c20
                        + c10 * c01
                        + c11 * c11
                        + c12 * c21
                        + c20 * c02
                        + c21 * c12
                        + c22 * c22;

                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(
                        j_pow * 0.5 * (i1 * i1 - tr_c2),
                    ));
                }
                SolidOpCode::Invariant2M3 => {
                    let tr_m = m[0] + m[4] + m[8];
                    let tr_m2 = (m[0] * m[0] + m[1] * m[3] + m[2] * m[6])
                        + (m[3] * m[1] + m[4] * m[4] + m[5] * m[7])
                        + (m[6] * m[2] + m[7] * m[5] + m[8] * m[8]);
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(
                        0.5 * (tr_m * tr_m - tr_m2),
                    ));
                }
                SolidOpCode::CofactorM3 => {
                    let res = [
                        m[4] * m[8] - m[5] * m[7],
                        m[5] * m[6] - m[3] * m[8],
                        m[3] * m[7] - m[4] * m[6],
                        m[2] * m[7] - m[1] * m[8],
                        m[0] * m[8] - m[2] * m[6],
                        m[1] * m[6] - m[0] * m[7],
                        m[1] * m[5] - m[2] * m[4],
                        m[2] * m[3] - m[0] * m[5],
                        m[0] * m[4] - m[1] * m[3],
                    ];
                    return SimplifyAction::ReplaceWithConstant(Scalar::Mat3(res));
                }
                SolidOpCode::TraceSqrM3 => {
                    let res = m[0] * m[0] + m[1] * m[3] + m[2] * m[6]
                            + m[3] * m[1] + m[4] * m[4] + m[5] * m[7]
                            + m[6] * m[2] + m[7] * m[5] + m[8] * m[8];
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(res));
                }
                SolidOpCode::DeviatoricM3 => {
                    let tr_third = (m[0] + m[4] + m[8]) / 3.0;
                    let res = [
                        m[0] - tr_third, m[1], m[2],
                        m[3], m[4] - tr_third, m[5],
                        m[6], m[7], m[8] - tr_third,
                    ];
                    return SimplifyAction::ReplaceWithConstant(Scalar::Mat3(res));
                }
                SolidOpCode::InvariantJ2M3 => {
                    let tr_third = (m[0] + m[4] + m[8]) / 3.0;
                    let s0 = m[0] - tr_third; let s1 = m[1]; let s2 = m[2];
                    let s3 = m[3]; let s4 = m[4] - tr_third; let s5 = m[5];
                    let s6 = m[6]; let s7 = m[7]; let s8 = m[8] - tr_third;
                    let j2 = 0.5 * (s0*s0 + s1*s3 + s2*s6 + s3*s1 + s4*s4 + s5*s7 + s6*s2 + s7*s5 + s8*s8);
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(j2));
                }
                SolidOpCode::InvariantJ3M3 => {
                    let tr_third = (m[0] + m[4] + m[8]) / 3.0;
                    let s0 = m[0] - tr_third; let s1 = m[1]; let s2 = m[2];
                    let s3 = m[3]; let s4 = m[4] - tr_third; let s5 = m[5];
                    let s6 = m[6]; let s7 = m[7]; let s8 = m[8] - tr_third;
                    let j3 = s0 * (s4 * s8 - s5 * s7) - s3 * (s1 * s8 - s2 * s7) + s6 * (s1 * s5 - s2 * s4);
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(j3));
                }
            }
        }
        SimplifyAction::None
    }

    fn arity(_op: Self::OpCode) -> usize {
        1
    }

    fn return_type(op: Self::OpCode) -> ValueType {
        match op {
            SolidOpCode::Invariant2M3
            | SolidOpCode::IsochoricInvariant1
            | SolidOpCode::IsochoricInvariant2
            | SolidOpCode::TraceSqrM3
            | SolidOpCode::InvariantJ2M3
            | SolidOpCode::InvariantJ3M3 => ValueType::Float,
            _ => ValueType::Mat3,
        }
    }

    fn expected_types(_op: Self::OpCode) -> &'static [ValueType] {
        &[ValueType::Mat3]
    }

    fn weight(op: Self::OpCode) -> usize {
        match op {
            SolidOpCode::DeviatoricM3 => 3,
            SolidOpCode::Invariant2M3 | SolidOpCode::TraceSqrM3 => 4,
            SolidOpCode::RightCauchyGreenM3 | SolidOpCode::LeftCauchyGreenM3 => 4,
            SolidOpCode::GreenLagrangeStrainM3 | SolidOpCode::CofactorM3 => 5,
            SolidOpCode::InvariantJ2M3 | SolidOpCode::InvariantJ3M3 => 5,
            SolidOpCode::IsochoricInvariant1 | SolidOpCode::IsochoricInvariant2 => 6,
        }
    }

    fn is_differentiable(_op: Self::OpCode) -> bool {
        true
    }

    fn is_forbidden_child(parent: Self::OpCode, child: Self::OpCode) -> bool {
        match parent {
            SolidOpCode::RightCauchyGreenM3
            | SolidOpCode::LeftCauchyGreenM3
            | SolidOpCode::GreenLagrangeStrainM3
            | SolidOpCode::IsochoricInvariant1
            | SolidOpCode::IsochoricInvariant2
            | SolidOpCode::CofactorM3 => {
                matches!(
                    child,
                    SolidOpCode::RightCauchyGreenM3
                        | SolidOpCode::LeftCauchyGreenM3
                        | SolidOpCode::GreenLagrangeStrainM3
                        | SolidOpCode::IsochoricInvariant1
                        | SolidOpCode::IsochoricInvariant2
                        | SolidOpCode::CofactorM3
                        | SolidOpCode::Invariant2M3
                        | SolidOpCode::TraceSqrM3
                        | SolidOpCode::DeviatoricM3
                )
            }
            SolidOpCode::Invariant2M3 => {
                matches!(
                    child,
                    SolidOpCode::GreenLagrangeStrainM3
                        | SolidOpCode::CofactorM3
                        | SolidOpCode::IsochoricInvariant1
                        | SolidOpCode::IsochoricInvariant2
                        | SolidOpCode::Invariant2M3
                        | SolidOpCode::TraceSqrM3
                )
            }
            SolidOpCode::DeviatoricM3 => {
                matches!(child, SolidOpCode::DeviatoricM3)
            }
            SolidOpCode::InvariantJ2M3 | SolidOpCode::InvariantJ3M3 => {
                matches!(child, SolidOpCode::DeviatoricM3 | SolidOpCode::RightCauchyGreenM3 | SolidOpCode::LeftCauchyGreenM3)
            }
            SolidOpCode::TraceSqrM3 => false
        }
    }

    fn format_op(op: Self::OpCode, args: &[String]) -> String {
        match op {
            SolidOpCode::RightCauchyGreenM3 => format!("C({})", args[0]),
            SolidOpCode::LeftCauchyGreenM3 => format!("B({})", args[0]),
            SolidOpCode::GreenLagrangeStrainM3 => format!("E({})", args[0]),
            SolidOpCode::IsochoricInvariant1 => format!("I1_bar({})", args[0]),
            SolidOpCode::IsochoricInvariant2 => format!("I2_bar({})", args[0]),
            SolidOpCode::Invariant2M3 => format!("I2({})", args[0]),
            SolidOpCode::CofactorM3 => format!("Cof({})", args[0]),
            SolidOpCode::TraceSqrM3 => format!("tr({}^2)", args[0]),
            SolidOpCode::DeviatoricM3 => format!("dev({})", args[0]),
            SolidOpCode::InvariantJ2M3 => format!("J2({})", args[0]),
            SolidOpCode::InvariantJ3M3 => format!("J3({})", args[0]),
        }
    }
}