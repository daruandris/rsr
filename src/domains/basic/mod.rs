use crate::engine::domain::{Domain, SimplifyAction};
use crate::engine::eval::autodiff;
use crate::engine::eval::scalar::Scalar;
use crate::engine::eval::state::{DualVmState, VmState};

mod eval;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BasicOpCode {
    AddF,
    SubF,
    MulF,
    DivF,
    SinF,
    CosF,
    ExpF,
    SqrF,
    SqrtF,
    LnF,
}

pub struct BasicDomain;

impl Domain for BasicDomain {
    type OpCode = BasicOpCode;

    #[inline(always)]
    fn eval(op: Self::OpCode, ctx: &mut VmState) {
        unsafe {
            match op {
                BasicOpCode::AddF => eval::eval_add_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SubF => eval::eval_sub_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::MulF => eval::eval_mul_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::DivF => eval::eval_div_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SinF => eval::eval_sin_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::CosF => eval::eval_cos_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::ExpF => eval::eval_exp_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SqrF => eval::eval_sqr_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SqrtF => eval::eval_sqrt_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::LnF => eval::eval_ln_f(&mut ctx.sp_f, &mut ctx.stack_f),
            }
        }
    }

    fn try_simplify(
        op: Self::OpCode,
        const_vals: &[Option<Scalar>],
        args_equal: bool,
    ) -> SimplifyAction {
        let all_const = const_vals.iter().all(|c| c.is_some());
        if all_const
            && !const_vals.is_empty()
            && let Scalar::Float(a) = const_vals[0].unwrap()
        {
            if const_vals.len() == 1 {
                let res = match op {
                    BasicOpCode::SinF => a.sin(),
                    BasicOpCode::CosF => a.cos(),
                    BasicOpCode::ExpF => a.exp(),
                    BasicOpCode::SqrF => a * a,
                    BasicOpCode::SqrtF => a.sqrt(),
                    BasicOpCode::LnF => a.ln(),
                    _ => f32::NAN,
                };
                if res.is_finite() {
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(res));
                }
            } else if const_vals.len() == 2
                && let Scalar::Float(b) = const_vals[1].unwrap()
            {
                let res = match op {
                    BasicOpCode::AddF => a + b,
                    BasicOpCode::SubF => a - b,
                    BasicOpCode::MulF => a * b,
                    BasicOpCode::DivF => a / b,
                    _ => f32::NAN,
                };
                if res.is_finite() {
                    return SimplifyAction::ReplaceWithConstant(Scalar::Float(res));
                }
            }
        }

        if const_vals.len() == 2 {
            let a_is_zero = const_vals[0].as_ref().is_some_and(|c| c.is_zero());
            let b_is_zero = const_vals[1].as_ref().is_some_and(|c| c.is_zero());
            let a_is_one = const_vals[0].as_ref().is_some_and(|c| c.is_one());
            let b_is_one = const_vals[1].as_ref().is_some_and(|c| c.is_one());

            match op {
                BasicOpCode::AddF => {
                    if b_is_zero {
                        return SimplifyAction::KeepArg(0);
                    }
                    if a_is_zero {
                        return SimplifyAction::KeepArg(1);
                    }
                }
                BasicOpCode::MulF => {
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
                BasicOpCode::SubF => {
                    if b_is_zero {
                        return SimplifyAction::KeepArg(0);
                    }
                    if args_equal {
                        return SimplifyAction::ReplaceWithConstant(Scalar::Float(0.0));
                    }
                }
                BasicOpCode::DivF => {
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
                _ => {}
            }
        }
        SimplifyAction::None
    }

    fn arity(op: Self::OpCode) -> usize {
        match op {
            BasicOpCode::AddF | BasicOpCode::SubF | BasicOpCode::MulF | BasicOpCode::DivF => 2,
            _ => 1,
        }
    }

    fn return_type(_op: Self::OpCode) -> crate::engine::eval::types::ValueType {
        crate::engine::eval::types::ValueType::Float
    }

    fn expected_types(op: Self::OpCode) -> &'static [crate::engine::eval::types::ValueType] {
        match op {
            BasicOpCode::AddF | BasicOpCode::SubF | BasicOpCode::MulF | BasicOpCode::DivF => &[
                crate::engine::eval::types::ValueType::Float,
                crate::engine::eval::types::ValueType::Float,
            ],
            _ => &[crate::engine::eval::types::ValueType::Float],
        }
    }

    fn weight(op: Self::OpCode) -> usize {
        match op {
            BasicOpCode::AddF | BasicOpCode::SubF | BasicOpCode::MulF => 1,
            BasicOpCode::DivF | BasicOpCode::SqrF | BasicOpCode::SqrtF => 2,
            BasicOpCode::SinF | BasicOpCode::CosF => 3,
            BasicOpCode::ExpF | BasicOpCode::LnF => 4,
        }
    }

    fn is_forbidden_child(parent: Self::OpCode, child: Self::OpCode) -> bool {
        match parent {
            BasicOpCode::SinF | BasicOpCode::CosF => matches!(
                child,
                BasicOpCode::SinF | BasicOpCode::CosF | BasicOpCode::ExpF | BasicOpCode::LnF
            ),
            BasicOpCode::ExpF => matches!(
                child,
                BasicOpCode::ExpF
                    | BasicOpCode::SinF
                    | BasicOpCode::CosF
                    | BasicOpCode::SqrF
                    | BasicOpCode::LnF
            ),
            BasicOpCode::SqrtF => matches!(
                child,
                BasicOpCode::SqrtF
                    | BasicOpCode::SqrF
                    | BasicOpCode::SinF
                    | BasicOpCode::CosF
                    | BasicOpCode::LnF
                    | BasicOpCode::ExpF
            ),
            BasicOpCode::SqrF => matches!(child, BasicOpCode::SqrF | BasicOpCode::SqrtF),
            BasicOpCode::LnF => matches!(
                child,
                BasicOpCode::LnF | BasicOpCode::ExpF | BasicOpCode::SinF | BasicOpCode::CosF
            ),
            _ => false,
        }
    }

    fn is_differentiable(_op: Self::OpCode) -> bool {
        true
    }
    fn requires_cmaes(_op: Self::OpCode) -> bool {
        false
    }

    #[inline(always)]
    fn eval_dual(op: Self::OpCode, ctx: &mut DualVmState) {
        unsafe {
            match op {
                BasicOpCode::AddF => autodiff::eval_add_dual_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SubF => autodiff::eval_sub_dual_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::MulF => autodiff::eval_mul_dual_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::DivF => autodiff::eval_div_dual_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SinF => autodiff::eval_sin_dual_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::CosF => autodiff::eval_cos_dual_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::ExpF => autodiff::eval_exp_dual_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SqrF => autodiff::eval_sqr_dual_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SqrtF => autodiff::eval_sqrt_dual_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::LnF => autodiff::eval_ln_dual_f(&mut ctx.sp_f, &mut ctx.stack_f),
            }
        }
    }

    fn format_op(op: Self::OpCode, args: &[String]) -> String {
        match op {
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
        }
    }
}
