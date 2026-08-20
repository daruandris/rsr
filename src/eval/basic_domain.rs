use crate::domain::{Domain, SimplifyAction};
use crate::eval::state::VmState;
use crate::eval::scalar::Scalar;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BasicOpCode {
    AddF, SubF, MulF, DivF,
    SinF, CosF, ExpF, SqrF, SqrtF, LnF,
}

pub struct BasicDomain;

impl Domain for BasicDomain {
    type OpCode = BasicOpCode;

    #[inline(always)]
    fn eval(op: Self::OpCode, ctx: &mut VmState) {
        unsafe {
            match op {
                BasicOpCode::AddF => crate::eval::basic::eval_add_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SubF => crate::eval::basic::eval_sub_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::MulF => crate::eval::basic::eval_mul_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::DivF => crate::eval::basic::eval_div_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SinF => crate::eval::basic::eval_sin_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::CosF => crate::eval::basic::eval_cos_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::ExpF => crate::eval::basic::eval_exp_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SqrF => crate::eval::basic::eval_sqr_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::SqrtF => crate::eval::basic::eval_sqrt_f(&mut ctx.sp_f, &mut ctx.stack_f),
                BasicOpCode::LnF => crate::eval::basic::eval_ln_f(&mut ctx.sp_f, &mut ctx.stack_f),
            }
        }
    }

    fn try_simplify(op: Self::OpCode, const_vals: &[Option<Scalar>], args_equal: bool) -> SimplifyAction {
        let all_const = const_vals.iter().all(|c| c.is_some());
        if all_const && !const_vals.is_empty() {
            if let Scalar::Float(a) = const_vals[0].unwrap() {
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
                } else if const_vals.len() == 2 {
                    if let Scalar::Float(b) = const_vals[1].unwrap() {
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
            }
        }

        if const_vals.len() == 2 {
            let a_is_zero = const_vals[0].as_ref().map_or(false, |c| c.is_zero());
            let b_is_zero = const_vals[1].as_ref().map_or(false, |c| c.is_zero());
            let a_is_one = const_vals[0].as_ref().map_or(false, |c| c.is_one());
            let b_is_one = const_vals[1].as_ref().map_or(false, |c| c.is_one());

            match op {
                BasicOpCode::AddF => {
                    if b_is_zero { return SimplifyAction::KeepArg(0); }
                    if a_is_zero { return SimplifyAction::KeepArg(1); }
                }
                BasicOpCode::MulF => {
                    if b_is_one { return SimplifyAction::KeepArg(0); }
                    if a_is_one { return SimplifyAction::KeepArg(1); }
                    if a_is_zero || b_is_zero { return SimplifyAction::ReplaceWithConstant(Scalar::Float(0.0)); }
                }
                BasicOpCode::SubF => {
                    if b_is_zero { return SimplifyAction::KeepArg(0); }
                    if args_equal { return SimplifyAction::ReplaceWithConstant(Scalar::Float(0.0)); }
                }
                BasicOpCode::DivF => {
                    if b_is_one { return SimplifyAction::KeepArg(0); }
                    if a_is_zero && !b_is_zero { return SimplifyAction::ReplaceWithConstant(Scalar::Float(0.0)); }
                    if args_equal { return SimplifyAction::ReplaceWithConstant(Scalar::Float(1.0)); }
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

    fn return_type(_op: Self::OpCode) -> crate::eval::types::ValueType {
        crate::eval::types::ValueType::Float
    }

    fn expected_types(op: Self::OpCode) -> &'static [crate::eval::types::ValueType] {
        match op {
            BasicOpCode::AddF | BasicOpCode::SubF | BasicOpCode::MulF | BasicOpCode::DivF => &[crate::eval::types::ValueType::Float, crate::eval::types::ValueType::Float],
            _ => &[crate::eval::types::ValueType::Float],
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

    fn forbidden_children(op: Self::OpCode) -> &'static [crate::Instruction] {
        use crate::Instruction::Basic;
        match op {
            BasicOpCode::SinF | BasicOpCode::CosF => &[Basic(BasicOpCode::SinF), Basic(BasicOpCode::CosF), Basic(BasicOpCode::ExpF)],
            BasicOpCode::ExpF => &[Basic(BasicOpCode::ExpF), Basic(BasicOpCode::SinF), Basic(BasicOpCode::CosF), Basic(BasicOpCode::SqrF), Basic(BasicOpCode::LnF)],
            BasicOpCode::SqrtF => &[Basic(BasicOpCode::SqrtF), Basic(BasicOpCode::SqrF)],
            BasicOpCode::SqrF => &[Basic(BasicOpCode::SqrF), Basic(BasicOpCode::SqrtF)],
            BasicOpCode::LnF => &[Basic(BasicOpCode::LnF), Basic(BasicOpCode::ExpF)],
            _ => &[],
        }
    }
}