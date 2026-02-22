#![allow(unsafe_op_in_unsafe_fn)]

use wide::{f32x4, CmpLt};
use crate::domain::universal::{UniversalOp, UniversalScalar, SimplifyAction};
// --- SIMD KIÉRTÉKELÉS ---

#[inline(always)] pub unsafe fn eval_add_f(sp_f: &mut usize, stack_f: &mut [f32x4; 32]) {
    *sp_f -= 2; *stack_f.get_unchecked_mut(*sp_f) = *stack_f.get_unchecked(*sp_f) + *stack_f.get_unchecked(*sp_f + 1); *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_sub_f(sp_f: &mut usize, stack_f: &mut [f32x4; 32]) {
    *sp_f -= 2; *stack_f.get_unchecked_mut(*sp_f) = *stack_f.get_unchecked(*sp_f) - *stack_f.get_unchecked(*sp_f + 1); *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_mul_f(sp_f: &mut usize, stack_f: &mut [f32x4; 32]) {
    *sp_f -= 2; *stack_f.get_unchecked_mut(*sp_f) = *stack_f.get_unchecked(*sp_f) * *stack_f.get_unchecked(*sp_f + 1); *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_div_f(sp_f: &mut usize, stack_f: &mut [f32x4; 32]) {
    *sp_f -= 2; let a = *stack_f.get_unchecked(*sp_f); let b = *stack_f.get_unchecked(*sp_f + 1);
    let safe_b = b.abs().simd_lt(f32x4::splat(1e-9)).blend(f32x4::splat(1.0), b);
    *stack_f.get_unchecked_mut(*sp_f) = a / safe_b; *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_sin_f(sp_f: &mut usize, stack_f: &mut [f32x4; 32]) {
    let idx = *sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).sin();
}
#[inline(always)] pub unsafe fn eval_cos_f(sp_f: &mut usize, stack_f: &mut [f32x4; 32]) {
    let idx = *sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).cos();
}
#[inline(always)] pub unsafe fn eval_exp_f(sp_f: &mut usize, stack_f: &mut [f32x4; 32]) {
    let idx = *sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).exp();
}
#[inline(always)] pub unsafe fn eval_sqr_f(sp_f: &mut usize, stack_f: &mut [f32x4; 32]) {
    let idx = *sp_f - 1; let a = *stack_f.get_unchecked(idx); *stack_f.get_unchecked_mut(idx) = a * a;
}
#[inline(always)] pub unsafe fn eval_sqrt_f(sp_f: &mut usize, stack_f: &mut [f32x4; 32]) {
    let idx = *sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).abs().sqrt();
}
#[inline(always)] pub unsafe fn eval_ln_f(sp_f: &mut usize, stack_f: &mut [f32x4; 32]) {
    let idx = *sp_f - 1; *stack_f.get_unchecked_mut(idx) = (stack_f.get_unchecked(idx).abs() + f32x4::splat(1e-9)).ln();
}

// --- FORMÁZÁS ---

pub fn format_op(op: UniversalOp, args: &[String]) -> Option<String> {
    match op {
        UniversalOp::AddF => Some(format!("({} + {})", args[0], args[1])),
        UniversalOp::SubF => Some(format!("({} - {})", args[0], args[1])),
        UniversalOp::MulF => Some(format!("({} * {})", args[0], args[1])),
        UniversalOp::DivF => Some(format!("({} / {})", args[0], args[1])),
        UniversalOp::SinF => Some(format!("sin({})", args[0])),
        UniversalOp::CosF => Some(format!("cos({})", args[0])),
        UniversalOp::ExpF => Some(format!("exp({})", args[0])),
        UniversalOp::SqrF => Some(format!("({})^2", args[0])),
        UniversalOp::SqrtF => Some(format!("sqrt(|{}|)", args[0])),
        UniversalOp::LnF => Some(format!("ln(|{}|)", args[0])),
        _ => None,
    }
}

// --- BASIC ALGEBRAIC SIMPLIFICATION ---
pub fn try_simplify(op: UniversalOp, const_vals: &[Option<UniversalScalar>], args_equal: bool) -> SimplifyAction {
    // 1. Teljes konstans kiértékelés (Folding)
    let all_const = const_vals.iter().all(|c| c.is_some());
    if all_const {
        let vals: Vec<UniversalScalar> = const_vals.iter().map(|c| c.unwrap()).collect();
        if let Some(folded) = fold_constants(op, &vals) {
            if let UniversalScalar::Float(f) = folded {
                if f.is_finite() { return SimplifyAction::ReplaceWithConstant(folded); }
            } else { return SimplifyAction::ReplaceWithConstant(folded); }
        }
    }

    // 2. Szabály alapú egyszerűsítések (Algebrai identitások)
    if const_vals.len() == 2 {
        let a_is_zero = const_vals[0].as_ref().map_or(false, |c| c.is_zero());
        let b_is_zero = const_vals[1].as_ref().map_or(false, |c| c.is_zero());
        let a_is_one = const_vals[0].as_ref().map_or(false, |c| c.is_one());
        let b_is_one = const_vals[1].as_ref().map_or(false, |c| c.is_one());

        match op {
            UniversalOp::AddF => {
                if b_is_zero { return SimplifyAction::KeepArg(0); }
                if a_is_zero { return SimplifyAction::KeepArg(1); }
            },
            UniversalOp::SubF => {
                if b_is_zero { return SimplifyAction::KeepArg(0); }
                if args_equal { return SimplifyAction::ReplaceWithConstant(UniversalScalar::Float(0.0)); }
            },
            UniversalOp::MulF => {
                if b_is_one { return SimplifyAction::KeepArg(0); }
                if a_is_one { return SimplifyAction::KeepArg(1); }
                if a_is_zero || b_is_zero { return SimplifyAction::ReplaceWithConstant(UniversalScalar::Float(0.0)); }
            },
            UniversalOp::DivF => {
                if b_is_one { return SimplifyAction::KeepArg(0); }
                if a_is_zero && !b_is_zero { return SimplifyAction::ReplaceWithConstant(UniversalScalar::Float(0.0)); }
                if args_equal { return SimplifyAction::ReplaceWithConstant(UniversalScalar::Float(1.0)); }
            },
            _ => {}
        }
    }
    SimplifyAction::None
}

fn fold_constants(op: UniversalOp, args: &[UniversalScalar]) -> Option<UniversalScalar> {
    use UniversalScalar::Float;
    match (op, args) {
        (UniversalOp::SinF, [Float(a)]) => Some(Float(a.sin())),
        (UniversalOp::CosF, [Float(a)]) => Some(Float(a.cos())),
        (UniversalOp::ExpF, [Float(a)]) => Some(Float(a.exp())),
        (UniversalOp::SqrF, [Float(a)]) => Some(Float(a * a)),
        (UniversalOp::SqrtF, [Float(a)]) => Some(Float(a.abs().sqrt())),
        (UniversalOp::LnF, [Float(a)]) => Some(Float((a.abs() + 1e-9).ln())),
        (UniversalOp::AddF, [Float(a), Float(b)]) => Some(Float(a + b)),
        (UniversalOp::SubF, [Float(a), Float(b)]) => Some(Float(a - b)),
        (UniversalOp::MulF, [Float(a), Float(b)]) => Some(Float(a * b)),
        (UniversalOp::DivF, [Float(a), Float(b)]) => if b.abs() > 1e-9 { Some(Float(a / b)) } else { None },
        _ => None,
    }
}