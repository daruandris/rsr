use wide::{f32x4, CmpLt};
use crate::domain::universal::UniversalOp;

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

// --- KONSTANS FOLDING DELEGÁLÁS (Simplify számára) ---

pub fn fold_unary_const(op: UniversalOp, val: f32) -> Option<f32> {
    match op {
        UniversalOp::SinF => Some(val.sin()),
        UniversalOp::CosF => Some(val.cos()),
        UniversalOp::ExpF => Some(val.exp()),
        UniversalOp::SqrF => Some(val * val),
        UniversalOp::SqrtF => Some(val.abs().sqrt()),
        UniversalOp::LnF => Some((val.abs() + 1e-9).ln()),
        _ => None,
    }
}

pub fn fold_binary_const(op: UniversalOp, a: f32, b: f32) -> Option<f32> {
    match op {
        UniversalOp::AddF => Some(a + b),
        UniversalOp::SubF => Some(a - b),
        UniversalOp::MulF => Some(a * b),
        UniversalOp::DivF => if b.abs() > 1e-9 { Some(a / b) } else { None },
        _ => None,
    }
}