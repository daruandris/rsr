#![allow(unsafe_op_in_unsafe_fn)]
use wide::f32x8;

#[inline(always)]
pub unsafe fn eval_add_f(sp_f: &mut usize, stack_f: &mut [f32x8; 32]) {
    *sp_f -= 2;
    *stack_f.get_unchecked_mut(*sp_f) =
        *stack_f.get_unchecked(*sp_f) + *stack_f.get_unchecked(*sp_f + 1);
    *sp_f += 1;
}
#[inline(always)]
pub unsafe fn eval_sub_f(sp_f: &mut usize, stack_f: &mut [f32x8; 32]) {
    *sp_f -= 2;
    *stack_f.get_unchecked_mut(*sp_f) =
        *stack_f.get_unchecked(*sp_f) - *stack_f.get_unchecked(*sp_f + 1);
    *sp_f += 1;
}
#[inline(always)]
pub unsafe fn eval_mul_f(sp_f: &mut usize, stack_f: &mut [f32x8; 32]) {
    *sp_f -= 2;
    *stack_f.get_unchecked_mut(*sp_f) =
        *stack_f.get_unchecked(*sp_f) * *stack_f.get_unchecked(*sp_f + 1);
    *sp_f += 1;
}
#[inline(always)]
pub unsafe fn eval_div_f(sp_f: &mut usize, stack_f: &mut [f32x8; 32]) {
    *sp_f -= 2;
    let a = *stack_f.get_unchecked(*sp_f);
    let b = *stack_f.get_unchecked(*sp_f + 1);
    *stack_f.get_unchecked_mut(*sp_f) = a / b;
    *sp_f += 1;
}
#[inline(always)]
pub unsafe fn eval_sin_f(sp_f: &mut usize, stack_f: &mut [f32x8; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).sin();
}
#[inline(always)]
pub unsafe fn eval_cos_f(sp_f: &mut usize, stack_f: &mut [f32x8; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).cos();
}
#[inline(always)]
pub unsafe fn eval_exp_f(sp_f: &mut usize, stack_f: &mut [f32x8; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).exp();
}
#[inline(always)]
pub unsafe fn eval_sqr_f(sp_f: &mut usize, stack_f: &mut [f32x8; 32]) {
    let idx = *sp_f - 1;
    let a = *stack_f.get_unchecked(idx);
    *stack_f.get_unchecked_mut(idx) = a * a;
}
#[inline(always)]
pub unsafe fn eval_sqrt_f(sp_f: &mut usize, stack_f: &mut [f32x8; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).sqrt();
}
#[inline(always)]
pub unsafe fn eval_ln_f(sp_f: &mut usize, stack_f: &mut [f32x8; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).ln();
}
