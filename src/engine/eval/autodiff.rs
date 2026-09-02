//! Forward-mode automatic differentiation using Dual numbers.
//!
//! Dual numbers simultaneously compute both the value and the derivative of an expression.
//! This is heavily leveraged by continuous optimizers (like L-BFGS) to calculate exact
//! gradients during constant optimization.
#![allow(unsafe_op_in_unsafe_fn)]
use std::ops::{Add, Div, Mul, Sub};
use wide::{CmpLt, f32x8};

/// Represents a dual number for forward-mode AD using SIMD vectors.
#[derive(Clone, Copy, Debug)]
pub struct DualSimd {
    pub val: f32x8,
    pub grad: f32x8,
}

impl DualSimd {
    #[inline(always)]
    pub fn new(val: f32x8, grad: f32x8) -> Self {
        Self { val, grad }
    }

    #[inline(always)]
    pub fn constant(val: f32x8) -> Self {
        Self {
            val,
            grad: f32x8::splat(0.0),
        }
    }

    #[inline(always)]
    pub fn sin(self) -> Self {
        Self {
            val: self.val.sin(),
            grad: self.grad * self.val.cos(),
        }
    }

    #[inline(always)]
    pub fn cos(self) -> Self {
        Self {
            val: self.val.cos(),
            grad: self.grad * (-self.val.sin()),
        }
    }

    #[inline(always)]
    pub fn exp(self) -> Self {
        let e = self.val.exp();
        Self {
            val: e,
            grad: self.grad * e,
        }
    }

    #[inline(always)]
    pub fn sqr(self) -> Self {
        Self {
            val: self.val * self.val,
            grad: f32x8::splat(2.0) * self.val * self.grad,
        }
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        Self {
            val: s,
            grad: self.grad / (f32x8::splat(2.0) * s),
        }
    }

    #[inline(always)]
    pub fn ln(self) -> Self {
        Self {
            val: self.val.ln(),
            grad: self.grad / self.val,
        }
    }

    #[inline(always)]
    pub fn cube(self) -> Self {
        Self {
            val: self.val * self.val * self.val,
            grad: f32x8::splat(3.0) * self.val * self.val * self.grad,
        }
    }
}

impl Add for DualSimd {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            grad: self.grad + rhs.grad,
        }
    }
}

impl Sub for DualSimd {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            grad: self.grad - rhs.grad,
        }
    }
}

impl Mul for DualSimd {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            val: self.val * rhs.val,
            grad: (self.val * rhs.grad) + (self.grad * rhs.val),
        }
    }
}

impl Div for DualSimd {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self {
            val: self.val / rhs.val,
            grad: ((self.grad * rhs.val) - (self.val * rhs.grad)) / (rhs.val * rhs.val),
        }
    }
}

#[inline(always)]
pub fn dual_dot_v3(a: &[DualSimd; 3], b: &[DualSimd; 3]) -> DualSimd {
    let val = (a[0].val * b[0].val) + (a[1].val * b[1].val) + (a[2].val * b[2].val);
    let grad = (a[0].val * b[0].grad + a[0].grad * b[0].val)
        + (a[1].val * b[1].grad + a[1].grad * b[1].val)
        + (a[2].val * b[2].grad + a[2].grad * b[2].val);
    DualSimd { val, grad }
}

#[inline(always)]
pub fn dual_norm_v3(v: &[DualSimd; 3]) -> DualSimd {
    let dot_p = (v[0].val * v[0].val) + (v[1].val * v[1].val) + (v[2].val * v[2].val);
    let norm_p = dot_p.sqrt();

    let safe_norm = norm_p
        .simd_lt(f32x8::splat(1e-9))
        .blend(f32x8::splat(1.0), norm_p);
    let dot_pd = (v[0].val * v[0].grad) + (v[1].val * v[1].grad) + (v[2].val * v[2].grad);
    let grad = dot_pd / safe_norm;

    DualSimd { val: norm_p, grad }
}

#[inline(always)]
pub fn dual_cross_v3(a: &[DualSimd; 3], b: &[DualSimd; 3]) -> [DualSimd; 3] {
    [
        DualSimd {
            val: a[1].val * b[2].val - a[2].val * b[1].val,
            grad: (a[1].val * b[2].grad + a[1].grad * b[2].val)
                - (a[2].val * b[1].grad + a[2].grad * b[1].val),
        },
        DualSimd {
            val: a[2].val * b[0].val - a[0].val * b[2].val,
            grad: (a[2].val * b[0].grad + a[2].grad * b[0].val)
                - (a[0].val * b[2].grad + a[0].grad * b[2].val),
        },
        DualSimd {
            val: a[0].val * b[1].val - a[1].val * b[0].val,
            grad: (a[0].val * b[1].grad + a[0].grad * b[1].val)
                - (a[1].val * b[0].grad + a[1].grad * b[0].val),
        },
    ]
}

#[inline(always)]
pub fn dual_mul_m2(a: &[DualSimd; 4], b: &[DualSimd; 4]) -> [DualSimd; 4] {
    let mut out = [DualSimd::constant(f32x8::splat(0.0)); 4];
    out[0].val = a[0].val * b[0].val + a[2].val * b[1].val;
    out[1].val = a[1].val * b[0].val + a[3].val * b[1].val;
    out[2].val = a[0].val * b[2].val + a[2].val * b[3].val;
    out[3].val = a[1].val * b[2].val + a[3].val * b[3].val;
    out[0].grad = (a[0].val * b[0].grad + a[2].val * b[1].grad)
        + (a[0].grad * b[0].val + a[2].grad * b[1].val);
    out[1].grad = (a[1].val * b[0].grad + a[3].val * b[1].grad)
        + (a[1].grad * b[0].val + a[3].grad * b[1].val);
    out[2].grad = (a[0].val * b[2].grad + a[2].val * b[3].grad)
        + (a[0].grad * b[2].val + a[2].grad * b[3].val);
    out[3].grad = (a[1].val * b[2].grad + a[3].val * b[3].grad)
        + (a[1].grad * b[2].val + a[3].grad * b[3].val);
    out
}

#[inline(always)]
pub fn dual_inverse_m2(m: &[DualSimd; 4]) -> [DualSimd; 4] {
    let det_p = (m[0].val * m[3].val) - (m[1].val * m[2].val);
    let is_singular = det_p.abs().simd_lt(f32x8::splat(1e-9));
    let safe_det = is_singular.blend(f32x8::splat(1.0), det_p);
    let inv_d = f32x8::splat(1.0) / safe_det;

    let mut inv_p = [f32x8::splat(0.0); 4];
    inv_p[0] = is_singular.blend(f32x8::splat(1.0), m[3].val * inv_d);
    inv_p[1] = is_singular.blend(f32x8::splat(0.0), -m[1].val * inv_d);
    inv_p[2] = is_singular.blend(f32x8::splat(0.0), -m[2].val * inv_d);
    inv_p[3] = is_singular.blend(f32x8::splat(1.0), m[0].val * inv_d);

    let mut temp = [f32x8::splat(0.0); 4];
    temp[0] = inv_p[0] * m[0].grad + inv_p[2] * m[1].grad;
    temp[1] = inv_p[1] * m[0].grad + inv_p[3] * m[1].grad;
    temp[2] = inv_p[0] * m[2].grad + inv_p[2] * m[3].grad;
    temp[3] = inv_p[1] * m[2].grad + inv_p[3] * m[3].grad;

    let mut out = [DualSimd::constant(f32x8::splat(0.0)); 4];
    out[0].val = inv_p[0];
    out[1].val = inv_p[1];
    out[2].val = inv_p[2];
    out[3].val = inv_p[3];
    out[0].grad = -(temp[0] * inv_p[0] + temp[2] * inv_p[1]);
    out[1].grad = -(temp[1] * inv_p[0] + temp[3] * inv_p[1]);
    out[2].grad = -(temp[0] * inv_p[2] + temp[2] * inv_p[3]);
    out[3].grad = -(temp[1] * inv_p[2] + temp[3] * inv_p[3]);
    out
}

// ---------------- Dual SIMD in-place Evaluation Functions ----------------

/// # Safety
/// The caller must ensure that `*sp_f >= 2` to prevent underflow, and that all memory accesses
/// via `get_unchecked` remain strictly within the bounds of `stack_f` and `stack_v2` (0..32).
#[inline(always)]
pub unsafe fn eval_make_dual_vec2(
    sp_f: &mut usize,
    stack_f: &[DualSimd; 32],
    sp_v2: &mut usize,
    stack_v2: &mut [[DualSimd; 2]; 32],
) {
    *sp_f -= 2;
    *stack_v2.get_unchecked_mut(*sp_v2) = [
        *stack_f.get_unchecked(*sp_f),
        *stack_f.get_unchecked(*sp_f + 1),
    ];
    *sp_v2 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_f >= 3` to prevent underflow, and that all memory accesses
/// via `get_unchecked` remain strictly within the bounds of `stack_f` and `stack_v3` (0..32).
#[inline(always)]
pub unsafe fn eval_make_dual_vec3(
    sp_f: &mut usize,
    stack_f: &[DualSimd; 32],
    sp_v3: &mut usize,
    stack_v3: &mut [[DualSimd; 3]; 32],
) {
    *sp_f -= 3;
    *stack_v3.get_unchecked_mut(*sp_v3) = [
        *stack_f.get_unchecked(*sp_f),
        *stack_f.get_unchecked(*sp_f + 1),
        *stack_f.get_unchecked(*sp_f + 2),
    ];
    *sp_v3 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_f >= 2` to prevent underflow and out-of-bounds access
#[inline(always)]
pub unsafe fn eval_add_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    *sp_f -= 2;
    let a = *stack_f.get_unchecked(*sp_f);
    let b = *stack_f.get_unchecked(*sp_f + 1);
    *stack_f.get_unchecked_mut(*sp_f) = a + b;
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_f >= 2` to prevent underflow and out-of-bounds ac
#[inline(always)]
pub unsafe fn eval_sub_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    *sp_f -= 2;
    let a = *stack_f.get_unchecked(*sp_f);
    let b = *stack_f.get_unchecked(*sp_f + 1);
    *stack_f.get_unchecked_mut(*sp_f) = a - b;
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_f >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_mul_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    *sp_f -= 2;
    let a = *stack_f.get_unchecked(*sp_f);
    let b = *stack_f.get_unchecked(*sp_f + 1);
    *stack_f.get_unchecked_mut(*sp_f) = a * b;
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_f >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_div_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    *sp_f -= 2;
    let a = *stack_f.get_unchecked(*sp_f);
    let b = *stack_f.get_unchecked(*sp_f + 1);
    *stack_f.get_unchecked_mut(*sp_f) = a / b;
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_f >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_sin_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).sin();
}

/// # Safety
/// The caller must ensure that `*sp_f >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_cos_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).cos();
}

/// # Safety
/// The caller must ensure that `*sp_f >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_exp_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).exp();
}

/// # Safety
/// The caller must ensure that `*sp_f >= 1` to prevent underflow and out-of-bounds ac
#[inline(always)]
pub unsafe fn eval_sqr_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).sqr();
}

/// # Safety
/// The caller must ensure that `*sp_f >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_sqrt_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).sqrt();
}

/// # Safety
/// The caller must ensure that `*sp_f >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_ln_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).ln();
}

/// # Safety
/// The caller must ensure that `*sp_v2 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_get_x_dual_v2(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_v2: &mut usize,
    stack_v2: &[[DualSimd; 2]; 32],
) {
    *sp_v2 -= 1;
    *stack_f.get_unchecked_mut(*sp_f) = stack_v2.get_unchecked(*sp_v2)[0];
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v2 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_get_y_dual_v2(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_v2: &mut usize,
    stack_v2: &[[DualSimd; 2]; 32],
) {
    *sp_v2 -= 1;
    *stack_f.get_unchecked_mut(*sp_f) = stack_v2.get_unchecked(*sp_v2)[1];
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v3 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_get_x_dual_v3(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_v3: &mut usize,
    stack_v3: &[[DualSimd; 3]; 32],
) {
    *sp_v3 -= 1;
    *stack_f.get_unchecked_mut(*sp_f) = stack_v3.get_unchecked(*sp_v3)[0];
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v3 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_get_y_dual_v3(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_v3: &mut usize,
    stack_v3: &[[DualSimd; 3]; 32],
) {
    *sp_v3 -= 1;
    *stack_f.get_unchecked_mut(*sp_f) = stack_v3.get_unchecked(*sp_v3)[1];
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v3 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_get_z_dual_v3(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_v3: &mut usize,
    stack_v3: &[[DualSimd; 3]; 32],
) {
    *sp_v3 -= 1;
    *stack_f.get_unchecked_mut(*sp_f) = stack_v3.get_unchecked(*sp_v3)[2];
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v2 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_add_dual_v2(sp_v2: &mut usize, stack_v2: &mut [[DualSimd; 2]; 32]) {
    *sp_v2 -= 2;
    let a = *stack_v2.get_unchecked(*sp_v2);
    let b = *stack_v2.get_unchecked(*sp_v2 + 1);
    *stack_v2.get_unchecked_mut(*sp_v2) = [a[0] + b[0], a[1] + b[1]];
    *sp_v2 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v2 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_sub_dual_v2(sp_v2: &mut usize, stack_v2: &mut [[DualSimd; 2]; 32]) {
    *sp_v2 -= 2;
    let a = *stack_v2.get_unchecked(*sp_v2);
    let b = *stack_v2.get_unchecked(*sp_v2 + 1);
    *stack_v2.get_unchecked_mut(*sp_v2) = [a[0] - b[0], a[1] - b[1]];
    *sp_v2 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_f >= 1` and `*sp_v2 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_scale_dual_v2(
    sp_f: &mut usize,
    stack_f: &[DualSimd; 32],
    sp_v2: &mut usize,
    stack_v2: &mut [[DualSimd; 2]; 32],
) {
    *sp_f -= 1;
    *sp_v2 -= 1;
    let s = *stack_f.get_unchecked(*sp_f);
    let v = *stack_v2.get_unchecked(*sp_v2);
    *stack_v2.get_unchecked_mut(*sp_v2) = [v[0] * s, v[1] * s];
    *sp_v2 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v2 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_dot_dual_v2(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_v2: &mut usize,
    stack_v2: &[[DualSimd; 2]; 32],
) {
    *sp_v2 -= 2;
    let a = *stack_v2.get_unchecked(*sp_v2);
    let b = *stack_v2.get_unchecked(*sp_v2 + 1);
    *stack_f.get_unchecked_mut(*sp_f) = a[0] * b[0] + a[1] * b[1];
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v2 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_norm_dual_v2(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_v2: &mut usize,
    stack_v2: &[[DualSimd; 2]; 32],
) {
    let idx = *sp_v2 - 1;
    let v = *stack_v2.get_unchecked(idx);
    *stack_f.get_unchecked_mut(*sp_f) = (v[0] * v[0] + v[1] * v[1]).sqrt();
    *sp_f += 1;
    *sp_v2 -= 1;
}

/// # Safety
/// The caller must ensure that `*sp_v3 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_add_dual_v3(sp_v3: &mut usize, stack_v3: &mut [[DualSimd; 3]; 32]) {
    *sp_v3 -= 2;
    let a = *stack_v3.get_unchecked(*sp_v3);
    let b = *stack_v3.get_unchecked(*sp_v3 + 1);
    *stack_v3.get_unchecked_mut(*sp_v3) = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    *sp_v3 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v3 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_sub_dual_v3(sp_v3: &mut usize, stack_v3: &mut [[DualSimd; 3]; 32]) {
    *sp_v3 -= 2;
    let a = *stack_v3.get_unchecked(*sp_v3);
    let b = *stack_v3.get_unchecked(*sp_v3 + 1);
    *stack_v3.get_unchecked_mut(*sp_v3) = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    *sp_v3 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_f >= 1` and `*sp_v3 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_scale_dual_v3(
    sp_f: &mut usize,
    stack_f: &[DualSimd; 32],
    sp_v3: &mut usize,
    stack_v3: &mut [[DualSimd; 3]; 32],
) {
    *sp_f -= 1;
    *sp_v3 -= 1;
    let s = *stack_f.get_unchecked(*sp_f);
    let v = *stack_v3.get_unchecked(*sp_v3);
    *stack_v3.get_unchecked_mut(*sp_v3) = [v[0] * s, v[1] * s, v[2] * s];
    *sp_v3 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v3 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_dot_dual_v3(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_v3: &mut usize,
    stack_v3: &[[DualSimd; 3]; 32],
) {
    *sp_v3 -= 2;
    let a = *stack_v3.get_unchecked(*sp_v3);
    let b = *stack_v3.get_unchecked(*sp_v3 + 1);
    *stack_f.get_unchecked_mut(*sp_f) = dual_dot_v3(&a, &b);
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v3 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_norm_dual_v3(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_v3: &mut usize,
    stack_v3: &[[DualSimd; 3]; 32],
) {
    let idx = *sp_v3 - 1;
    let v = *stack_v3.get_unchecked(idx);
    *stack_f.get_unchecked_mut(*sp_f) = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    *sp_f += 1;
    *sp_v3 -= 1;
}

/// # Safety
/// The caller must ensure that `*sp_v3 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_cross_dual_v3(sp_v3: &mut usize, stack_v3: &mut [[DualSimd; 3]; 32]) {
    *sp_v3 -= 2;
    let a = *stack_v3.get_unchecked(*sp_v3);
    let b = *stack_v3.get_unchecked(*sp_v3 + 1);
    *stack_v3.get_unchecked_mut(*sp_v3) = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    *sp_v3 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_v2 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_make_dual_mat2(
    sp_v2: &mut usize,
    stack_v2: &[[DualSimd; 2]; 32],
    sp_m2: &mut usize,
    stack_m2: &mut [[DualSimd; 4]; 32],
) {
    *sp_v2 -= 2;
    let c0 = *stack_v2.get_unchecked(*sp_v2);
    let c1 = *stack_v2.get_unchecked(*sp_v2 + 1);
    *stack_m2.get_unchecked_mut(*sp_m2) = [c0[0], c0[1], c1[0], c1[1]];
    *sp_m2 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m2 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_add_dual_m2(sp_m2: &mut usize, stack_m2: &mut [[DualSimd; 4]; 32]) {
    *sp_m2 -= 2;
    let a = *stack_m2.get_unchecked(*sp_m2);
    let b = *stack_m2.get_unchecked(*sp_m2 + 1);
    *stack_m2.get_unchecked_mut(*sp_m2) = [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    *sp_m2 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m2 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_sub_dual_m2(sp_m2: &mut usize, stack_m2: &mut [[DualSimd; 4]; 32]) {
    *sp_m2 -= 2;
    let a = *stack_m2.get_unchecked(*sp_m2);
    let b = *stack_m2.get_unchecked(*sp_m2 + 1);
    *stack_m2.get_unchecked_mut(*sp_m2) = [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    *sp_m2 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_f >= 1` and `*sp_m2 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_scale_dual_m2(
    sp_f: &mut usize,
    stack_f: &[DualSimd; 32],
    sp_m2: &mut usize,
    stack_m2: &mut [[DualSimd; 4]; 32],
) {
    *sp_f -= 1;
    *sp_m2 -= 1;
    let s = *stack_f.get_unchecked(*sp_f);
    let m = *stack_m2.get_unchecked(*sp_m2);
    *stack_m2.get_unchecked_mut(*sp_m2) = [m[0] * s, m[1] * s, m[2] * s, m[3] * s];
    *sp_m2 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m2 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_mul_dual_m2(sp_m2: &mut usize, stack_m2: &mut [[DualSimd; 4]; 32]) {
    *sp_m2 -= 2;
    let a = *stack_m2.get_unchecked(*sp_m2);
    let b = *stack_m2.get_unchecked(*sp_m2 + 1);
    *stack_m2.get_unchecked_mut(*sp_m2) = [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
    ];
    *sp_m2 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m2 >= 1` and `*sp_v2 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_mul_dual_m2v2(
    sp_m2: &mut usize,
    stack_m2: &[[DualSimd; 4]; 32],
    sp_v2: &mut usize,
    stack_v2: &mut [[DualSimd; 2]; 32],
) {
    *sp_m2 -= 1;
    *sp_v2 -= 1;
    let m = *stack_m2.get_unchecked(*sp_m2);
    let v = *stack_v2.get_unchecked(*sp_v2);
    *stack_v2.get_unchecked_mut(*sp_v2) = [m[0] * v[0] + m[2] * v[1], m[1] * v[0] + m[3] * v[1]];
    *sp_v2 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m2 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_det_dual_m2(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_m2: &mut usize,
    stack_m2: &[[DualSimd; 4]; 32],
) {
    *sp_m2 -= 1;
    let m = *stack_m2.get_unchecked(*sp_m2);
    *stack_f.get_unchecked_mut(*sp_f) = m[0] * m[3] - m[1] * m[2];
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m2 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_trace_dual_m2(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_m2: &mut usize,
    stack_m2: &[[DualSimd; 4]; 32],
) {
    *sp_m2 -= 1;
    let m = *stack_m2.get_unchecked(*sp_m2);
    *stack_f.get_unchecked_mut(*sp_f) = m[0] + m[3];
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m2 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_transpose_dual_m2(sp_m2: &mut usize, stack_m2: &mut [[DualSimd; 4]; 32]) {
    let idx = *sp_m2 - 1;
    let m = *stack_m2.get_unchecked(idx);
    *stack_m2.get_unchecked_mut(idx) = [m[0], m[2], m[1], m[3]];
}

/// # Safety
/// The caller must ensure that `*sp_v3 >= 3` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_make_dual_mat3(
    sp_v3: &mut usize,
    stack_v3: &[[DualSimd; 3]; 32],
    sp_m3: &mut usize,
    stack_m3: &mut [[DualSimd; 9]; 32],
) {
    *sp_v3 -= 3;
    let c0 = *stack_v3.get_unchecked(*sp_v3);
    let c1 = *stack_v3.get_unchecked(*sp_v3 + 1);
    let c2 = *stack_v3.get_unchecked(*sp_v3 + 2);
    *stack_m3.get_unchecked_mut(*sp_m3) = [
        c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2],
    ];
    *sp_m3 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m3 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_add_dual_m3(sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) {
    *sp_m3 -= 2;
    let a = *stack_m3.get_unchecked(*sp_m3);
    let b = *stack_m3.get_unchecked(*sp_m3 + 1);
    *stack_m3.get_unchecked_mut(*sp_m3) = [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
        a[5] + b[5],
        a[6] + b[6],
        a[7] + b[7],
        a[8] + b[8],
    ];
    *sp_m3 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m3 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_sub_dual_m3(sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) {
    *sp_m3 -= 2;
    let a = *stack_m3.get_unchecked(*sp_m3);
    let b = *stack_m3.get_unchecked(*sp_m3 + 1);
    *stack_m3.get_unchecked_mut(*sp_m3) = [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
        a[3] - b[3],
        a[4] - b[4],
        a[5] - b[5],
        a[6] - b[6],
        a[7] - b[7],
        a[8] - b[8],
    ];
    *sp_m3 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_f >= 1` and `*sp_m3 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_scale_dual_m3(
    sp_f: &mut usize,
    stack_f: &[DualSimd; 32],
    sp_m3: &mut usize,
    stack_m3: &mut [[DualSimd; 9]; 32],
) {
    *sp_f -= 1;
    *sp_m3 -= 1;
    let s = *stack_f.get_unchecked(*sp_f);
    let m = *stack_m3.get_unchecked(*sp_m3);
    *stack_m3.get_unchecked_mut(*sp_m3) = [
        m[0] * s,
        m[1] * s,
        m[2] * s,
        m[3] * s,
        m[4] * s,
        m[5] * s,
        m[6] * s,
        m[7] * s,
        m[8] * s,
    ];
    *sp_m3 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m3 >= 2` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_mul_dual_m3(sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) {
    *sp_m3 -= 2;
    let a = *stack_m3.get_unchecked(*sp_m3);
    let b = *stack_m3.get_unchecked(*sp_m3 + 1);
    *stack_m3.get_unchecked_mut(*sp_m3) = [
        a[0] * b[0] + a[3] * b[1] + a[6] * b[2],
        a[1] * b[0] + a[4] * b[1] + a[7] * b[2],
        a[2] * b[0] + a[5] * b[1] + a[8] * b[2],
        a[0] * b[3] + a[3] * b[4] + a[6] * b[5],
        a[1] * b[3] + a[4] * b[4] + a[7] * b[5],
        a[2] * b[3] + a[5] * b[4] + a[8] * b[5],
        a[0] * b[6] + a[3] * b[7] + a[6] * b[8],
        a[1] * b[6] + a[4] * b[7] + a[7] * b[8],
        a[2] * b[6] + a[5] * b[7] + a[8] * b[8],
    ];
    *sp_m3 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m3 >= 1` and `*sp_v3 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_mul_dual_m3v3(
    sp_m3: &mut usize,
    stack_m3: &[[DualSimd; 9]; 32],
    sp_v3: &mut usize,
    stack_v3: &mut [[DualSimd; 3]; 32],
) {
    *sp_m3 -= 1;
    *sp_v3 -= 1;
    let m = *stack_m3.get_unchecked(*sp_m3);
    let v = *stack_v3.get_unchecked(*sp_v3);
    *stack_v3.get_unchecked_mut(*sp_v3) = [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ];
    *sp_v3 += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m3 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_det_dual_m3(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_m3: &mut usize,
    stack_m3: &[[DualSimd; 9]; 32],
) {
    *sp_m3 -= 1;
    let m = *stack_m3.get_unchecked(*sp_m3);
    *stack_f.get_unchecked_mut(*sp_f) = m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[3] * (m[1] * m[8] - m[2] * m[7])
        + m[6] * (m[1] * m[5] - m[2] * m[4]);
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m3 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_trace_dual_m3(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_m3: &mut usize,
    stack_m3: &[[DualSimd; 9]; 32],
) {
    *sp_m3 -= 1;
    let m = *stack_m3.get_unchecked(*sp_m3);
    *stack_f.get_unchecked_mut(*sp_f) = m[0] + m[4] + m[8];
    *sp_f += 1;
}

/// # Safety
/// The caller must ensure that `*sp_m3 >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_transpose_dual_m3(sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) {
    let idx = *sp_m3 - 1;
    let m = *stack_m3.get_unchecked(idx);
    *stack_m3.get_unchecked_mut(idx) = [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}

/// # Safety
/// The caller must ensure that `*sp_f >= 1` to prevent underflow and out-of-bounds access.
#[inline(always)]
pub unsafe fn eval_cube_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).cube();
}
