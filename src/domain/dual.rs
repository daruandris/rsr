#![allow(unsafe_op_in_unsafe_fn)]
use std::ops::{Add, Sub, Mul, Div};
use wide::{f32x4, CmpLt};

#[derive(Clone, Copy, Debug)]
pub struct DualSimd {
    pub val: f32x4,   // Primal érték
    pub grad: f32x4,  // Tangent (derivált) érték
}

impl DualSimd {
    #[inline(always)]
    pub fn new(val: f32x4, grad: f32x4) -> Self {
        Self { val, grad }
    }

    #[inline(always)]
    pub fn constant(val: f32x4) -> Self {
        // A konstansok (és bemeneti adatok) deriváltja alapértelmezetten 0
        Self { val, grad: f32x4::splat(0.0) }
    }

    // --- Egyváltozós matematikai függvények (Láncszabály alkalmazása) ---

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
            grad: f32x4::splat(2.0) * self.val * self.grad,
        }
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        // Biztonsági védelem a negatív gyökvonás ellen
        let safe_val = self.val.abs();
        let s = safe_val.sqrt();
        
        // Elkerüljük a nullával osztást a deriváltban
        let safe_s = s.simd_lt(f32x4::splat(1e-9)).blend(f32x4::splat(1.0), s);
        
        Self {
            val: s,
            grad: self.grad / (f32x4::splat(2.0) * safe_s),
        }
    }

    #[inline(always)]
    pub fn ln(self) -> Self {
        // Biztonsági védelem, ahogy a korábbi kódban is csináltad
        let safe_val = self.val.abs() + f32x4::splat(1e-9);
        Self {
            val: safe_val.ln(),
            grad: self.grad / safe_val,
        }
    }
}

// --- Operátor túlterhelések (Operator Overloading) ---

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
        // Zero-div védelem a hányadosszabálynál
        let safe_b = rhs.val.abs().simd_lt(f32x4::splat(1e-9)).blend(f32x4::splat(1.0), rhs.val);
        Self {
            val: self.val / safe_b,
            grad: ((self.grad * rhs.val) - (self.val * rhs.grad)) / (safe_b * safe_b),
        }
    }
}


#[inline(always)]
pub fn dual_dot_v3(a: &[DualSimd; 3], b: &[DualSimd; 3]) -> DualSimd {
    // U_p * V_p
    let val = (a[0].val * b[0].val) + (a[1].val * b[1].val) + (a[2].val * b[2].val);
    // U_p * V_d + U_d * V_p
    let grad = (a[0].val * b[0].grad + a[0].grad * b[0].val) +
               (a[1].val * b[1].grad + a[1].grad * b[1].val) +
               (a[2].val * b[2].grad + a[2].grad * b[2].val);
    DualSimd { val, grad }
}

#[inline(always)]
pub fn dual_norm_v3(v: &[DualSimd; 3]) -> DualSimd {
    let dot_p = (v[0].val * v[0].val) + (v[1].val * v[1].val) + (v[2].val * v[2].val);
    let norm_p = dot_p.sqrt();
    
    // Védelem a nullával osztás ellen
    let safe_norm = norm_p.simd_lt(f32x4::splat(1e-9)).blend(f32x4::splat(1.0), norm_p);
    
    // (V_p * V_d) / ||V_p||
    let dot_pd = (v[0].val * v[0].grad) + (v[1].val * v[1].grad) + (v[2].val * v[2].grad);
    let grad = dot_pd / safe_norm;
    
    DualSimd { val: norm_p, grad }
}

#[inline(always)]
pub fn dual_cross_v3(a: &[DualSimd; 3], b: &[DualSimd; 3]) -> [DualSimd; 3] {
    [
        DualSimd {
            val: a[1].val * b[2].val - a[2].val * b[1].val,
            grad: (a[1].val * b[2].grad + a[1].grad * b[2].val) - (a[2].val * b[1].grad + a[2].grad * b[1].val),
        },
        DualSimd {
            val: a[2].val * b[0].val - a[0].val * b[2].val,
            grad: (a[2].val * b[0].grad + a[2].grad * b[0].val) - (a[0].val * b[2].grad + a[0].grad * b[2].val),
        },
        DualSimd {
            val: a[0].val * b[1].val - a[1].val * b[0].val,
            grad: (a[0].val * b[1].grad + a[0].grad * b[1].val) - (a[1].val * b[0].grad + a[1].grad * b[0].val),
        }
    ]
}

// --- 2x2 MÁTRIX MŰVELETEK ---

#[inline(always)]
pub fn dual_mul_m2(a: &[DualSimd; 4], b: &[DualSimd; 4]) -> [DualSimd; 4] {
    // Mátrixszorzás szabálya: AB = A_p*B_p + (A_p*B_d + A_d*B_p)e
    let mut out = [DualSimd::constant(f32x4::splat(0.0)); 4];
    
    // Indexek: 0: 00, 1: 01, 2: 10, 3: 11
    // Primal (A_p * B_p)
    out[0].val = a[0].val * b[0].val + a[2].val * b[1].val;
    out[1].val = a[1].val * b[0].val + a[3].val * b[1].val;
    out[2].val = a[0].val * b[2].val + a[2].val * b[3].val;
    out[3].val = a[1].val * b[2].val + a[3].val * b[3].val;

    // Tangent (A_p * B_d + A_d * B_p)
    out[0].grad = (a[0].val * b[0].grad + a[2].val * b[1].grad) + (a[0].grad * b[0].val + a[2].grad * b[1].val);
    out[1].grad = (a[1].val * b[0].grad + a[3].val * b[1].grad) + (a[1].grad * b[0].val + a[3].grad * b[1].val);
    out[2].grad = (a[0].val * b[2].grad + a[2].val * b[3].grad) + (a[0].grad * b[2].val + a[2].grad * b[3].val);
    out[3].grad = (a[1].val * b[2].grad + a[3].val * b[3].grad) + (a[1].grad * b[2].val + a[3].grad * b[3].val);

    out
}

#[inline(always)]
pub fn dual_inverse_m2(m: &[DualSimd; 4]) -> [DualSimd; 4] {
    // 1. Kiszámoljuk az A_p inverzét
    let det_p = (m[0].val * m[3].val) - (m[1].val * m[2].val);
    let is_singular = det_p.abs().simd_lt(f32x4::splat(1e-9));
    let safe_det = is_singular.blend(f32x4::splat(1.0), det_p);
    let inv_d = f32x4::splat(1.0) / safe_det;

    let mut inv_p = [f32x4::splat(0.0); 4];
    inv_p[0] = is_singular.blend(f32x4::splat(1.0), m[3].val * inv_d);
    inv_p[1] = is_singular.blend(f32x4::splat(0.0), -m[1].val * inv_d);
    inv_p[2] = is_singular.blend(f32x4::splat(0.0), -m[2].val * inv_d);
    inv_p[3] = is_singular.blend(f32x4::splat(1.0), m[0].val * inv_d);

    // 2. Kiszámoljuk a tangenst: - A_p^{-1} * A_d * A_p^{-1}
    // Először: Temp = A_p^{-1} * A_d
    let mut temp = [f32x4::splat(0.0); 4];
    temp[0] = inv_p[0] * m[0].grad + inv_p[2] * m[1].grad;
    temp[1] = inv_p[1] * m[0].grad + inv_p[3] * m[1].grad;
    temp[2] = inv_p[0] * m[2].grad + inv_p[2] * m[3].grad;
    temp[3] = inv_p[1] * m[2].grad + inv_p[3] * m[3].grad;

    // Majd: Grad = - Temp * A_p^{-1}
    let mut out = [DualSimd::constant(f32x4::splat(0.0)); 4];
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

#[inline(always)] pub unsafe fn eval_make_dual_vec2(sp_f: &mut usize, stack_f: &[DualSimd; 32], sp_v2: &mut usize, stack_v2: &mut [[DualSimd; 2]; 32]) {
    *sp_f -= 2; *stack_v2.get_unchecked_mut(*sp_v2) = [*stack_f.get_unchecked(*sp_f), *stack_f.get_unchecked(*sp_f + 1)]; *sp_v2 += 1;
}
#[inline(always)] pub unsafe fn eval_make_dual_vec3(sp_f: &mut usize, stack_f: &[DualSimd; 32], sp_v3: &mut usize, stack_v3: &mut [[DualSimd; 3]; 32]) {
    *sp_f -= 3; *stack_v3.get_unchecked_mut(*sp_v3) = [*stack_f.get_unchecked(*sp_f), *stack_f.get_unchecked(*sp_f + 1), *stack_f.get_unchecked(*sp_f + 2)]; *sp_v3 += 1;
}

#[inline(always)] pub unsafe fn eval_add_dual_v2(sp_v2: &mut usize, stack_v2: &mut [[DualSimd; 2]; 32]) {
    *sp_v2 -= 2; let a = *stack_v2.get_unchecked(*sp_v2); let b = *stack_v2.get_unchecked(*sp_v2 + 1);
    *stack_v2.get_unchecked_mut(*sp_v2) = [a[0]+b[0], a[1]+b[1]]; *sp_v2 += 1;
}
#[inline(always)] pub unsafe fn eval_sub_dual_v2(sp_v2: &mut usize, stack_v2: &mut [[DualSimd; 2]; 32]) {
    *sp_v2 -= 2; let a = *stack_v2.get_unchecked(*sp_v2); let b = *stack_v2.get_unchecked(*sp_v2 + 1);
    *stack_v2.get_unchecked_mut(*sp_v2) = [a[0]-b[0], a[1]-b[1]]; *sp_v2 += 1;
}
#[inline(always)] pub unsafe fn eval_scale_dual_v2(sp_f: &mut usize, stack_f: &[DualSimd; 32], sp_v2: &mut usize, stack_v2: &mut [[DualSimd; 2]; 32]) {
    *sp_f -= 1; *sp_v2 -= 1; let s = *stack_f.get_unchecked(*sp_f); let v = *stack_v2.get_unchecked(*sp_v2);
    *stack_v2.get_unchecked_mut(*sp_v2) = [v[0]*s, v[1]*s]; *sp_v2 += 1;
}

#[inline(always)] pub unsafe fn eval_sub_dual_v3(sp_v3: &mut usize, stack_v3: &mut [[DualSimd; 3]; 32]) {
    *sp_v3 -= 2; let a = *stack_v3.get_unchecked(*sp_v3); let b = *stack_v3.get_unchecked(*sp_v3 + 1);
    *stack_v3.get_unchecked_mut(*sp_v3) = [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; *sp_v3 += 1;
}

#[inline(always)] pub unsafe fn eval_add_dual_m2(sp_m2: &mut usize, stack_m2: &mut [[DualSimd; 4]; 32]) {
    *sp_m2 -= 2; let a = *stack_m2.get_unchecked(*sp_m2); let b = *stack_m2.get_unchecked(*sp_m2 + 1);
    *stack_m2.get_unchecked_mut(*sp_m2) = [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]]; *sp_m2 += 1;
}
#[inline(always)] pub unsafe fn eval_scale_dual_m2(sp_f: &mut usize, stack_f: &[DualSimd; 32], sp_m2: &mut usize, stack_m2: &mut [[DualSimd; 4]; 32]) {
    *sp_f -= 1; *sp_m2 -= 1; let s = *stack_f.get_unchecked(*sp_f); let m = *stack_m2.get_unchecked(*sp_m2);
    *stack_m2.get_unchecked_mut(*sp_m2) = [m[0]*s, m[1]*s, m[2]*s, m[3]*s]; *sp_m2 += 1;
}
#[inline(always)] pub unsafe fn eval_transpose_dual_m2(sp_m2: &mut usize, stack_m2: &mut [[DualSimd; 4]; 32]) {
    let idx = *sp_m2 - 1; let m = *stack_m2.get_unchecked(idx);
    *stack_m2.get_unchecked_mut(idx) = [m[0], m[2], m[1], m[3]]; // Primal és tangens is egyszerre cserél helyet!
}
#[inline(always)] pub unsafe fn eval_sub_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    *sp_f -= 2; let a = *stack_f.get_unchecked(*sp_f); let b = *stack_f.get_unchecked(*sp_f + 1);
    *stack_f.get_unchecked_mut(*sp_f) = a - b; *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_mul_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    *sp_f -= 2; let a = *stack_f.get_unchecked(*sp_f); let b = *stack_f.get_unchecked(*sp_f + 1);
    *stack_f.get_unchecked_mut(*sp_f) = a * b; *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_div_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    *sp_f -= 2; let a = *stack_f.get_unchecked(*sp_f); let b = *stack_f.get_unchecked(*sp_f + 1);
    *stack_f.get_unchecked_mut(*sp_f) = a / b; *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_cos_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).cos();
}
#[inline(always)] pub unsafe fn eval_exp_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).exp();
}
#[inline(always)] pub unsafe fn eval_sqr_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).sqr();
}
#[inline(always)] pub unsafe fn eval_sqrt_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).sqrt();
}
#[inline(always)] pub unsafe fn eval_ln_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).ln();
}

// --- LINALG DUAL MŰVELETEK (V2, V3, M2, M3) ---
#[inline(always)] pub unsafe fn eval_get_x_dual_v2(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_v2: &mut usize, stack_v2: &[[DualSimd; 2]; 32]) { *sp_v2 -= 1; *stack_f.get_unchecked_mut(*sp_f) = stack_v2.get_unchecked(*sp_v2)[0]; *sp_f += 1; }
#[inline(always)] pub unsafe fn eval_get_y_dual_v2(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_v2: &mut usize, stack_v2: &[[DualSimd; 2]; 32]) { *sp_v2 -= 1; *stack_f.get_unchecked_mut(*sp_f) = stack_v2.get_unchecked(*sp_v2)[1]; *sp_f += 1; }
#[inline(always)] pub unsafe fn eval_get_x_dual_v3(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_v3: &mut usize, stack_v3: &[[DualSimd; 3]; 32]) { *sp_v3 -= 1; *stack_f.get_unchecked_mut(*sp_f) = stack_v3.get_unchecked(*sp_v3)[0]; *sp_f += 1; }
#[inline(always)] pub unsafe fn eval_get_y_dual_v3(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_v3: &mut usize, stack_v3: &[[DualSimd; 3]; 32]) { *sp_v3 -= 1; *stack_f.get_unchecked_mut(*sp_f) = stack_v3.get_unchecked(*sp_v3)[1]; *sp_f += 1; }
#[inline(always)] pub unsafe fn eval_get_z_dual_v3(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_v3: &mut usize, stack_v3: &[[DualSimd; 3]; 32]) { *sp_v3 -= 1; *stack_f.get_unchecked_mut(*sp_f) = stack_v3.get_unchecked(*sp_v3)[2]; *sp_f += 1; }

#[inline(always)] pub unsafe fn eval_dot_dual_v2(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_v2: &mut usize, stack_v2: &[[DualSimd; 2]; 32]) { *sp_v2 -= 2; let a = *stack_v2.get_unchecked(*sp_v2); let b = *stack_v2.get_unchecked(*sp_v2 + 1); *stack_f.get_unchecked_mut(*sp_f) = a[0]*b[0] + a[1]*b[1]; *sp_f += 1; }
#[inline(always)] pub unsafe fn eval_norm_dual_v2(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_v2: &mut usize, stack_v2: &[[DualSimd; 2]; 32]) { let idx = *sp_v2 - 1; let v = *stack_v2.get_unchecked(idx); *stack_f.get_unchecked_mut(*sp_f) = (v[0]*v[0] + v[1]*v[1]).sqrt(); *sp_f += 1; *sp_v2 -= 1; }
#[inline(always)] pub unsafe fn eval_norm_dual_v3(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_v3: &mut usize, stack_v3: &[[DualSimd; 3]; 32]) { let idx = *sp_v3 - 1; let v = *stack_v3.get_unchecked(idx); *stack_f.get_unchecked_mut(*sp_f) = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt(); *sp_f += 1; *sp_v3 -= 1; }
#[inline(always)] pub unsafe fn eval_cross_dual_v3(sp_v3: &mut usize, stack_v3: &mut [[DualSimd; 3]; 32]) { *sp_v3 -= 2; let a = *stack_v3.get_unchecked(*sp_v3); let b = *stack_v3.get_unchecked(*sp_v3 + 1); *stack_v3.get_unchecked_mut(*sp_v3) = [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]; *sp_v3 += 1; }

#[inline(always)] pub unsafe fn eval_make_dual_mat2(sp_v2: &mut usize, stack_v2: &[[DualSimd; 2]; 32], sp_m2: &mut usize, stack_m2: &mut [[DualSimd; 4]; 32]) { *sp_v2 -= 2; let c0 = *stack_v2.get_unchecked(*sp_v2); let c1 = *stack_v2.get_unchecked(*sp_v2 + 1); *stack_m2.get_unchecked_mut(*sp_m2) = [c0[0], c0[1], c1[0], c1[1]]; *sp_m2 += 1; }
#[inline(always)] pub unsafe fn eval_mul_dual_m2(sp_m2: &mut usize, stack_m2: &mut [[DualSimd; 4]; 32]) { *sp_m2 -= 2; let a = *stack_m2.get_unchecked(*sp_m2); let b = *stack_m2.get_unchecked(*sp_m2 + 1); *stack_m2.get_unchecked_mut(*sp_m2) = [a[0]*b[0] + a[2]*b[1], a[1]*b[0] + a[3]*b[1], a[0]*b[2] + a[2]*b[3], a[1]*b[2] + a[3]*b[3]]; *sp_m2 += 1; }
#[inline(always)] pub unsafe fn eval_mul_dual_m2v2(sp_m2: &mut usize, stack_m2: &[[DualSimd; 4]; 32], sp_v2: &mut usize, stack_v2: &mut [[DualSimd; 2]; 32]) { *sp_m2 -= 1; *sp_v2 -= 1; let m = *stack_m2.get_unchecked(*sp_m2); let v = *stack_v2.get_unchecked(*sp_v2); *stack_v2.get_unchecked_mut(*sp_v2) = [m[0]*v[0] + m[2]*v[1], m[1]*v[0] + m[3]*v[1]]; *sp_v2 += 1; }
#[inline(always)] pub unsafe fn eval_det_dual_m2(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_m2: &mut usize, stack_m2: &[[DualSimd; 4]; 32]) { *sp_m2 -= 1; let m = *stack_m2.get_unchecked(*sp_m2); *stack_f.get_unchecked_mut(*sp_f) = m[0]*m[3] - m[1]*m[2]; *sp_f += 1; }
#[inline(always)] pub unsafe fn eval_trace_dual_m2(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_m2: &mut usize, stack_m2: &[[DualSimd; 4]; 32]) { *sp_m2 -= 1; let m = *stack_m2.get_unchecked(*sp_m2); *stack_f.get_unchecked_mut(*sp_f) = m[0] + m[3]; *sp_f += 1; }

#[inline(always)] pub unsafe fn eval_make_dual_mat3(sp_v3: &mut usize, stack_v3: &[[DualSimd; 3]; 32], sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) { *sp_v3 -= 3; let c0 = *stack_v3.get_unchecked(*sp_v3); let c1 = *stack_v3.get_unchecked(*sp_v3 + 1); let c2 = *stack_v3.get_unchecked(*sp_v3 + 2); *stack_m3.get_unchecked_mut(*sp_m3) = [c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]]; *sp_m3 += 1; }
#[inline(always)] pub unsafe fn eval_add_dual_m3(sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) { *sp_m3 -= 2; let a = *stack_m3.get_unchecked(*sp_m3); let b = *stack_m3.get_unchecked(*sp_m3 + 1); *stack_m3.get_unchecked_mut(*sp_m3) = [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4], a[5]+b[5], a[6]+b[6], a[7]+b[7], a[8]+b[8]]; *sp_m3 += 1; }
#[inline(always)] pub unsafe fn eval_sub_dual_m3(sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) { *sp_m3 -= 2; let a = *stack_m3.get_unchecked(*sp_m3); let b = *stack_m3.get_unchecked(*sp_m3 + 1); *stack_m3.get_unchecked_mut(*sp_m3) = [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3], a[4]-b[4], a[5]-b[5], a[6]-b[6], a[7]-b[7], a[8]-b[8]]; *sp_m3 += 1; }
#[inline(always)] pub unsafe fn eval_scale_dual_m3(sp_f: &mut usize, stack_f: &[DualSimd; 32], sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) { *sp_f -= 1; *sp_m3 -= 1; let s = *stack_f.get_unchecked(*sp_f); let m = *stack_m3.get_unchecked(*sp_m3); *stack_m3.get_unchecked_mut(*sp_m3) = [m[0]*s, m[1]*s, m[2]*s, m[3]*s, m[4]*s, m[5]*s, m[6]*s, m[7]*s, m[8]*s]; *sp_m3 += 1; }
#[inline(always)] pub unsafe fn eval_mul_dual_m3(sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) { *sp_m3 -= 2; let a = *stack_m3.get_unchecked(*sp_m3); let b = *stack_m3.get_unchecked(*sp_m3 + 1); *stack_m3.get_unchecked_mut(*sp_m3) = [ a[0]*b[0] + a[3]*b[1] + a[6]*b[2], a[1]*b[0] + a[4]*b[1] + a[7]*b[2], a[2]*b[0] + a[5]*b[1] + a[8]*b[2], a[0]*b[3] + a[3]*b[4] + a[6]*b[5], a[1]*b[3] + a[4]*b[4] + a[7]*b[5], a[2]*b[3] + a[5]*b[4] + a[8]*b[5], a[0]*b[6] + a[3]*b[7] + a[6]*b[8], a[1]*b[6] + a[4]*b[7] + a[7]*b[8], a[2]*b[6] + a[5]*b[7] + a[8]*b[8] ]; *sp_m3 += 1; }
#[inline(always)] pub unsafe fn eval_mul_dual_m3v3(sp_m3: &mut usize, stack_m3: &[[DualSimd; 9]; 32], sp_v3: &mut usize, stack_v3: &mut [[DualSimd; 3]; 32]) { *sp_m3 -= 1; *sp_v3 -= 1; let m = *stack_m3.get_unchecked(*sp_m3); let v = *stack_v3.get_unchecked(*sp_v3); *stack_v3.get_unchecked_mut(*sp_v3) = [m[0]*v[0] + m[3]*v[1] + m[6]*v[2], m[1]*v[0] + m[4]*v[1] + m[7]*v[2], m[2]*v[0] + m[5]*v[1] + m[8]*v[2]]; *sp_v3 += 1; }
#[inline(always)] pub unsafe fn eval_det_dual_m3(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_m3: &mut usize, stack_m3: &[[DualSimd; 9]; 32]) { *sp_m3 -= 1; let m = *stack_m3.get_unchecked(*sp_m3); *stack_f.get_unchecked_mut(*sp_f) = m[0]*(m[4]*m[8] - m[5]*m[7]) - m[3]*(m[1]*m[8] - m[2]*m[7]) + m[6]*(m[1]*m[5] - m[2]*m[4]); *sp_f += 1; }
#[inline(always)] pub unsafe fn eval_trace_dual_m3(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_m3: &mut usize, stack_m3: &[[DualSimd; 9]; 32]) { *sp_m3 -= 1; let m = *stack_m3.get_unchecked(*sp_m3); *stack_f.get_unchecked_mut(*sp_f) = m[0] + m[4] + m[8]; *sp_f += 1; }
#[inline(always)] pub unsafe fn eval_transpose_dual_m3(sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) { let idx = *sp_m3 - 1; let m = *stack_m3.get_unchecked(idx); *stack_m3.get_unchecked_mut(idx) = [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]; }
#[inline(always)] pub unsafe fn eval_scale_dual_v3(sp_f: &mut usize, stack_f: &[DualSimd; 32], sp_v3: &mut usize, stack_v3: &mut [[DualSimd; 3]; 32]) { *sp_f -= 1; *sp_v3 -= 1; let s = *stack_f.get_unchecked(*sp_f); let v = *stack_v3.get_unchecked(*sp_v3); *stack_v3.get_unchecked_mut(*sp_v3) = [v[0]*s, v[1]*s, v[2]*s]; *sp_v3 += 1; }

#[inline(always)] pub unsafe fn eval_sub_dual_m2(sp_m2: &mut usize, stack_m2: &mut [[DualSimd; 4]; 32]) { *sp_m2 -= 2; let a = *stack_m2.get_unchecked(*sp_m2); let b = *stack_m2.get_unchecked(*sp_m2 + 1); *stack_m2.get_unchecked_mut(*sp_m2) = [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]]; *sp_m2 += 1; }
