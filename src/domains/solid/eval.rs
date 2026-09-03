#![allow(unsafe_op_in_unsafe_fn)]
use crate::engine::eval::autodiff::DualSimd;
use wide::f32x8;

// =====================================================================
// STANDARD SIMD EVALUATION (f32x8)
// =====================================================================

#[inline(always)]
pub unsafe fn eval_right_cauchy_green_m3(sp_m3: &mut usize, stack_m3: &mut [[f32x8; 9]; 32]) {
    let idx = *sp_m3 - 1;
    let f = *stack_m3.get_unchecked(idx);
    *stack_m3.get_unchecked_mut(idx) = [
        f[0] * f[0] + f[3] * f[3] + f[6] * f[6],
        f[0] * f[1] + f[3] * f[4] + f[6] * f[7],
        f[0] * f[2] + f[3] * f[5] + f[6] * f[8],
        f[1] * f[0] + f[4] * f[3] + f[7] * f[6],
        f[1] * f[1] + f[4] * f[4] + f[7] * f[7],
        f[1] * f[2] + f[4] * f[5] + f[7] * f[8],
        f[2] * f[0] + f[5] * f[3] + f[8] * f[6],
        f[2] * f[1] + f[5] * f[4] + f[8] * f[7],
        f[2] * f[2] + f[5] * f[5] + f[8] * f[8],
    ];
}

#[inline(always)]
pub unsafe fn eval_left_cauchy_green_m3(sp_m3: &mut usize, stack_m3: &mut [[f32x8; 9]; 32]) {
    let idx = *sp_m3 - 1;
    let f = *stack_m3.get_unchecked(idx);
    *stack_m3.get_unchecked_mut(idx) = [
        f[0] * f[0] + f[1] * f[1] + f[2] * f[2],
        f[0] * f[3] + f[1] * f[4] + f[2] * f[5],
        f[0] * f[6] + f[1] * f[7] + f[2] * f[8],
        f[3] * f[0] + f[4] * f[1] + f[5] * f[2],
        f[3] * f[3] + f[4] * f[4] + f[5] * f[5],
        f[3] * f[6] + f[4] * f[7] + f[5] * f[8],
        f[6] * f[0] + f[7] * f[1] + f[8] * f[2],
        f[6] * f[3] + f[7] * f[4] + f[8] * f[5],
        f[6] * f[6] + f[7] * f[7] + f[8] * f[8],
    ];
}

#[inline(always)]
pub unsafe fn eval_invariant2_m3(
    sp_f: &mut usize,
    stack_f: &mut [f32x8; 32],
    sp_m3: &mut usize,
    stack_m3: &[[f32x8; 9]; 32],
) {
    *sp_m3 -= 1;
    let a = *stack_m3.get_unchecked(*sp_m3);

    let tr_a = a[0] + a[4] + a[8];
    let tr_a_sq = tr_a * tr_a;
    let tr_a2 = (a[0] * a[0] + a[1] * a[3] + a[2] * a[6])
        + (a[3] * a[1] + a[4] * a[4] + a[5] * a[7])
        + (a[6] * a[2] + a[7] * a[5] + a[8] * a[8]);

    *stack_f.get_unchecked_mut(*sp_f) = f32x8::splat(0.5) * (tr_a_sq - tr_a2);
    *sp_f += 1;
}

#[inline(always)]
pub unsafe fn eval_cofactor_m3(sp_m3: &mut usize, stack_m3: &mut [[f32x8; 9]; 32]) {
    let idx = *sp_m3 - 1;
    let m = *stack_m3.get_unchecked(idx);
    *stack_m3.get_unchecked_mut(idx) = [
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
}

#[inline(always)]
pub unsafe fn eval_green_lagrange_strain_m3(sp_m3: &mut usize, stack_m3: &mut [[f32x8; 9]; 32]) {
    let idx = *sp_m3 - 1;
    let f = *stack_m3.get_unchecked(idx);
    let half = f32x8::splat(0.5);
    let one = f32x8::splat(1.0);

    // E = 0.5 * (F^T F - I)
    *stack_m3.get_unchecked_mut(idx) = [
        half * (f[0] * f[0] + f[3] * f[3] + f[6] * f[6] - one),
        half * (f[0] * f[1] + f[3] * f[4] + f[6] * f[7]),
        half * (f[0] * f[2] + f[3] * f[5] + f[6] * f[8]),
        half * (f[1] * f[0] + f[4] * f[3] + f[7] * f[6]),
        half * (f[1] * f[1] + f[4] * f[4] + f[7] * f[7] - one),
        half * (f[1] * f[2] + f[4] * f[5] + f[7] * f[8]),
        half * (f[2] * f[0] + f[5] * f[3] + f[8] * f[6]),
        half * (f[2] * f[1] + f[5] * f[4] + f[8] * f[7]),
        half * (f[2] * f[2] + f[5] * f[5] + f[8] * f[8] - one),
    ];
}

#[inline(always)]
pub unsafe fn eval_isochoric_invariant1(
    sp_f: &mut usize,
    stack_f: &mut [f32x8; 32],
    sp_m3: &mut usize,
    stack_m3: &[[f32x8; 9]; 32],
) {
    *sp_m3 -= 1;
    let f = *stack_m3.get_unchecked(*sp_m3);

    let det = f[0] * (f[4] * f[8] - f[5] * f[7]) - f[3] * (f[1] * f[8] - f[2] * f[7])
        + f[6] * (f[1] * f[5] - f[2] * f[4]);
    let j_pow = (det.ln() * f32x8::splat(-2.0 / 3.0)).exp();

    let i1 = f[0] * f[0]
        + f[1] * f[1]
        + f[2] * f[2]
        + f[3] * f[3]
        + f[4] * f[4]
        + f[5] * f[5]
        + f[6] * f[6]
        + f[7] * f[7]
        + f[8] * f[8];

    *stack_f.get_unchecked_mut(*sp_f) = j_pow * i1;
    *sp_f += 1;
}

#[inline(always)]
pub unsafe fn eval_isochoric_invariant2(
    sp_f: &mut usize,
    stack_f: &mut [f32x8; 32],
    sp_m3: &mut usize,
    stack_m3: &[[f32x8; 9]; 32],
) {
    *sp_m3 -= 1;
    let f = *stack_m3.get_unchecked(*sp_m3);

    let det = f[0] * (f[4] * f[8] - f[5] * f[7]) - f[3] * (f[1] * f[8] - f[2] * f[7])
        + f[6] * (f[1] * f[5] - f[2] * f[4]);
    let j_pow = (det.ln() * f32x8::splat(-4.0 / 3.0)).exp();

    let i1 = f[0] * f[0]
        + f[1] * f[1]
        + f[2] * f[2]
        + f[3] * f[3]
        + f[4] * f[4]
        + f[5] * f[5]
        + f[6] * f[6]
        + f[7] * f[7]
        + f[8] * f[8];

    let c00 = f[0] * f[0] + f[3] * f[3] + f[6] * f[6];
    let c01 = f[0] * f[1] + f[3] * f[4] + f[6] * f[7];
    let c02 = f[0] * f[2] + f[3] * f[5] + f[6] * f[8];
    let c10 = f[1] * f[0] + f[4] * f[3] + f[7] * f[6];
    let c11 = f[1] * f[1] + f[4] * f[4] + f[7] * f[7];
    let c12 = f[1] * f[2] + f[4] * f[5] + f[7] * f[8];
    let c20 = f[2] * f[0] + f[5] * f[3] + f[8] * f[6];
    let c21 = f[2] * f[1] + f[5] * f[4] + f[8] * f[7];
    let c22 = f[2] * f[2] + f[5] * f[5] + f[8] * f[8];

    let tr_c2 = c00 * c00
        + c01 * c10
        + c02 * c20
        + c10 * c01
        + c11 * c11
        + c12 * c21
        + c20 * c02
        + c21 * c12
        + c22 * c22;

    *stack_f.get_unchecked_mut(*sp_f) = j_pow * f32x8::splat(0.5) * (i1 * i1 - tr_c2);
    *sp_f += 1;
}

#[inline(always)]
pub unsafe fn eval_trace_sqr_m3(
    sp_f: &mut usize,
    stack_f: &mut [f32x8; 32],
    sp_m3: &mut usize,
    stack_m3: &[[f32x8; 9]; 32],
) {
    *sp_m3 -= 1;
    let a = *stack_m3.get_unchecked(*sp_m3);
    // tr(A^2) = sum_{i,j} A_{ij} A_{ji}
    let tr_sq = a[0]*a[0] + a[1]*a[3] + a[2]*a[6] +
                a[3]*a[1] + a[4]*a[4] + a[5]*a[7] +
                a[6]*a[2] + a[7]*a[5] + a[8]*a[8];
    
    *stack_f.get_unchecked_mut(*sp_f) = tr_sq;
    *sp_f += 1;
}

#[inline(always)]
pub unsafe fn eval_deviatoric_m3(sp_m3: &mut usize, stack_m3: &mut [[f32x8; 9]; 32]) {
    let idx = *sp_m3 - 1;
    let a = *stack_m3.get_unchecked(idx);
    let tr_third = (a[0] + a[4] + a[8]) * f32x8::splat(1.0 / 3.0);
    
    *stack_m3.get_unchecked_mut(idx) = [
        a[0] - tr_third, a[1], a[2],
        a[3], a[4] - tr_third, a[5],
        a[6], a[7], a[8] - tr_third,
    ];
}

#[inline(always)]
pub unsafe fn eval_invariant_j2_m3(
    sp_f: &mut usize,
    stack_f: &mut [f32x8; 32],
    sp_m3: &mut usize,
    stack_m3: &[[f32x8; 9]; 32],
) {
    *sp_m3 -= 1;
    let a = *stack_m3.get_unchecked(*sp_m3);
    let tr_third = (a[0] + a[4] + a[8]) * f32x8::splat(1.0 / 3.0);
    
    let s0 = a[0] - tr_third; let s1 = a[1]; let s2 = a[2];
    let s3 = a[3]; let s4 = a[4] - tr_third; let s5 = a[5];
    let s6 = a[6]; let s7 = a[7]; let s8 = a[8] - tr_third;

    let tr_s2 = s0*s0 + s1*s3 + s2*s6 +
                s3*s1 + s4*s4 + s5*s7 +
                s6*s2 + s7*s5 + s8*s8;
                
    *stack_f.get_unchecked_mut(*sp_f) = tr_s2 * f32x8::splat(0.5);
    *sp_f += 1;
}

#[inline(always)]
pub unsafe fn eval_invariant_j3_m3(
    sp_f: &mut usize,
    stack_f: &mut [f32x8; 32],
    sp_m3: &mut usize,
    stack_m3: &[[f32x8; 9]; 32],
) {
    *sp_m3 -= 1;
    let a = *stack_m3.get_unchecked(*sp_m3);
    let tr_third = (a[0] + a[4] + a[8]) * f32x8::splat(1.0 / 3.0);
    
    let s0 = a[0] - tr_third; let s1 = a[1]; let s2 = a[2];
    let s3 = a[3]; let s4 = a[4] - tr_third; let s5 = a[5];
    let s6 = a[6]; let s7 = a[7]; let s8 = a[8] - tr_third;

    let det_s = s0 * (s4 * s8 - s5 * s7)
              - s3 * (s1 * s8 - s2 * s7)
              + s6 * (s1 * s5 - s2 * s4);
              
    *stack_f.get_unchecked_mut(*sp_f) = det_s;
    *sp_f += 1;
}

// =====================================================================
// DUAL SIMD EVALUATION (Autodiff)
// =====================================================================

#[inline(always)]
pub unsafe fn eval_dual_right_cauchy_green_m3(
    sp_m3: &mut usize,
    stack_m3: &mut [[DualSimd; 9]; 32],
) {
    let idx = *sp_m3 - 1;
    let f = *stack_m3.get_unchecked(idx);
    *stack_m3.get_unchecked_mut(idx) = [
        f[0] * f[0] + f[3] * f[3] + f[6] * f[6],
        f[0] * f[1] + f[3] * f[4] + f[6] * f[7],
        f[0] * f[2] + f[3] * f[5] + f[6] * f[8],
        f[1] * f[0] + f[4] * f[3] + f[7] * f[6],
        f[1] * f[1] + f[4] * f[4] + f[7] * f[7],
        f[1] * f[2] + f[4] * f[5] + f[7] * f[8],
        f[2] * f[0] + f[5] * f[3] + f[8] * f[6],
        f[2] * f[1] + f[5] * f[4] + f[8] * f[7],
        f[2] * f[2] + f[5] * f[5] + f[8] * f[8],
    ];
}

#[inline(always)]
pub unsafe fn eval_dual_left_cauchy_green_m3(
    sp_m3: &mut usize,
    stack_m3: &mut [[DualSimd; 9]; 32],
) {
    let idx = *sp_m3 - 1;
    let f = *stack_m3.get_unchecked(idx);
    *stack_m3.get_unchecked_mut(idx) = [
        f[0] * f[0] + f[1] * f[1] + f[2] * f[2],
        f[0] * f[3] + f[1] * f[4] + f[2] * f[5],
        f[0] * f[6] + f[1] * f[7] + f[2] * f[8],
        f[3] * f[0] + f[4] * f[1] + f[5] * f[2],
        f[3] * f[3] + f[4] * f[4] + f[5] * f[5],
        f[3] * f[6] + f[4] * f[7] + f[5] * f[8],
        f[6] * f[0] + f[7] * f[1] + f[8] * f[2],
        f[6] * f[3] + f[7] * f[4] + f[8] * f[5],
        f[6] * f[6] + f[7] * f[7] + f[8] * f[8],
    ];
}

#[inline(always)]
pub unsafe fn eval_dual_invariant2_m3(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_m3: &mut usize,
    stack_m3: &[[DualSimd; 9]; 32],
) {
    *sp_m3 -= 1;
    let a = *stack_m3.get_unchecked(*sp_m3);

    let tr_a = a[0] + a[4] + a[8];
    let tr_a_sq = tr_a * tr_a;
    let tr_a2 = (a[0] * a[0] + a[1] * a[3] + a[2] * a[6])
        + (a[3] * a[1] + a[4] * a[4] + a[5] * a[7])
        + (a[6] * a[2] + a[7] * a[5] + a[8] * a[8]);

    let half = DualSimd::constant(f32x8::splat(0.5));
    *stack_f.get_unchecked_mut(*sp_f) = half * (tr_a_sq - tr_a2);
    *sp_f += 1;
}

#[inline(always)]
pub unsafe fn eval_dual_cofactor_m3(sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) {
    let idx = *sp_m3 - 1;
    let m = *stack_m3.get_unchecked(idx);
    *stack_m3.get_unchecked_mut(idx) = [
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
}

#[inline(always)]
pub unsafe fn eval_dual_green_lagrange_strain_m3(
    sp_m3: &mut usize,
    stack_m3: &mut [[DualSimd; 9]; 32],
) {
    let idx = *sp_m3 - 1;
    let f = *stack_m3.get_unchecked(idx);
    let half = DualSimd::constant(f32x8::splat(0.5));
    let one = DualSimd::constant(f32x8::splat(1.0));

    *stack_m3.get_unchecked_mut(idx) = [
        half * (f[0] * f[0] + f[3] * f[3] + f[6] * f[6] - one),
        half * (f[0] * f[1] + f[3] * f[4] + f[6] * f[7]),
        half * (f[0] * f[2] + f[3] * f[5] + f[6] * f[8]),
        half * (f[1] * f[0] + f[4] * f[3] + f[7] * f[6]),
        half * (f[1] * f[1] + f[4] * f[4] + f[7] * f[7] - one),
        half * (f[1] * f[2] + f[4] * f[5] + f[7] * f[8]),
        half * (f[2] * f[0] + f[5] * f[3] + f[8] * f[6]),
        half * (f[2] * f[1] + f[5] * f[4] + f[8] * f[7]),
        half * (f[2] * f[2] + f[5] * f[5] + f[8] * f[8] - one),
    ];
}

#[inline(always)]
pub unsafe fn eval_dual_isochoric_invariant1(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_m3: &mut usize,
    stack_m3: &[[DualSimd; 9]; 32],
) {
    *sp_m3 -= 1;
    let f = *stack_m3.get_unchecked(*sp_m3);

    let det = f[0] * (f[4] * f[8] - f[5] * f[7]) - f[3] * (f[1] * f[8] - f[2] * f[7])
        + f[6] * (f[1] * f[5] - f[2] * f[4]);
    let exp_fact = DualSimd::constant(f32x8::splat(-2.0 / 3.0));
    let j_pow = (det.ln() * exp_fact).exp();

    let i1 = f[0] * f[0]
        + f[1] * f[1]
        + f[2] * f[2]
        + f[3] * f[3]
        + f[4] * f[4]
        + f[5] * f[5]
        + f[6] * f[6]
        + f[7] * f[7]
        + f[8] * f[8];

    *stack_f.get_unchecked_mut(*sp_f) = j_pow * i1;
    *sp_f += 1;
}

#[inline(always)]
pub unsafe fn eval_dual_isochoric_invariant2(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_m3: &mut usize,
    stack_m3: &[[DualSimd; 9]; 32],
) {
    *sp_m3 -= 1;
    let f = *stack_m3.get_unchecked(*sp_m3);

    let det = f[0] * (f[4] * f[8] - f[5] * f[7]) - f[3] * (f[1] * f[8] - f[2] * f[7])
        + f[6] * (f[1] * f[5] - f[2] * f[4]);
    let exp_fact = DualSimd::constant(f32x8::splat(-4.0 / 3.0));
    let j_pow = (det.ln() * exp_fact).exp();

    let i1 = f[0] * f[0]
        + f[1] * f[1]
        + f[2] * f[2]
        + f[3] * f[3]
        + f[4] * f[4]
        + f[5] * f[5]
        + f[6] * f[6]
        + f[7] * f[7]
        + f[8] * f[8];

    let c00 = f[0] * f[0] + f[3] * f[3] + f[6] * f[6];
    let c01 = f[0] * f[1] + f[3] * f[4] + f[6] * f[7];
    // ... [Ugyanaz a C kiszámítás, mint fent, csak DualSimd típusokkal, ezt a motor automatikusan túléli] ...
    let c02 = f[0] * f[2] + f[3] * f[5] + f[6] * f[8];
    let c10 = f[1] * f[0] + f[4] * f[3] + f[7] * f[6];
    let c11 = f[1] * f[1] + f[4] * f[4] + f[7] * f[7];
    let c12 = f[1] * f[2] + f[4] * f[5] + f[7] * f[8];
    let c20 = f[2] * f[0] + f[5] * f[3] + f[8] * f[6];
    let c21 = f[2] * f[1] + f[5] * f[4] + f[8] * f[7];
    let c22 = f[2] * f[2] + f[5] * f[5] + f[8] * f[8];

    let tr_c2 = c00 * c00
        + c01 * c10
        + c02 * c20
        + c10 * c01
        + c11 * c11
        + c12 * c21
        + c20 * c02
        + c21 * c12
        + c22 * c22;

    *stack_f.get_unchecked_mut(*sp_f) =
        j_pow * DualSimd::constant(f32x8::splat(0.5)) * (i1 * i1 - tr_c2);
    *sp_f += 1;
}

#[inline(always)]
pub unsafe fn eval_dual_trace_sqr_m3(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_m3: &mut usize,
    stack_m3: &[[DualSimd; 9]; 32],
) {
    *sp_m3 -= 1;
    let a = *stack_m3.get_unchecked(*sp_m3);
    
    // Mivel a DualSimd-re implementálva van a Mul és Add operátor,
    // a matematika pontosan ugyanaz, a gradiens kiszámítása automatikus!
    let tr_sq = (a[0] * a[0] + a[1] * a[3] + a[2] * a[6])
              + (a[3] * a[1] + a[4] * a[4] + a[5] * a[7])
              + (a[6] * a[2] + a[7] * a[5] + a[8] * a[8]);
              
    *stack_f.get_unchecked_mut(*sp_f) = tr_sq;
    *sp_f += 1;
}

#[inline(always)]
pub unsafe fn eval_dual_deviatoric_m3(sp_m3: &mut usize, stack_m3: &mut [[DualSimd; 9]; 32]) {
    let idx = *sp_m3 - 1;
    let a = *stack_m3.get_unchecked(idx);
    
    let tr = a[0] + a[4] + a[8];
    let third = DualSimd::constant(f32x8::splat(1.0 / 3.0));
    let tr_third = tr * third;
    
    *stack_m3.get_unchecked_mut(idx) = [
        a[0] - tr_third, a[1], a[2],
        a[3], a[4] - tr_third, a[5],
        a[6], a[7], a[8] - tr_third,
    ];
}


#[inline(always)]
pub unsafe fn eval_dual_invariant_j2_m3(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_m3: &mut usize,
    stack_m3: &[[DualSimd; 9]; 32],
) {
    *sp_m3 -= 1;
    let a = *stack_m3.get_unchecked(*sp_m3);
    let third = DualSimd::constant(f32x8::splat(1.0 / 3.0));
    let tr_third = (a[0] + a[4] + a[8]) * third;
    
    let s0 = a[0] - tr_third; let s1 = a[1]; let s2 = a[2];
    let s3 = a[3]; let s4 = a[4] - tr_third; let s5 = a[5];
    let s6 = a[6]; let s7 = a[7]; let s8 = a[8] - tr_third;

    let tr_s2 = s0*s0 + s1*s3 + s2*s6 +
                s3*s1 + s4*s4 + s5*s7 +
                s6*s2 + s7*s5 + s8*s8;
                
    let half = DualSimd::constant(f32x8::splat(0.5));
    *stack_f.get_unchecked_mut(*sp_f) = tr_s2 * half;
    *sp_f += 1;
}

#[inline(always)]
pub unsafe fn eval_dual_invariant_j3_m3(
    sp_f: &mut usize,
    stack_f: &mut [DualSimd; 32],
    sp_m3: &mut usize,
    stack_m3: &[[DualSimd; 9]; 32],
) {
    *sp_m3 -= 1;
    let a = *stack_m3.get_unchecked(*sp_m3);
    let third = DualSimd::constant(f32x8::splat(1.0 / 3.0));
    let tr_third = (a[0] + a[4] + a[8]) * third;
    
    let s0 = a[0] - tr_third; let s1 = a[1]; let s2 = a[2];
    let s3 = a[3]; let s4 = a[4] - tr_third; let s5 = a[5];
    let s6 = a[6]; let s7 = a[7]; let s8 = a[8] - tr_third;

    let det_s = s0 * (s4 * s8 - s5 * s7)
              - s3 * (s1 * s8 - s2 * s7)
              + s6 * (s1 * s5 - s2 * s4);
              
    *stack_f.get_unchecked_mut(*sp_f) = det_s;
    *sp_f += 1;
}