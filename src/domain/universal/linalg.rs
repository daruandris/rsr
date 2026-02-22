use wide::{f32x4, CmpLt};
use crate::domain::universal::UniversalOp;

// --- VEKTOR KONSTRUKTOROK ÉS GETTEREK ---
#[inline(always)] pub unsafe fn eval_make_vec2(sp_f: &mut usize, stack_f: &[f32x4; 32], sp_v2: &mut usize, stack_v2: &mut [[f32x4; 2]; 32]) {
    *sp_f -= 2; *stack_v2.get_unchecked_mut(*sp_v2) = [*stack_f.get_unchecked(*sp_f), *stack_f.get_unchecked(*sp_f + 1)]; *sp_v2 += 1;
}
#[inline(always)] pub unsafe fn eval_make_vec3(sp_f: &mut usize, stack_f: &[f32x4; 32], sp_v3: &mut usize, stack_v3: &mut [[f32x4; 3]; 32]) {
    *sp_f -= 3; *stack_v3.get_unchecked_mut(*sp_v3) = [*stack_f.get_unchecked(*sp_f), *stack_f.get_unchecked(*sp_f + 1), *stack_f.get_unchecked(*sp_f + 2)]; *sp_v3 += 1;
}
#[inline(always)] pub unsafe fn eval_get_x_v2(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_v2: &mut usize, stack_v2: &[[f32x4; 2]; 32]) {
    *sp_v2 -= 1; *stack_f.get_unchecked_mut(*sp_f) = stack_v2.get_unchecked(*sp_v2)[0]; *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_get_y_v2(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_v2: &mut usize, stack_v2: &[[f32x4; 2]; 32]) {
    *sp_v2 -= 1; *stack_f.get_unchecked_mut(*sp_f) = stack_v2.get_unchecked(*sp_v2)[1]; *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_get_x_v3(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_v3: &mut usize, stack_v3: &[[f32x4; 3]; 32]) {
    *sp_v3 -= 1; *stack_f.get_unchecked_mut(*sp_f) = stack_v3.get_unchecked(*sp_v3)[0]; *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_get_y_v3(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_v3: &mut usize, stack_v3: &[[f32x4; 3]; 32]) {
    *sp_v3 -= 1; *stack_f.get_unchecked_mut(*sp_f) = stack_v3.get_unchecked(*sp_v3)[1]; *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_get_z_v3(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_v3: &mut usize, stack_v3: &[[f32x4; 3]; 32]) {
    *sp_v3 -= 1; *stack_f.get_unchecked_mut(*sp_f) = stack_v3.get_unchecked(*sp_v3)[2]; *sp_f += 1;
}

// --- 2D VEKTOR MATEMATIKA ---
#[inline(always)] pub unsafe fn eval_add_v2(sp_v2: &mut usize, stack_v2: &mut [[f32x4; 2]; 32]) {
    *sp_v2 -= 2; let a = *stack_v2.get_unchecked(*sp_v2); let b = *stack_v2.get_unchecked(*sp_v2 + 1); *stack_v2.get_unchecked_mut(*sp_v2) = [a[0]+b[0], a[1]+b[1]]; *sp_v2 += 1;
}
#[inline(always)] pub unsafe fn eval_sub_v2(sp_v2: &mut usize, stack_v2: &mut [[f32x4; 2]; 32]) {
    *sp_v2 -= 2; let a = *stack_v2.get_unchecked(*sp_v2); let b = *stack_v2.get_unchecked(*sp_v2 + 1); *stack_v2.get_unchecked_mut(*sp_v2) = [a[0]-b[0], a[1]-b[1]]; *sp_v2 += 1;
}
#[inline(always)] pub unsafe fn eval_scale_v2(sp_f: &mut usize, stack_f: &[f32x4; 32], sp_v2: &mut usize, stack_v2: &mut [[f32x4; 2]; 32]) {
    *sp_f -= 1; *sp_v2 -= 1; let s = *stack_f.get_unchecked(*sp_f); let v = *stack_v2.get_unchecked(*sp_v2); *stack_v2.get_unchecked_mut(*sp_v2) = [v[0]*s, v[1]*s]; *sp_v2 += 1;
}
#[inline(always)] pub unsafe fn eval_dot_v2(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_v2: &mut usize, stack_v2: &[[f32x4; 2]; 32]) {
    *sp_v2 -= 2; let a = *stack_v2.get_unchecked(*sp_v2); let b = *stack_v2.get_unchecked(*sp_v2 + 1); *stack_f.get_unchecked_mut(*sp_f) = (a[0]*b[0]) + (a[1]*b[1]); *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_norm_v2(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_v2: &mut usize, stack_v2: &[[f32x4; 2]; 32]) {
    let idx = *sp_v2 - 1; let v = *stack_v2.get_unchecked(idx); *stack_f.get_unchecked_mut(*sp_f) = ((v[0]*v[0]) + (v[1]*v[1])).sqrt(); *sp_f += 1; *sp_v2 -= 1;
}

// --- 3D VEKTOR MATEMATIKA ---
#[inline(always)] pub unsafe fn eval_add_v3(sp_v3: &mut usize, stack_v3: &mut [[f32x4; 3]; 32]) {
    *sp_v3 -= 2; let a = *stack_v3.get_unchecked(*sp_v3); let b = *stack_v3.get_unchecked(*sp_v3 + 1); *stack_v3.get_unchecked_mut(*sp_v3) = [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; *sp_v3 += 1;
}
#[inline(always)] pub unsafe fn eval_sub_v3(sp_v3: &mut usize, stack_v3: &mut [[f32x4; 3]; 32]) {
    *sp_v3 -= 2; let a = *stack_v3.get_unchecked(*sp_v3); let b = *stack_v3.get_unchecked(*sp_v3 + 1); *stack_v3.get_unchecked_mut(*sp_v3) = [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; *sp_v3 += 1;
}
#[inline(always)] pub unsafe fn eval_scale_v3(sp_f: &mut usize, stack_f: &[f32x4; 32], sp_v3: &mut usize, stack_v3: &mut [[f32x4; 3]; 32]) {
    *sp_f -= 1; *sp_v3 -= 1; let s = *stack_f.get_unchecked(*sp_f); let v = *stack_v3.get_unchecked(*sp_v3); *stack_v3.get_unchecked_mut(*sp_v3) = [v[0]*s, v[1]*s, v[2]*s]; *sp_v3 += 1;
}
#[inline(always)] pub unsafe fn eval_dot_v3(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_v3: &mut usize, stack_v3: &[[f32x4; 3]; 32]) {
    *sp_v3 -= 2; let a = *stack_v3.get_unchecked(*sp_v3); let b = *stack_v3.get_unchecked(*sp_v3 + 1); *stack_f.get_unchecked_mut(*sp_f) = (a[0]*b[0]) + (a[1]*b[1]) + (a[2]*b[2]); *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_norm_v3(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_v3: &mut usize, stack_v3: &[[f32x4; 3]; 32]) {
    let idx = *sp_v3 - 1; let v = *stack_v3.get_unchecked(idx); *stack_f.get_unchecked_mut(*sp_f) = ((v[0]*v[0]) + (v[1]*v[1]) + (v[2]*v[2])).sqrt(); *sp_f += 1; *sp_v3 -= 1;
}
#[inline(always)] pub unsafe fn eval_cross_v3(sp_v3: &mut usize, stack_v3: &mut [[f32x4; 3]; 32]) {
    *sp_v3 -= 2; let a = *stack_v3.get_unchecked(*sp_v3); let b = *stack_v3.get_unchecked(*sp_v3 + 1);
    *stack_v3.get_unchecked_mut(*sp_v3) = [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]; *sp_v3 += 1;
}

// --- 2x2 MÁTRIX MŰVELETEK ---
#[inline(always)] pub unsafe fn eval_make_mat2(sp_v2: &mut usize, stack_v2: &[[f32x4; 2]; 32], sp_m2: &mut usize, stack_m2: &mut [[f32x4; 4]; 32]) {
    *sp_v2 -= 2; let c0 = *stack_v2.get_unchecked(*sp_v2); let c1 = *stack_v2.get_unchecked(*sp_v2 + 1); *stack_m2.get_unchecked_mut(*sp_m2) = [c0[0], c0[1], c1[0], c1[1]]; *sp_m2 += 1;
}
#[inline(always)] pub unsafe fn eval_add_m2(sp_m2: &mut usize, stack_m2: &mut [[f32x4; 4]; 32]) {
    *sp_m2 -= 2; let a = *stack_m2.get_unchecked(*sp_m2); let b = *stack_m2.get_unchecked(*sp_m2 + 1); *stack_m2.get_unchecked_mut(*sp_m2) = [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]]; *sp_m2 += 1;
}
#[inline(always)] pub unsafe fn eval_sub_m2(sp_m2: &mut usize, stack_m2: &mut [[f32x4; 4]; 32]) {
    *sp_m2 -= 2; let a = *stack_m2.get_unchecked(*sp_m2); let b = *stack_m2.get_unchecked(*sp_m2 + 1); *stack_m2.get_unchecked_mut(*sp_m2) = [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]]; *sp_m2 += 1;
}
#[inline(always)] pub unsafe fn eval_scale_m2(sp_f: &mut usize, stack_f: &[f32x4; 32], sp_m2: &mut usize, stack_m2: &mut [[f32x4; 4]; 32]) {
    *sp_f -= 1; *sp_m2 -= 1; let s = *stack_f.get_unchecked(*sp_f); let m = *stack_m2.get_unchecked(*sp_m2); *stack_m2.get_unchecked_mut(*sp_m2) = [m[0]*s, m[1]*s, m[2]*s, m[3]*s]; *sp_m2 += 1;
}
#[inline(always)] pub unsafe fn eval_mul_m2(sp_m2: &mut usize, stack_m2: &mut [[f32x4; 4]; 32]) {
    *sp_m2 -= 2; let a = *stack_m2.get_unchecked(*sp_m2); let b = *stack_m2.get_unchecked(*sp_m2 + 1);
    *stack_m2.get_unchecked_mut(*sp_m2) = [a[0]*b[0] + a[2]*b[1], a[1]*b[0] + a[3]*b[1], a[0]*b[2] + a[2]*b[3], a[1]*b[2] + a[3]*b[3]]; *sp_m2 += 1;
}
#[inline(always)] pub unsafe fn eval_mul_m2v2(sp_m2: &mut usize, stack_m2: &[[f32x4; 4]; 32], sp_v2: &mut usize, stack_v2: &mut [[f32x4; 2]; 32]) {
    *sp_m2 -= 1; *sp_v2 -= 1; let m = *stack_m2.get_unchecked(*sp_m2); let v = *stack_v2.get_unchecked(*sp_v2);
    *stack_v2.get_unchecked_mut(*sp_v2) = [m[0]*v[0] + m[2]*v[1], m[1]*v[0] + m[3]*v[1]]; *sp_v2 += 1;
}
#[inline(always)] pub unsafe fn eval_det_m2(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_m2: &mut usize, stack_m2: &[[f32x4; 4]; 32]) {
    *sp_m2 -= 1; let m = *stack_m2.get_unchecked(*sp_m2); *stack_f.get_unchecked_mut(*sp_f) = (m[0]*m[3]) - (m[1]*m[2]); *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_trace_m2(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_m2: &mut usize, stack_m2: &[[f32x4; 4]; 32]) {
    *sp_m2 -= 1; let m = *stack_m2.get_unchecked(*sp_m2); *stack_f.get_unchecked_mut(*sp_f) = m[0] + m[3]; *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_transpose_m2(sp_m2: &mut usize, stack_m2: &mut [[f32x4; 4]; 32]) {
    let idx = *sp_m2 - 1; let m = *stack_m2.get_unchecked(idx); *stack_m2.get_unchecked_mut(idx) = [m[0], m[2], m[1], m[3]];
}
#[inline(always)] pub unsafe fn eval_inverse_m2(sp_m2: &mut usize, stack_m2: &mut [[f32x4; 4]; 32]) {
    let idx = *sp_m2 - 1; let m = *stack_m2.get_unchecked(idx); let det = (m[0]*m[3]) - (m[1]*m[2]);
    let is_singular = det.abs().simd_lt(f32x4::splat(1e-9)); let safe_det = is_singular.blend(f32x4::splat(1.0), det);
    let inv_d = f32x4::splat(1.0) / safe_det;
    *stack_m2.get_unchecked_mut(idx) = [
        is_singular.blend(f32x4::splat(1.0), m[3]*inv_d), is_singular.blend(f32x4::splat(0.0), -m[1]*inv_d),
        is_singular.blend(f32x4::splat(0.0), -m[2]*inv_d), is_singular.blend(f32x4::splat(1.0), m[0]*inv_d)
    ];
}

// --- 3x3 MÁTRIX MŰVELETEK ---
#[inline(always)] pub unsafe fn eval_make_mat3(sp_v3: &mut usize, stack_v3: &[[f32x4; 3]; 32], sp_m3: &mut usize, stack_m3: &mut [[f32x4; 9]; 32]) {
    *sp_v3 -= 3; let c0 = *stack_v3.get_unchecked(*sp_v3); let c1 = *stack_v3.get_unchecked(*sp_v3 + 1); let c2 = *stack_v3.get_unchecked(*sp_v3 + 2);
    *stack_m3.get_unchecked_mut(*sp_m3) = [c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]]; *sp_m3 += 1;
}
#[inline(always)] pub unsafe fn eval_add_m3(sp_m3: &mut usize, stack_m3: &mut [[f32x4; 9]; 32]) {
    *sp_m3 -= 2; let a = *stack_m3.get_unchecked(*sp_m3); let b = *stack_m3.get_unchecked(*sp_m3 + 1);
    *stack_m3.get_unchecked_mut(*sp_m3) = [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4], a[5]+b[5], a[6]+b[6], a[7]+b[7], a[8]+b[8]]; *sp_m3 += 1;
}
#[inline(always)] pub unsafe fn eval_sub_m3(sp_m3: &mut usize, stack_m3: &mut [[f32x4; 9]; 32]) {
    *sp_m3 -= 2; let a = *stack_m3.get_unchecked(*sp_m3); let b = *stack_m3.get_unchecked(*sp_m3 + 1);
    *stack_m3.get_unchecked_mut(*sp_m3) = [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3], a[4]-b[4], a[5]-b[5], a[6]-b[6], a[7]-b[7], a[8]-b[8]]; *sp_m3 += 1;
}
#[inline(always)] pub unsafe fn eval_scale_m3(sp_f: &mut usize, stack_f: &[f32x4; 32], sp_m3: &mut usize, stack_m3: &mut [[f32x4; 9]; 32]) {
    *sp_f -= 1; *sp_m3 -= 1; let s = *stack_f.get_unchecked(*sp_f); let m = *stack_m3.get_unchecked(*sp_m3);
    *stack_m3.get_unchecked_mut(*sp_m3) = [m[0]*s, m[1]*s, m[2]*s, m[3]*s, m[4]*s, m[5]*s, m[6]*s, m[7]*s, m[8]*s]; *sp_m3 += 1;
}
#[inline(always)] pub unsafe fn eval_mul_m3(sp_m3: &mut usize, stack_m3: &mut [[f32x4; 9]; 32]) {
    *sp_m3 -= 2; let a = *stack_m3.get_unchecked(*sp_m3); let b = *stack_m3.get_unchecked(*sp_m3 + 1);
    *stack_m3.get_unchecked_mut(*sp_m3) = [
        a[0]*b[0] + a[3]*b[1] + a[6]*b[2], a[1]*b[0] + a[4]*b[1] + a[7]*b[2], a[2]*b[0] + a[5]*b[1] + a[8]*b[2],
        a[0]*b[3] + a[3]*b[4] + a[6]*b[5], a[1]*b[3] + a[4]*b[4] + a[7]*b[5], a[2]*b[3] + a[5]*b[4] + a[8]*b[5],
        a[0]*b[6] + a[3]*b[7] + a[6]*b[8], a[1]*b[6] + a[4]*b[7] + a[7]*b[8], a[2]*b[6] + a[5]*b[7] + a[8]*b[8]
    ]; *sp_m3 += 1;
}
#[inline(always)] pub unsafe fn eval_mul_m3v3(sp_m3: &mut usize, stack_m3: &[[f32x4; 9]; 32], sp_v3: &mut usize, stack_v3: &mut [[f32x4; 3]; 32]) {
    *sp_m3 -= 1; *sp_v3 -= 1; let m = *stack_m3.get_unchecked(*sp_m3); let v = *stack_v3.get_unchecked(*sp_v3);
    *stack_v3.get_unchecked_mut(*sp_v3) = [m[0]*v[0] + m[3]*v[1] + m[6]*v[2], m[1]*v[0] + m[4]*v[1] + m[7]*v[2], m[2]*v[0] + m[5]*v[1] + m[8]*v[2]]; *sp_v3 += 1;
}
#[inline(always)] pub unsafe fn eval_trace_m3(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_m3: &mut usize, stack_m3: &[[f32x4; 9]; 32]) {
    *sp_m3 -= 1; let m = *stack_m3.get_unchecked(*sp_m3); *stack_f.get_unchecked_mut(*sp_f) = m[0] + m[4] + m[8]; *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_transpose_m3(sp_m3: &mut usize, stack_m3: &mut [[f32x4; 9]; 32]) {
    let idx = *sp_m3 - 1; let m = *stack_m3.get_unchecked(idx);
    *stack_m3.get_unchecked_mut(idx) = [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}
#[inline(always)] pub unsafe fn eval_det_m3(sp_f: &mut usize, stack_f: &mut [f32x4; 32], sp_m3: &mut usize, stack_m3: &[[f32x4; 9]; 32]) {
    *sp_m3 -= 1; let m = *stack_m3.get_unchecked(*sp_m3);
    *stack_f.get_unchecked_mut(*sp_f) = m[0]*(m[4]*m[8] - m[5]*m[7]) - m[3]*(m[1]*m[8] - m[2]*m[7]) + m[6]*(m[1]*m[5] - m[2]*m[4]); *sp_f += 1;
}
#[inline(always)] pub unsafe fn eval_inverse_m3(sp_m3: &mut usize, stack_m3: &mut [[f32x4; 9]; 32]) {
    let idx = *sp_m3 - 1; let m = *stack_m3.get_unchecked(idx);
    let det = m[0]*(m[4]*m[8] - m[5]*m[7]) - m[3]*(m[1]*m[8] - m[2]*m[7]) + m[6]*(m[1]*m[5] - m[2]*m[4]);
    let is_singular = det.abs().simd_lt(f32x4::splat(1e-9)); let safe_det = is_singular.blend(f32x4::splat(1.0), det);
    let inv_d = f32x4::splat(1.0) / safe_det;
    let adj = [
         (m[4]*m[8] - m[5]*m[7])*inv_d, -(m[1]*m[8] - m[2]*m[7])*inv_d,  (m[1]*m[5] - m[2]*m[4])*inv_d,
        -(m[3]*m[8] - m[5]*m[6])*inv_d,  (m[0]*m[8] - m[2]*m[6])*inv_d, -(m[0]*m[5] - m[2]*m[3])*inv_d,
         (m[3]*m[7] - m[4]*m[6])*inv_d, -(m[0]*m[7] - m[1]*m[6])*inv_d,  (m[0]*m[4] - m[1]*m[3])*inv_d
    ];
    *stack_m3.get_unchecked_mut(idx) = [
        is_singular.blend(f32x4::splat(1.0), adj[0]), is_singular.blend(f32x4::splat(0.0), adj[1]), is_singular.blend(f32x4::splat(0.0), adj[2]),
        is_singular.blend(f32x4::splat(0.0), adj[3]), is_singular.blend(f32x4::splat(1.0), adj[4]), is_singular.blend(f32x4::splat(0.0), adj[5]),
        is_singular.blend(f32x4::splat(0.0), adj[6]), is_singular.blend(f32x4::splat(0.0), adj[7]), is_singular.blend(f32x4::splat(1.0), adj[8])
    ];
}

// --- FORMÁZÁS ---

pub fn format_op(op: UniversalOp, args: &[String]) -> Option<String> {
    match op {
        UniversalOp::MakeVec2 => Some(format!("({}, {})", args[0], args[1])),
        UniversalOp::MakeVec3 => Some(format!("({}, {}, {})", args[0], args[1], args[2])),
        UniversalOp::GetXV2 | UniversalOp::GetXV3 => Some(format!("{}.x", args[0])),
        UniversalOp::GetYV2 | UniversalOp::GetYV3 => Some(format!("{}.y", args[0])),
        UniversalOp::GetZV3 => Some(format!("{}.z", args[0])),
        UniversalOp::DotV2 | UniversalOp::DotV3 => Some(format!("<{} • {}>", args[0], args[1])),
        UniversalOp::CrossV3 => Some(format!("({} x {})", args[0], args[1])),
        UniversalOp::NormV2 | UniversalOp::NormV3 => Some(format!("||{}||", args[0])),
        UniversalOp::ScaleV2 | UniversalOp::ScaleV3 | UniversalOp::ScaleM2 | UniversalOp::ScaleM3 => Some(format!("({} * {})", args[0], args[1])),
        UniversalOp::InverseM2 | UniversalOp::InverseM3 => Some(format!("{}^-1", args[0])),
        UniversalOp::TransposeM2 | UniversalOp::TransposeM3 => Some(format!("{}^T", args[0])),
        UniversalOp::DetM2 | UniversalOp::DetM3 => Some(format!("det({})", args[0])),
        UniversalOp::TraceM2 | UniversalOp::TraceM3 => Some(format!("tr({})", args[0])),
        UniversalOp::AddV2 | UniversalOp::AddV3 | UniversalOp::AddM2 | UniversalOp::AddM3 => Some(format!("({} + {})", args[0], args[1])),
        UniversalOp::SubV2 | UniversalOp::SubV3 | UniversalOp::SubM2 | UniversalOp::SubM3 => Some(format!("({} + {})", args[0], args[1])),
        _ => None,
    }
}