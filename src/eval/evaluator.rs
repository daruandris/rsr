use wide::f32x4;

use super::autodiff::{self, DualSimd};
use super::basic;
use super::instruction::Instruction;
use super::linalg;
use super::scalar::Scalar;
use crate::data::dataset::Dataset;
use crate::expr::program::Program;

#[inline(always)]
pub fn eval_simd(program: &Program, features: &[f32x4]) -> f32x4 {
    let code = &program.code;
    let constants = &program.constants;

    let mut stack_f: [f32x4; 32] = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
    let mut sp_f: usize = 0;

    let mut stack_v2: [[f32x4; 2]; 32] = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
    let mut sp_v2: usize = 0;

    let mut stack_v3: [[f32x4; 3]; 32] = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
    let mut sp_v3: usize = 0;

    let mut stack_m2: [[f32x4; 4]; 32] = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
    let mut sp_m2: usize = 0;

    let mut stack_m3: [[f32x4; 9]; 32] = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
    let mut sp_m3: usize = 0;

    for op in code {
        match op {
            Instruction::LoadVarF(idx) => unsafe {
                *stack_f.get_unchecked_mut(sp_f) = *features.get_unchecked(*idx as usize);
                sp_f += 1;
            },
            Instruction::LoadConstF(idx) => unsafe {
                if let Scalar::Float(val) = constants.get_unchecked(*idx as usize) {
                    *stack_f.get_unchecked_mut(sp_f) = f32x4::splat(*val);
                }
                sp_f += 1;
            },
            Instruction::LoadConstV2(idx) => unsafe {
                if let Scalar::Vec2(val) = constants.get_unchecked(*idx as usize) {
                    *stack_v2.get_unchecked_mut(sp_v2) =
                        [f32x4::splat(val[0]), f32x4::splat(val[1])];
                }
                sp_v2 += 1;
            },
            Instruction::LoadConstV3(idx) => unsafe {
                if let Scalar::Vec3(val) = constants.get_unchecked(*idx as usize) {
                    *stack_v3.get_unchecked_mut(sp_v3) = [
                        f32x4::splat(val[0]),
                        f32x4::splat(val[1]),
                        f32x4::splat(val[2]),
                    ];
                }
                sp_v3 += 1;
            },
            Instruction::LoadConstM2(idx) => unsafe {
                if let Scalar::Mat2(val) = constants.get_unchecked(*idx as usize) {
                    *stack_m2.get_unchecked_mut(sp_m2) = [
                        f32x4::splat(val[0]),
                        f32x4::splat(val[1]),
                        f32x4::splat(val[2]),
                        f32x4::splat(val[3]),
                    ];
                }
                sp_m2 += 1;
            },
            Instruction::LoadConstM3(idx) => unsafe {
                if let Scalar::Mat3(val) = constants.get_unchecked(*idx as usize) {
                    *stack_m3.get_unchecked_mut(sp_m3) = [
                        f32x4::splat(val[0]),
                        f32x4::splat(val[1]),
                        f32x4::splat(val[2]),
                        f32x4::splat(val[3]),
                        f32x4::splat(val[4]),
                        f32x4::splat(val[5]),
                        f32x4::splat(val[6]),
                        f32x4::splat(val[7]),
                        f32x4::splat(val[8]),
                    ];
                }
                sp_m3 += 1;
            },

            Instruction::AddF => unsafe { basic::eval_add_f(&mut sp_f, &mut stack_f) },
            Instruction::SubF => unsafe { basic::eval_sub_f(&mut sp_f, &mut stack_f) },
            Instruction::MulF => unsafe { basic::eval_mul_f(&mut sp_f, &mut stack_f) },
            Instruction::DivF => unsafe { basic::eval_div_f(&mut sp_f, &mut stack_f) },
            Instruction::SinF => unsafe { basic::eval_sin_f(&mut sp_f, &mut stack_f) },
            Instruction::CosF => unsafe { basic::eval_cos_f(&mut sp_f, &mut stack_f) },
            Instruction::ExpF => unsafe { basic::eval_exp_f(&mut sp_f, &mut stack_f) },
            Instruction::SqrF => unsafe { basic::eval_sqr_f(&mut sp_f, &mut stack_f) },
            Instruction::SqrtF => unsafe { basic::eval_sqrt_f(&mut sp_f, &mut stack_f) },
            Instruction::LnF => unsafe { basic::eval_ln_f(&mut sp_f, &mut stack_f) },

            Instruction::MakeVec2 => unsafe {
                linalg::eval_make_vec2(&mut sp_f, &stack_f, &mut sp_v2, &mut stack_v2)
            },
            Instruction::MakeVec3 => unsafe {
                linalg::eval_make_vec3(&mut sp_f, &stack_f, &mut sp_v3, &mut stack_v3)
            },
            Instruction::GetXV2 => unsafe {
                linalg::eval_get_x_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2)
            },
            Instruction::GetYV2 => unsafe {
                linalg::eval_get_y_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2)
            },
            Instruction::GetXV3 => unsafe {
                linalg::eval_get_x_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3)
            },
            Instruction::GetYV3 => unsafe {
                linalg::eval_get_y_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3)
            },
            Instruction::GetZV3 => unsafe {
                linalg::eval_get_z_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3)
            },
            Instruction::AddV2 => unsafe { linalg::eval_add_v2(&mut sp_v2, &mut stack_v2) },
            Instruction::SubV2 => unsafe { linalg::eval_sub_v2(&mut sp_v2, &mut stack_v2) },
            Instruction::ScaleV2 => unsafe {
                linalg::eval_scale_v2(&mut sp_f, &stack_f, &mut sp_v2, &mut stack_v2)
            },
            Instruction::DotV2 => unsafe {
                linalg::eval_dot_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2)
            },
            Instruction::NormV2 => unsafe {
                linalg::eval_norm_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2)
            },
            Instruction::AddV3 => unsafe { linalg::eval_add_v3(&mut sp_v3, &mut stack_v3) },
            Instruction::SubV3 => unsafe { linalg::eval_sub_v3(&mut sp_v3, &mut stack_v3) },
            Instruction::ScaleV3 => unsafe {
                linalg::eval_scale_v3(&mut sp_f, &stack_f, &mut sp_v3, &mut stack_v3)
            },
            Instruction::DotV3 => unsafe {
                linalg::eval_dot_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3)
            },
            Instruction::NormV3 => unsafe {
                linalg::eval_norm_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3)
            },
            Instruction::CrossV3 => unsafe { linalg::eval_cross_v3(&mut sp_v3, &mut stack_v3) },
            Instruction::MakeMat2 => unsafe {
                linalg::eval_make_mat2(&mut sp_v2, &stack_v2, &mut sp_m2, &mut stack_m2)
            },
            Instruction::AddM2 => unsafe { linalg::eval_add_m2(&mut sp_m2, &mut stack_m2) },
            Instruction::SubM2 => unsafe { linalg::eval_sub_m2(&mut sp_m2, &mut stack_m2) },
            Instruction::ScaleM2 => unsafe {
                linalg::eval_scale_m2(&mut sp_f, &stack_f, &mut sp_m2, &mut stack_m2)
            },
            Instruction::MulM2 => unsafe { linalg::eval_mul_m2(&mut sp_m2, &mut stack_m2) },
            Instruction::MulM2V2 => unsafe {
                linalg::eval_mul_m2v2(&mut sp_m2, &stack_m2, &mut sp_v2, &mut stack_v2)
            },
            Instruction::DetM2 => unsafe {
                linalg::eval_det_m2(&mut sp_f, &mut stack_f, &mut sp_m2, &stack_m2)
            },
            Instruction::TraceM2 => unsafe {
                linalg::eval_trace_m2(&mut sp_f, &mut stack_f, &mut sp_m2, &stack_m2)
            },
            Instruction::TransposeM2 => unsafe {
                linalg::eval_transpose_m2(&mut sp_m2, &mut stack_m2)
            },
            Instruction::InverseM2 => unsafe { linalg::eval_inverse_m2(&mut sp_m2, &mut stack_m2) },
            Instruction::MakeMat3 => unsafe {
                linalg::eval_make_mat3(&mut sp_v3, &stack_v3, &mut sp_m3, &mut stack_m3)
            },
            Instruction::AddM3 => unsafe { linalg::eval_add_m3(&mut sp_m3, &mut stack_m3) },
            Instruction::SubM3 => unsafe { linalg::eval_sub_m3(&mut sp_m3, &mut stack_m3) },
            Instruction::ScaleM3 => unsafe {
                linalg::eval_scale_m3(&mut sp_f, &stack_f, &mut sp_m3, &mut stack_m3)
            },
            Instruction::MulM3 => unsafe { linalg::eval_mul_m3(&mut sp_m3, &mut stack_m3) },
            Instruction::MulM3V3 => unsafe {
                linalg::eval_mul_m3v3(&mut sp_m3, &stack_m3, &mut sp_v3, &mut stack_v3)
            },
            Instruction::DetM3 => unsafe {
                linalg::eval_det_m3(&mut sp_f, &mut stack_f, &mut sp_m3, &stack_m3)
            },
            Instruction::TraceM3 => unsafe {
                linalg::eval_trace_m3(&mut sp_f, &mut stack_f, &mut sp_m3, &stack_m3)
            },
            Instruction::TransposeM3 => unsafe {
                linalg::eval_transpose_m3(&mut sp_m3, &mut stack_m3)
            },
            Instruction::InverseM3 => unsafe { linalg::eval_inverse_m3(&mut sp_m3, &mut stack_m3) },

            Instruction::LoadVarV2(idx) => unsafe {
                let i = *idx as usize;
                *stack_v2.get_unchecked_mut(sp_v2) =
                    [*features.get_unchecked(i), *features.get_unchecked(i + 1)];
                sp_v2 += 1;
            },
            Instruction::LoadVarV3(idx) => unsafe {
                let i = *idx as usize;
                *stack_v3.get_unchecked_mut(sp_v3) = [
                    *features.get_unchecked(i),
                    *features.get_unchecked(i + 1),
                    *features.get_unchecked(i + 2),
                ];
                sp_v3 += 1;
            },
            Instruction::LoadVarM2(idx) => unsafe {
                let i = *idx as usize;
                *stack_m2.get_unchecked_mut(sp_m2) = [
                    *features.get_unchecked(i),
                    *features.get_unchecked(i + 1),
                    *features.get_unchecked(i + 2),
                    *features.get_unchecked(i + 3),
                ];
                sp_m2 += 1;
            },
            Instruction::LoadVarM3(idx) => unsafe {
                let i = *idx as usize;
                *stack_m3.get_unchecked_mut(sp_m3) = [
                    *features.get_unchecked(i),
                    *features.get_unchecked(i + 1),
                    *features.get_unchecked(i + 2),
                    *features.get_unchecked(i + 3),
                    *features.get_unchecked(i + 4),
                    *features.get_unchecked(i + 5),
                    *features.get_unchecked(i + 6),
                    *features.get_unchecked(i + 7),
                    *features.get_unchecked(i + 8),
                ];
                sp_m3 += 1;
            },
            _ => {}
        }
    }
    unsafe { *stack_f.get_unchecked(0) }
}

pub fn compute_mse(program: &Program, dataset: &Dataset) -> f32 {
    let mut sum_squared_error = 0.0;
    let num_features = dataset.num_features as usize;
    let flat_features = &dataset.feature_flat;
    let targets = &dataset.target_batches;

    for i in 0..dataset.num_batches {
        let start = i * num_features;
        let input_batch = unsafe { flat_features.get_unchecked(start..start + num_features) };

        let prediction = eval_simd(program, input_batch);

        let target = unsafe { *targets.get_unchecked(i) };
        let diff = prediction - target;
        sum_squared_error += (diff * diff).reduce_add();
    }

    if !sum_squared_error.is_finite() {
        return f32::MAX;
    }

    sum_squared_error / (dataset.num_samples as f32)
}

pub fn compute_mse_with_gradient(program: &Program, dataset: &Dataset) -> (f32, [f32; 32]) {
    let mut sum_squared_error = f32x4::splat(0.0);
    let mut grad_sum = [f32x4::splat(0.0); 32];

    let num_features = dataset.num_features as usize;
    let flat_features = &dataset.feature_flat;
    let targets = &dataset.target_batches;

    let mut active_params_count = 0;
    for c in &program.constants {
        active_params_count += match c {
            Scalar::Float(_) => 1,
            Scalar::Vec2(_) => 2,
            Scalar::Vec3(_) => 3,
            Scalar::Mat2(_) => 4,
            Scalar::Mat3(_) => 9,
            _ => 0,
        };
    }
    active_params_count = active_params_count.min(32);

    if active_params_count == 0 {
        return (compute_mse(program, dataset), [0.0; 32]);
    }

    for i in 0..dataset.num_batches {
        let start = i * num_features;
        let input_batch = unsafe { flat_features.get_unchecked(start..start + num_features) };
        let target = unsafe { *targets.get_unchecked(i) };

        let mut diff = f32x4::splat(0.0);

        for k in 0..active_params_count {
            let dual_result = eval_simd_dual(program, input_batch, k);

            if k == 0 {
                diff = dual_result.val - target;
                sum_squared_error += diff * diff;
            }
            grad_sum[k] += f32x4::splat(2.0) * diff * dual_result.grad;
        }
    }

    let num_samples_f32 = dataset.num_samples as f32;
    let total_mse = sum_squared_error.reduce_add() / num_samples_f32;

    if !total_mse.is_finite() {
        return (f32::MAX, [0.0; 32]);
    }

    let mut final_gradient = [0.0f32; 32];
    for k in 0..active_params_count {
        final_gradient[k] = grad_sum[k].reduce_add() / num_samples_f32;
    }

    (total_mse, final_gradient)
}

#[inline(always)]
pub fn eval_simd_dual(program: &Program, features: &[f32x4], active_const_idx: usize) -> DualSimd {
    let code = &program.code;
    let constants = &program.constants;

    let mut stack_f: [DualSimd; 32] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let mut sp_f: usize = 0;

    let mut stack_v2: [[DualSimd; 2]; 32] =
        unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let mut sp_v2: usize = 0;

    let mut stack_v3: [[DualSimd; 3]; 32] =
        unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let mut sp_v3: usize = 0;

    let mut stack_m2: [[DualSimd; 4]; 32] =
        unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let mut sp_m2: usize = 0;

    let mut stack_m3: [[DualSimd; 9]; 32] =
        unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let mut sp_m3: usize = 0;

    let get_flat_start_idx = |target_c_idx: usize| -> usize {
        let mut flat_idx = 0;
        for c in constants.iter().take(target_c_idx) {
            flat_idx += match c {
                Scalar::Float(_) => 1,
                Scalar::Vec2(_) => 2,
                Scalar::Vec3(_) => 3,
                Scalar::Mat2(_) => 4,
                Scalar::Mat3(_) => 9,
                _ => 0,
            };
        }
        flat_idx
    };

    let get_grad = |flat_idx: usize| -> f32x4 {
        if flat_idx == active_const_idx {
            f32x4::splat(1.0)
        } else {
            f32x4::splat(0.0)
        }
    };

    for op in code {
        match op {
            Instruction::LoadVarF(idx) => unsafe {
                *stack_f.get_unchecked_mut(sp_f) =
                    DualSimd::constant(*features.get_unchecked(*idx as usize));
                sp_f += 1;
            },
            Instruction::LoadVarV2(idx) => unsafe {
                let i = *idx as usize;
                *stack_v2.get_unchecked_mut(sp_v2) = [
                    DualSimd::constant(*features.get_unchecked(i)),
                    DualSimd::constant(*features.get_unchecked(i + 1)),
                ];
                sp_v2 += 1;
            },
            Instruction::LoadVarV3(idx) => unsafe {
                let i = *idx as usize;
                *stack_v3.get_unchecked_mut(sp_v3) = [
                    DualSimd::constant(*features.get_unchecked(i)),
                    DualSimd::constant(*features.get_unchecked(i + 1)),
                    DualSimd::constant(*features.get_unchecked(i + 2)),
                ];
                sp_v3 += 1;
            },
            Instruction::LoadConstF(idx) => unsafe {
                if let Scalar::Float(val) = constants.get_unchecked(*idx as usize) {
                    let flat_idx = get_flat_start_idx(*idx as usize);
                    *stack_f.get_unchecked_mut(sp_f) =
                        DualSimd::new(f32x4::splat(*val), get_grad(flat_idx));
                }
                sp_f += 1;
            },
            Instruction::LoadConstV2(idx) => unsafe {
                if let Scalar::Vec2(val) = constants.get_unchecked(*idx as usize) {
                    let flat_idx = get_flat_start_idx(*idx as usize);
                    *stack_v2.get_unchecked_mut(sp_v2) = [
                        DualSimd::new(f32x4::splat(val[0]), get_grad(flat_idx)),
                        DualSimd::new(f32x4::splat(val[1]), get_grad(flat_idx + 1)),
                    ];
                }
                sp_v2 += 1;
            },
            Instruction::LoadConstV3(idx) => unsafe {
                if let Scalar::Vec3(val) = constants.get_unchecked(*idx as usize) {
                    let flat_idx = get_flat_start_idx(*idx as usize);
                    *stack_v3.get_unchecked_mut(sp_v3) = [
                        DualSimd::new(f32x4::splat(val[0]), get_grad(flat_idx)),
                        DualSimd::new(f32x4::splat(val[1]), get_grad(flat_idx + 1)),
                        DualSimd::new(f32x4::splat(val[2]), get_grad(flat_idx + 2)),
                    ];
                }
                sp_v3 += 1;
            },

            Instruction::AddF => unsafe { autodiff::eval_add_dual_f(&mut sp_f, &mut stack_f) },
            Instruction::SubF => unsafe { autodiff::eval_sub_dual_f(&mut sp_f, &mut stack_f) },
            Instruction::MulF => unsafe { autodiff::eval_mul_dual_f(&mut sp_f, &mut stack_f) },
            Instruction::DivF => unsafe { autodiff::eval_div_dual_f(&mut sp_f, &mut stack_f) },
            Instruction::SinF => unsafe { autodiff::eval_sin_dual_f(&mut sp_f, &mut stack_f) },
            Instruction::CosF => unsafe { autodiff::eval_cos_dual_f(&mut sp_f, &mut stack_f) },
            Instruction::ExpF => unsafe { autodiff::eval_exp_dual_f(&mut sp_f, &mut stack_f) },
            Instruction::SqrF => unsafe { autodiff::eval_sqr_dual_f(&mut sp_f, &mut stack_f) },
            Instruction::SqrtF => unsafe { autodiff::eval_sqrt_dual_f(&mut sp_f, &mut stack_f) },
            Instruction::LnF => unsafe { autodiff::eval_ln_dual_f(&mut sp_f, &mut stack_f) },

            Instruction::MakeVec2 => unsafe {
                autodiff::eval_make_dual_vec2(&mut sp_f, &stack_f, &mut sp_v2, &mut stack_v2)
            },
            Instruction::MakeVec3 => unsafe {
                autodiff::eval_make_dual_vec3(&mut sp_f, &stack_f, &mut sp_v3, &mut stack_v3)
            },
            Instruction::GetXV2 => unsafe {
                autodiff::eval_get_x_dual_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2)
            },
            Instruction::GetYV2 => unsafe {
                autodiff::eval_get_y_dual_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2)
            },
            Instruction::GetXV3 => unsafe {
                autodiff::eval_get_x_dual_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3)
            },
            Instruction::GetYV3 => unsafe {
                autodiff::eval_get_y_dual_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3)
            },
            Instruction::GetZV3 => unsafe {
                autodiff::eval_get_z_dual_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3)
            },

            Instruction::AddV2 => unsafe { autodiff::eval_add_dual_v2(&mut sp_v2, &mut stack_v2) },
            Instruction::SubV2 => unsafe { autodiff::eval_sub_dual_v2(&mut sp_v2, &mut stack_v2) },
            Instruction::ScaleV2 => unsafe {
                autodiff::eval_scale_dual_v2(&mut sp_f, &stack_f, &mut sp_v2, &mut stack_v2)
            },
            Instruction::DotV2 => unsafe {
                autodiff::eval_dot_dual_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2)
            },
            Instruction::NormV2 => unsafe {
                autodiff::eval_norm_dual_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2)
            },

            Instruction::AddV3 => unsafe { autodiff::eval_add_dual_v3(&mut sp_v3, &mut stack_v3) },
            Instruction::SubV3 => unsafe { autodiff::eval_sub_dual_v3(&mut sp_v3, &mut stack_v3) },
            Instruction::ScaleV3 => unsafe {
                autodiff::eval_scale_dual_v3(&mut sp_f, &stack_f, &mut sp_v3, &mut stack_v3)
            },
            Instruction::DotV3 => unsafe {
                autodiff::eval_dot_dual_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3)
            },
            Instruction::NormV3 => unsafe {
                autodiff::eval_norm_dual_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3)
            },
            Instruction::CrossV3 => unsafe {
                autodiff::eval_cross_dual_v3(&mut sp_v3, &mut stack_v3)
            },

            Instruction::MakeMat2 => unsafe {
                autodiff::eval_make_dual_mat2(&mut sp_v2, &stack_v2, &mut sp_m2, &mut stack_m2)
            },
            Instruction::AddM2 => unsafe { autodiff::eval_add_dual_m2(&mut sp_m2, &mut stack_m2) },
            Instruction::SubM2 => unsafe { autodiff::eval_sub_dual_m2(&mut sp_m2, &mut stack_m2) },
            Instruction::ScaleM2 => unsafe {
                autodiff::eval_scale_dual_m2(&mut sp_f, &stack_f, &mut sp_m2, &mut stack_m2)
            },
            Instruction::MulM2 => unsafe { autodiff::eval_mul_dual_m2(&mut sp_m2, &mut stack_m2) },
            Instruction::MulM2V2 => unsafe {
                autodiff::eval_mul_dual_m2v2(&mut sp_m2, &stack_m2, &mut sp_v2, &mut stack_v2)
            },
            Instruction::DetM2 => unsafe {
                autodiff::eval_det_dual_m2(&mut sp_f, &mut stack_f, &mut sp_m2, &stack_m2)
            },
            Instruction::TraceM2 => unsafe {
                autodiff::eval_trace_dual_m2(&mut sp_f, &mut stack_f, &mut sp_m2, &stack_m2)
            },
            Instruction::TransposeM2 => unsafe {
                autodiff::eval_transpose_dual_m2(&mut sp_m2, &mut stack_m2)
            },

            Instruction::MakeMat3 => unsafe {
                autodiff::eval_make_dual_mat3(&mut sp_v3, &stack_v3, &mut sp_m3, &mut stack_m3)
            },
            Instruction::AddM3 => unsafe { autodiff::eval_add_dual_m3(&mut sp_m3, &mut stack_m3) },
            Instruction::SubM3 => unsafe { autodiff::eval_sub_dual_m3(&mut sp_m3, &mut stack_m3) },
            Instruction::ScaleM3 => unsafe {
                autodiff::eval_scale_dual_m3(&mut sp_f, &stack_f, &mut sp_m3, &mut stack_m3)
            },
            Instruction::MulM3 => unsafe { autodiff::eval_mul_dual_m3(&mut sp_m3, &mut stack_m3) },
            Instruction::MulM3V3 => unsafe {
                autodiff::eval_mul_dual_m3v3(&mut sp_m3, &stack_m3, &mut sp_v3, &mut stack_v3)
            },
            Instruction::DetM3 => unsafe {
                autodiff::eval_det_dual_m3(&mut sp_f, &mut stack_f, &mut sp_m3, &stack_m3)
            },
            Instruction::TraceM3 => unsafe {
                autodiff::eval_trace_dual_m3(&mut sp_f, &mut stack_f, &mut sp_m3, &stack_m3)
            },
            Instruction::TransposeM3 => unsafe {
                autodiff::eval_transpose_dual_m3(&mut sp_m3, &mut stack_m3)
            },

            _ => {}
        }
    }

    unsafe { *stack_f.get_unchecked(0) }
}
