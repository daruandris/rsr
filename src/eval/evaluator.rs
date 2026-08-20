use wide::f32x4;
use crate::Instruction;
use crate::data::dataset::Dataset;
use crate::expr::program::Program;
use crate::eval::state::{VmState, DualVmState};
use crate::eval::scalar::Scalar;
use crate::eval::autodiff::{self, DualSimd};
use crate::eval::basic_domain::BasicOpCode;
use crate::eval::linalg_domain::LinalgOpCode;
use crate::optimize::Parameterized;
use crate::SymbolicEngine;

#[inline(always)]
pub fn eval_simd(program: &Program, features: &[f32x4]) -> f32x4 {
    let mut ctx = VmState::new();
    let constants = &program.constants;

    for op in &program.code {
        match op {
            Instruction::LoadVarF(idx) => unsafe {
                *ctx.stack_f.get_unchecked_mut(ctx.sp_f) = *features.get_unchecked(*idx as usize);
                ctx.sp_f += 1;
            },
            Instruction::LoadConstF(idx) => unsafe {
                if let Scalar::Float(val) = constants.get_unchecked(*idx as usize) {
                    *ctx.stack_f.get_unchecked_mut(ctx.sp_f) = f32x4::splat(*val);
                }
                ctx.sp_f += 1;
            },
            Instruction::LoadVarV2(idx) => unsafe {
                let i = *idx as usize;
                *ctx.stack_v2.get_unchecked_mut(ctx.sp_v2) = [*features.get_unchecked(i), *features.get_unchecked(i + 1)];
                ctx.sp_v2 += 1;
            },
            Instruction::LoadConstV2(idx) => unsafe {
                if let Scalar::Vec2(val) = constants.get_unchecked(*idx as usize) {
                    *ctx.stack_v2.get_unchecked_mut(ctx.sp_v2) = [f32x4::splat(val[0]), f32x4::splat(val[1])];
                }
                ctx.sp_v2 += 1;
            },
            Instruction::LoadVarV3(idx) => unsafe {
                let i = *idx as usize;
                *ctx.stack_v3.get_unchecked_mut(ctx.sp_v3) = [*features.get_unchecked(i), *features.get_unchecked(i + 1), *features.get_unchecked(i + 2)];
                ctx.sp_v3 += 1;
            },
            Instruction::LoadConstV3(idx) => unsafe {
                if let Scalar::Vec3(val) = constants.get_unchecked(*idx as usize) {
                    *ctx.stack_v3.get_unchecked_mut(ctx.sp_v3) = [f32x4::splat(val[0]), f32x4::splat(val[1]), f32x4::splat(val[2])];
                }
                ctx.sp_v3 += 1;
            },
            Instruction::LoadVarM2(idx) => unsafe {
                let i = *idx as usize;
                *ctx.stack_m2.get_unchecked_mut(ctx.sp_m2) = [
                    *features.get_unchecked(i), *features.get_unchecked(i + 1),
                    *features.get_unchecked(i + 2), *features.get_unchecked(i + 3),
                ];
                ctx.sp_m2 += 1;
            },
            Instruction::LoadConstM2(idx) => unsafe {
                if let Scalar::Mat2(val) = constants.get_unchecked(*idx as usize) {
                    *ctx.stack_m2.get_unchecked_mut(ctx.sp_m2) = [
                        f32x4::splat(val[0]), f32x4::splat(val[1]),
                        f32x4::splat(val[2]), f32x4::splat(val[3]),
                    ];
                }
                ctx.sp_m2 += 1;
            },
            Instruction::LoadVarM3(idx) => unsafe {
                let i = *idx as usize;
                *ctx.stack_m3.get_unchecked_mut(ctx.sp_m3) = [
                    *features.get_unchecked(i), *features.get_unchecked(i + 1), *features.get_unchecked(i + 2),
                    *features.get_unchecked(i + 3), *features.get_unchecked(i + 4), *features.get_unchecked(i + 5),
                    *features.get_unchecked(i + 6), *features.get_unchecked(i + 7), *features.get_unchecked(i + 8),
                ];
                ctx.sp_m3 += 1;
            },
            Instruction::LoadConstM3(idx) => unsafe {
                if let Scalar::Mat3(val) = constants.get_unchecked(*idx as usize) {
                    *ctx.stack_m3.get_unchecked_mut(ctx.sp_m3) = [
                        f32x4::splat(val[0]), f32x4::splat(val[1]), f32x4::splat(val[2]),
                        f32x4::splat(val[3]), f32x4::splat(val[4]), f32x4::splat(val[5]),
                        f32x4::splat(val[6]), f32x4::splat(val[7]), f32x4::splat(val[8]),
                    ];
                }
                ctx.sp_m3 += 1;
            },
            
            _ => SymbolicEngine::eval_single(*op, &mut ctx),
        }
    }
    unsafe { *ctx.stack_f.get_unchecked(0) }
}

pub fn compute_mse(program: &Program, dataset: &Dataset) -> f32 {
    let mut sum_squared_error = f32x4::splat(0.0);
    let num_features = dataset.num_features as usize;
    let flat_features = &dataset.feature_flat;
    let targets = &dataset.target_batches;

    for i in 0..dataset.num_batches {
        let start = i * num_features;
        let input_batch = unsafe { flat_features.get_unchecked(start..start + num_features) };

        let prediction = eval_simd(program, input_batch);
        let target = unsafe { *targets.get_unchecked(i) };
        
        let diff = prediction - target;
        sum_squared_error += diff * diff;
    }

    let mse = sum_squared_error.reduce_add() / (dataset.num_samples as f32);
    if !mse.is_finite() {
        f32::MAX
    } else {
        mse
    }
}

pub fn compute_mse_with_gradient(program: &Program, dataset: &Dataset) -> (f32, [f32; 32]) {
    let mut sum_squared_error = f32x4::splat(0.0);
    let mut grad_sum = [f32x4::splat(0.0); 32];

    let num_features = dataset.num_features as usize;
    let flat_features = &dataset.feature_flat;
    let targets = &dataset.target_batches;

    let active_params_count = program.param_count().min(32);

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
    let mut ctx = DualVmState::new();
    let constants = &program.constants;

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

    for op in &program.code {
        match op {
            Instruction::LoadVarF(idx) => unsafe {
                *ctx.stack_f.get_unchecked_mut(ctx.sp_f) = DualSimd::constant(*features.get_unchecked(*idx as usize));
                ctx.sp_f += 1;
            },
            Instruction::LoadConstF(idx) => unsafe {
                if let Scalar::Float(val) = constants.get_unchecked(*idx as usize) {
                    let flat_idx = get_flat_start_idx(*idx as usize);
                    *ctx.stack_f.get_unchecked_mut(ctx.sp_f) = DualSimd::new(f32x4::splat(*val), get_grad(flat_idx));
                }
                ctx.sp_f += 1;
            },
            Instruction::LoadVarV2(idx) => unsafe {
                let i = *idx as usize;
                *ctx.stack_v2.get_unchecked_mut(ctx.sp_v2) = [
                    DualSimd::constant(*features.get_unchecked(i)),
                    DualSimd::constant(*features.get_unchecked(i + 1)),
                ];
                ctx.sp_v2 += 1;
            },
            Instruction::LoadConstV2(idx) => unsafe {
                if let Scalar::Vec2(val) = constants.get_unchecked(*idx as usize) {
                    let flat_idx = get_flat_start_idx(*idx as usize);
                    *ctx.stack_v2.get_unchecked_mut(ctx.sp_v2) = [
                        DualSimd::new(f32x4::splat(val[0]), get_grad(flat_idx)),
                        DualSimd::new(f32x4::splat(val[1]), get_grad(flat_idx + 1)),
                    ];
                }
                ctx.sp_v2 += 1;
            },
            Instruction::LoadVarV3(idx) => unsafe {
                let i = *idx as usize;
                *ctx.stack_v3.get_unchecked_mut(ctx.sp_v3) = [
                    DualSimd::constant(*features.get_unchecked(i)),
                    DualSimd::constant(*features.get_unchecked(i + 1)),
                    DualSimd::constant(*features.get_unchecked(i + 2)),
                ];
                ctx.sp_v3 += 1;
            },
            Instruction::LoadConstV3(idx) => unsafe {
                if let Scalar::Vec3(val) = constants.get_unchecked(*idx as usize) {
                    let flat_idx = get_flat_start_idx(*idx as usize);
                    *ctx.stack_v3.get_unchecked_mut(ctx.sp_v3) = [
                        DualSimd::new(f32x4::splat(val[0]), get_grad(flat_idx)),
                        DualSimd::new(f32x4::splat(val[1]), get_grad(flat_idx + 1)),
                        DualSimd::new(f32x4::splat(val[2]), get_grad(flat_idx + 2)),
                    ];
                }
                ctx.sp_v3 += 1;
            },
            Instruction::LoadVarM2(idx) => unsafe {
                let i = *idx as usize;
                *ctx.stack_m2.get_unchecked_mut(ctx.sp_m2) = [
                    DualSimd::constant(*features.get_unchecked(i)),
                    DualSimd::constant(*features.get_unchecked(i + 1)),
                    DualSimd::constant(*features.get_unchecked(i + 2)),
                    DualSimd::constant(*features.get_unchecked(i + 3)),
                ];
                ctx.sp_m2 += 1;
            },
            Instruction::LoadConstM2(idx) => unsafe {
                if let Scalar::Mat2(val) = constants.get_unchecked(*idx as usize) {
                    let flat_idx = get_flat_start_idx(*idx as usize);
                    *ctx.stack_m2.get_unchecked_mut(ctx.sp_m2) = [
                        DualSimd::new(f32x4::splat(val[0]), get_grad(flat_idx)),
                        DualSimd::new(f32x4::splat(val[1]), get_grad(flat_idx + 1)),
                        DualSimd::new(f32x4::splat(val[2]), get_grad(flat_idx + 2)),
                        DualSimd::new(f32x4::splat(val[3]), get_grad(flat_idx + 3)),
                    ];
                }
                ctx.sp_m2 += 1;
            },
            Instruction::LoadVarM3(idx) => unsafe {
                let i = *idx as usize;
                *ctx.stack_m3.get_unchecked_mut(ctx.sp_m3) = [
                    DualSimd::constant(*features.get_unchecked(i)),
                    DualSimd::constant(*features.get_unchecked(i + 1)),
                    DualSimd::constant(*features.get_unchecked(i + 2)),
                    DualSimd::constant(*features.get_unchecked(i + 3)),
                    DualSimd::constant(*features.get_unchecked(i + 4)),
                    DualSimd::constant(*features.get_unchecked(i + 5)),
                    DualSimd::constant(*features.get_unchecked(i + 6)),
                    DualSimd::constant(*features.get_unchecked(i + 7)),
                    DualSimd::constant(*features.get_unchecked(i + 8)),
                ];
                ctx.sp_m3 += 1;
            },
            Instruction::LoadConstM3(idx) => unsafe {
                if let Scalar::Mat3(val) = constants.get_unchecked(*idx as usize) {
                    let flat_idx = get_flat_start_idx(*idx as usize);
                    *ctx.stack_m3.get_unchecked_mut(ctx.sp_m3) = [
                        DualSimd::new(f32x4::splat(val[0]), get_grad(flat_idx)),
                        DualSimd::new(f32x4::splat(val[1]), get_grad(flat_idx + 1)),
                        DualSimd::new(f32x4::splat(val[2]), get_grad(flat_idx + 2)),
                        DualSimd::new(f32x4::splat(val[3]), get_grad(flat_idx + 3)),
                        DualSimd::new(f32x4::splat(val[4]), get_grad(flat_idx + 4)),
                        DualSimd::new(f32x4::splat(val[5]), get_grad(flat_idx + 5)),
                        DualSimd::new(f32x4::splat(val[6]), get_grad(flat_idx + 6)),
                        DualSimd::new(f32x4::splat(val[7]), get_grad(flat_idx + 7)),
                        DualSimd::new(f32x4::splat(val[8]), get_grad(flat_idx + 8)),
                    ];
                }
                ctx.sp_m3 += 1;
            },

            Instruction::Basic(b) => unsafe {
                match b {
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
            },

            Instruction::Linalg(l) => unsafe {
                match l {
                    LinalgOpCode::MakeVec2 => autodiff::eval_make_dual_vec2(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_v2, &mut ctx.stack_v2),
                    LinalgOpCode::MakeVec3 => autodiff::eval_make_dual_vec3(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_v3, &mut ctx.stack_v3),
                    LinalgOpCode::GetXV2 => autodiff::eval_get_x_dual_v2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v2, &ctx.stack_v2),
                    LinalgOpCode::GetYV2 => autodiff::eval_get_y_dual_v2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v2, &ctx.stack_v2),
                    LinalgOpCode::GetXV3 => autodiff::eval_get_x_dual_v3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v3, &ctx.stack_v3),
                    LinalgOpCode::GetYV3 => autodiff::eval_get_y_dual_v3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v3, &ctx.stack_v3),
                    LinalgOpCode::GetZV3 => autodiff::eval_get_z_dual_v3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v3, &ctx.stack_v3),
                    
                    LinalgOpCode::AddV2 => autodiff::eval_add_dual_v2(&mut ctx.sp_v2, &mut ctx.stack_v2),
                    LinalgOpCode::SubV2 => autodiff::eval_sub_dual_v2(&mut ctx.sp_v2, &mut ctx.stack_v2),
                    LinalgOpCode::ScaleV2 => autodiff::eval_scale_dual_v2(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_v2, &mut ctx.stack_v2),
                    LinalgOpCode::DotV2 => autodiff::eval_dot_dual_v2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v2, &ctx.stack_v2),
                    LinalgOpCode::NormV2 => autodiff::eval_norm_dual_v2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v2, &ctx.stack_v2),
                    
                    LinalgOpCode::AddV3 => autodiff::eval_add_dual_v3(&mut ctx.sp_v3, &mut ctx.stack_v3),
                    LinalgOpCode::SubV3 => autodiff::eval_sub_dual_v3(&mut ctx.sp_v3, &mut ctx.stack_v3),
                    LinalgOpCode::ScaleV3 => autodiff::eval_scale_dual_v3(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_v3, &mut ctx.stack_v3),
                    LinalgOpCode::DotV3 => autodiff::eval_dot_dual_v3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v3, &ctx.stack_v3),
                    LinalgOpCode::NormV3 => autodiff::eval_norm_dual_v3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v3, &ctx.stack_v3),
                    LinalgOpCode::CrossV3 => autodiff::eval_cross_dual_v3(&mut ctx.sp_v3, &mut ctx.stack_v3),

                    LinalgOpCode::MakeMat2 => autodiff::eval_make_dual_mat2(&mut ctx.sp_v2, &ctx.stack_v2, &mut ctx.sp_m2, &mut ctx.stack_m2),
                    LinalgOpCode::AddM2 => autodiff::eval_add_dual_m2(&mut ctx.sp_m2, &mut ctx.stack_m2),
                    LinalgOpCode::SubM2 => autodiff::eval_sub_dual_m2(&mut ctx.sp_m2, &mut ctx.stack_m2),
                    LinalgOpCode::ScaleM2 => autodiff::eval_scale_dual_m2(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_m2, &mut ctx.stack_m2),
                    LinalgOpCode::MulM2 => autodiff::eval_mul_dual_m2(&mut ctx.sp_m2, &mut ctx.stack_m2),
                    LinalgOpCode::MulM2V2 => autodiff::eval_mul_dual_m2v2(&mut ctx.sp_m2, &ctx.stack_m2, &mut ctx.sp_v2, &mut ctx.stack_v2),
                    LinalgOpCode::DetM2 => autodiff::eval_det_dual_m2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m2, &ctx.stack_m2),
                    LinalgOpCode::TraceM2 => autodiff::eval_trace_dual_m2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m2, &ctx.stack_m2),
                    LinalgOpCode::TransposeM2 => autodiff::eval_transpose_dual_m2(&mut ctx.sp_m2, &mut ctx.stack_m2),

                    LinalgOpCode::MakeMat3 => autodiff::eval_make_dual_mat3(&mut ctx.sp_v3, &ctx.stack_v3, &mut ctx.sp_m3, &mut ctx.stack_m3),
                    LinalgOpCode::AddM3 => autodiff::eval_add_dual_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                    LinalgOpCode::SubM3 => autodiff::eval_sub_dual_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                    LinalgOpCode::ScaleM3 => autodiff::eval_scale_dual_m3(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_m3, &mut ctx.stack_m3),
                    LinalgOpCode::MulM3 => autodiff::eval_mul_dual_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                    LinalgOpCode::MulM3V3 => autodiff::eval_mul_dual_m3v3(&mut ctx.sp_m3, &ctx.stack_m3, &mut ctx.sp_v3, &mut ctx.stack_v3),
                    LinalgOpCode::DetM3 => autodiff::eval_det_dual_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                    LinalgOpCode::TraceM3 => autodiff::eval_trace_dual_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                    LinalgOpCode::TransposeM3 => autodiff::eval_transpose_dual_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                    
                    _ => {}
                }
            },
            _ => {}
        }
    }

    unsafe { *ctx.stack_f.get_unchecked(0) }
}