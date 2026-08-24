//! The core execution engine for evaluating compiled programs.
//!
//! This module provides the functions to execute `Program` instances over SIMD-aligned
//! datasets, calculating both predictions and exact gradients using forward-mode AD.
use crate::Instruction;
use crate::SymbolicEngine;
use crate::engine::data::dataset::Dataset;
use crate::engine::eval::autodiff::DualSimd;
use crate::engine::eval::scalar::Scalar;
use crate::engine::eval::state::{DualVmState, VmState};
use crate::engine::expr::program::Program;
use crate::engine::optimize::Parameterized;
use wide::f32x4;

/// Evaluates a compiled program on a single batch of SIMD features.
///
/// This function acts as the main virtual machine loop, processing instructions
/// sequentially and operating entirely on pre-allocated stacks.
#[inline(always)]
pub fn eval_simd(program: &Program, features: &[f32x4]) -> f32x4 {
    let mut ctx = VmState::new();
    let constants = &program.constants;

    for op in &program.code {
        match op {
            Instruction::LoadVarF(idx) =>
            // SAFETY: The AST compiler ensures that `idx` is strictly less than the number
            // of features. It also guarantees `ctx.sp_f` will not exceed the stack capacity (32).
            unsafe {
                *ctx.stack_f.get_unchecked_mut(ctx.sp_f) = *features.get_unchecked(*idx as usize);
                ctx.sp_f += 1;
            },
            Instruction::LoadConstF(idx) =>
            // SAFETY: The compilation phase registers all constants, ensuring `idx` is within
            // bounds of the `constants` array. Stack capacity is guaranteed by AST structural limits.
            unsafe {
                if let Scalar::Float(val) = constants.get_unchecked(*idx as usize) {
                    *ctx.stack_f.get_unchecked_mut(ctx.sp_f) = f32x4::splat(*val);
                }
                ctx.sp_f += 1;
            },
            Instruction::LoadVarV2(idx) =>
            // SAFETY: Index `i` and `i+1` are bounds-checked during schema validation.
            // Stack capacity prevents overflow.
            unsafe {
                let i = *idx as usize;
                *ctx.stack_v2.get_unchecked_mut(ctx.sp_v2) =
                    [*features.get_unchecked(i), *features.get_unchecked(i + 1)];
                ctx.sp_v2 += 1;
            },
            Instruction::LoadConstV2(idx) =>
            // SAFETY: Constant arrays are pre-filled, index bound is guaranteed by compiler.
            unsafe {
                if let Scalar::Vec2(val) = constants.get_unchecked(*idx as usize) {
                    *ctx.stack_v2.get_unchecked_mut(ctx.sp_v2) =
                        [f32x4::splat(val[0]), f32x4::splat(val[1])];
                }
                ctx.sp_v2 += 1;
            },
            Instruction::LoadVarV3(idx) =>
            // SAFETY: Feature boundaries and stack capacity are strictly preserved by compilation.
            unsafe {
                let i = *idx as usize;
                *ctx.stack_v3.get_unchecked_mut(ctx.sp_v3) = [
                    *features.get_unchecked(i),
                    *features.get_unchecked(i + 1),
                    *features.get_unchecked(i + 2),
                ];
                ctx.sp_v3 += 1;
            },
            Instruction::LoadConstV3(idx) =>
            // SAFETY: Constant loading respects the `constants` vector bounds.
            unsafe {
                if let Scalar::Vec3(val) = constants.get_unchecked(*idx as usize) {
                    *ctx.stack_v3.get_unchecked_mut(ctx.sp_v3) = [
                        f32x4::splat(val[0]),
                        f32x4::splat(val[1]),
                        f32x4::splat(val[2]),
                    ];
                }
                ctx.sp_v3 += 1;
            },
            Instruction::LoadVarM2(idx) =>
            // SAFETY: Memory layout for Mat2 features guarantees up to `i+3` is safe.
            unsafe {
                let i = *idx as usize;
                *ctx.stack_m2.get_unchecked_mut(ctx.sp_m2) = [
                    *features.get_unchecked(i),
                    *features.get_unchecked(i + 1),
                    *features.get_unchecked(i + 2),
                    *features.get_unchecked(i + 3),
                ];
                ctx.sp_m2 += 1;
            },
            Instruction::LoadConstM2(idx) =>
            // SAFETY: Verified constant load.
            unsafe {
                if let Scalar::Mat2(val) = constants.get_unchecked(*idx as usize) {
                    *ctx.stack_m2.get_unchecked_mut(ctx.sp_m2) = [
                        f32x4::splat(val[0]),
                        f32x4::splat(val[1]),
                        f32x4::splat(val[2]),
                        f32x4::splat(val[3]),
                    ];
                }
                ctx.sp_m2 += 1;
            },
            Instruction::LoadVarM3(idx) =>
            // SAFETY: Memory layout for Mat3 features guarantees up to `i+8` is safe.
            unsafe {
                let i = *idx as usize;
                *ctx.stack_m3.get_unchecked_mut(ctx.sp_m3) = [
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
                ctx.sp_m3 += 1;
            },
            Instruction::LoadConstM3(idx) =>
            // SAFETY: Verified constant load.
            unsafe {
                if let Scalar::Mat3(val) = constants.get_unchecked(*idx as usize) {
                    *ctx.stack_m3.get_unchecked_mut(ctx.sp_m3) = [
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
                ctx.sp_m3 += 1;
            },

            _ => SymbolicEngine::eval_single(*op, &mut ctx),
        }
    }
    // SAFETY: The expression tree structure guarantees that exactly one scalar result
    // remains on the float stack at index 0 upon completion.
    unsafe { *ctx.stack_f.get_unchecked(0) }
}

/// Computes the Mean Squared Error (MSE) of a program over the entire dataset.
pub fn compute_mse(program: &Program, dataset: &Dataset) -> f32 {
    let mut sum_squared_error = f32x4::splat(0.0);
    let num_features = dataset.num_features as usize;
    let flat_features = &dataset.feature_flat;
    let targets = &dataset.target_batches;

    for i in 0..dataset.num_batches {
        let start = i * num_features;

        // SAFETY: The Dataset initialization guarantees that `feature_flat` contains exactly
        // `num_batches * num_features` elements. `targets` has exactly `num_batches`.
        let input_batch = unsafe { flat_features.get_unchecked(start..start + num_features) };
        let prediction = eval_simd(program, input_batch);
        let target = unsafe { *targets.get_unchecked(i) };

        let diff = prediction - target;
        sum_squared_error += diff * diff;
    }

    let mse = sum_squared_error.reduce_add() / (dataset.num_samples as f32);
    if !mse.is_finite() { f32::MAX } else { mse }
}

/// Computes both the MSE and the gradient of the MSE with respect to the program's constants.
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

        // SAFETY: Ensured by the memory layout structure computed during `Dataset` creation.
        let input_batch = unsafe { flat_features.get_unchecked(start..start + num_features) };
        let target = unsafe { *targets.get_unchecked(i) };

        let mut diff = f32x4::splat(0.0);

        for (k, item) in grad_sum.iter_mut().enumerate().take(active_params_count) {
            let dual_result = eval_simd_dual(program, input_batch, k);

            if k == 0 {
                diff = dual_result.val - target;
                sum_squared_error += diff * diff;
            }
            *item += f32x4::splat(2.0) * diff * dual_result.grad;
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

/// Evaluates a compiled program using forward-mode automatic differentiation.
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
            Instruction::LoadVarF(idx) =>
            // SAFETY: AST guarantees `idx` is within dataset bounds.
            unsafe {
                *ctx.stack_f.get_unchecked_mut(ctx.sp_f) =
                    DualSimd::constant(*features.get_unchecked(*idx as usize));
                ctx.sp_f += 1;
            },
            Instruction::LoadConstF(idx) =>
            // SAFETY: Constant loading respects bounds checked at compile time.
            unsafe {
                if let Scalar::Float(val) = constants.get_unchecked(*idx as usize) {
                    let flat_idx = get_flat_start_idx(*idx as usize);
                    *ctx.stack_f.get_unchecked_mut(ctx.sp_f) =
                        DualSimd::new(f32x4::splat(*val), get_grad(flat_idx));
                }
                ctx.sp_f += 1;
            },
            Instruction::LoadVarV2(idx) =>
            // SAFETY: Layout is ensured by feature vector limits.
            unsafe {
                let i = *idx as usize;
                *ctx.stack_v2.get_unchecked_mut(ctx.sp_v2) = [
                    DualSimd::constant(*features.get_unchecked(i)),
                    DualSimd::constant(*features.get_unchecked(i + 1)),
                ];
                ctx.sp_v2 += 1;
            },
            Instruction::LoadConstV2(idx) =>
            // SAFETY: Layout is ensured by constant array mapping.
            unsafe {
                if let Scalar::Vec2(val) = constants.get_unchecked(*idx as usize) {
                    let flat_idx = get_flat_start_idx(*idx as usize);
                    *ctx.stack_v2.get_unchecked_mut(ctx.sp_v2) = [
                        DualSimd::new(f32x4::splat(val[0]), get_grad(flat_idx)),
                        DualSimd::new(f32x4::splat(val[1]), get_grad(flat_idx + 1)),
                    ];
                }
                ctx.sp_v2 += 1;
            },
            Instruction::LoadVarV3(idx) =>
            // SAFETY: Stack and dataset boundary checks verified during AST load.
            unsafe {
                let i = *idx as usize;
                *ctx.stack_v3.get_unchecked_mut(ctx.sp_v3) = [
                    DualSimd::constant(*features.get_unchecked(i)),
                    DualSimd::constant(*features.get_unchecked(i + 1)),
                    DualSimd::constant(*features.get_unchecked(i + 2)),
                ];
                ctx.sp_v3 += 1;
            },
            Instruction::LoadConstV3(idx) =>
            // SAFETY: Verifed continuous block load for constant components.
            unsafe {
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
            Instruction::LoadVarM2(idx) =>
            // SAFETY: Linear dataset boundaries allow for safe matrix data fetch.
            unsafe {
                let i = *idx as usize;
                *ctx.stack_m2.get_unchecked_mut(ctx.sp_m2) = [
                    DualSimd::constant(*features.get_unchecked(i)),
                    DualSimd::constant(*features.get_unchecked(i + 1)),
                    DualSimd::constant(*features.get_unchecked(i + 2)),
                    DualSimd::constant(*features.get_unchecked(i + 3)),
                ];
                ctx.sp_m2 += 1;
            },
            Instruction::LoadConstM2(idx) =>
            // SAFETY: Verified contiguous fetch from registered constants.
            unsafe {
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
            Instruction::LoadVarM3(idx) =>
            // SAFETY: Bounds verified before structural compilation.
            unsafe {
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
            Instruction::LoadConstM3(idx) =>
            // SAFETY: Valid bounds mapping to scalar flattened arrays.
            unsafe {
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
            _ => crate::SymbolicEngine::eval_dual_single(*op, &mut ctx),
        }
    }

    // SAFETY: AST ensures exactly one DualSimd float value represents the root answer.
    unsafe { *ctx.stack_f.get_unchecked(0) }
}
