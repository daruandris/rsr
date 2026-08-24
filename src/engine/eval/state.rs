//! Virtual Machine states for evaluation and automatic differentiation.

use crate::engine::eval::autodiff::DualSimd;
use wide::f32x4;

/// The execution context for the SIMD virtual machine.
///
/// `VmState` provides pre-allocated, fixed-size stacks for different data types
/// (floats, vectors, matrices) to ensure zero-cost evaluation without heap allocations.
/// It operates on `f32x4` SIMD vectors, meaning it processes 4 data rows simultaneously
/// in a single CPU instruction set loop.
pub struct VmState {
    /// The evaluation stack for scalar float batches.
    pub stack_f: [f32x4; 32],
    /// The stack pointer for the float stack.
    pub sp_f: usize,

    /// The evaluation stack for 2D vectors.
    pub stack_v2: [[f32x4; 2]; 32],
    /// The stack pointer for the 2D vector stack.
    pub sp_v2: usize,

    /// The evaluation stack for 3D vectors.
    pub stack_v3: [[f32x4; 3]; 32],
    /// The stack pointer for the 3D vector stack.
    pub sp_v3: usize,

    /// The evaluation stack for 2x2 matrices.
    pub stack_m2: [[f32x4; 4]; 32],
    /// The stack pointer for the 2x2 matrix stack.
    pub sp_m2: usize,

    /// The evaluation stack for 3x3 matrices.
    pub stack_m3: [[f32x4; 9]; 32],
    /// The stack pointer for the 3x3 matrix stack.
    pub sp_m3: usize,
}

impl Default for VmState {
    fn default() -> Self {
        Self::new()
    }
}

impl VmState {
    pub fn new() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
}

/// The execution context for the SIMD virtual machine with Dual numbers.
///
/// Similar to [`VmState`], but operates on [`DualSimd`] values instead of primitive floats.
/// This state is used internally during forward-mode Automatic Differentiation (AD)
/// to compute exact gradients alongside the Mean Squared Error.
pub struct DualVmState {
    pub stack_f: [DualSimd; 32],
    pub sp_f: usize,
    pub stack_v2: [[DualSimd; 2]; 32],
    pub sp_v2: usize,
    pub stack_v3: [[DualSimd; 3]; 32],
    pub sp_v3: usize,
    pub stack_m2: [[DualSimd; 4]; 32],
    pub sp_m2: usize,
    pub stack_m3: [[DualSimd; 9]; 32],
    pub sp_m3: usize,
}

impl Default for DualVmState {
    fn default() -> Self {
        Self::new()
    }
}

impl DualVmState {
    pub fn new() -> Self {
        unsafe { std::mem::MaybeUninit::zeroed().assume_init() }
    }
}
