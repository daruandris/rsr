use crate::engine::eval::autodiff::DualSimd;
use wide::f32x4;

pub struct VmState {
    pub stack_f: [f32x4; 32],
    pub sp_f: usize,

    pub stack_v2: [[f32x4; 2]; 32],
    pub sp_v2: usize,

    pub stack_v3: [[f32x4; 3]; 32],
    pub sp_v3: usize,

    pub stack_m2: [[f32x4; 4]; 32],
    pub sp_m2: usize,

    pub stack_m3: [[f32x4; 9]; 32],
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
