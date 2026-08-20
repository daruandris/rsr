use crate::engine::domain::{Domain, SimplifyAction};
use crate::engine::eval::scalar::Scalar;
use crate::engine::eval::state::{VmState, DualVmState};
use crate::engine::eval::autodiff;

mod eval;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinalgOpCode {
    MakeVec2, MakeVec3,
    GetXV2, GetYV2, 
    GetXV3, GetYV3, GetZV3,
    
    AddV2, SubV2, ScaleV2, DotV2, NormV2,
    AddV3, SubV3, ScaleV3, DotV3, NormV3, CrossV3,
    
    MakeMat2, AddM2, SubM2, ScaleM2, MulM2, MulM2V2, 
    DetM2, TraceM2, TransposeM2, InverseM2,
    
    MakeMat3, AddM3, SubM3, ScaleM3, MulM3, MulM3V3, 
    DetM3, TraceM3, TransposeM3, InverseM3,
}

pub struct LinalgDomain;

impl Domain for LinalgDomain {
    type OpCode = LinalgOpCode;

    #[inline(always)]
    fn eval(op: Self::OpCode, ctx: &mut VmState) {
        unsafe {
            match op {
                LinalgOpCode::MakeVec2 => eval::eval_make_vec2(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_v2, &mut ctx.stack_v2),
                LinalgOpCode::MakeVec3 => eval::eval_make_vec3(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_v3, &mut ctx.stack_v3),
                LinalgOpCode::GetXV2 => eval::eval_get_x_v2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v2, &ctx.stack_v2),
                LinalgOpCode::GetYV2 => eval::eval_get_y_v2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v2, &ctx.stack_v2),
                LinalgOpCode::GetXV3 => eval::eval_get_x_v3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v3, &ctx.stack_v3),
                LinalgOpCode::GetYV3 => eval::eval_get_y_v3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v3, &ctx.stack_v3),
                LinalgOpCode::GetZV3 => eval::eval_get_z_v3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v3, &ctx.stack_v3),

                LinalgOpCode::AddV2 => eval::eval_add_v2(&mut ctx.sp_v2, &mut ctx.stack_v2),
                LinalgOpCode::SubV2 => eval::eval_sub_v2(&mut ctx.sp_v2, &mut ctx.stack_v2),
                LinalgOpCode::ScaleV2 => eval::eval_scale_v2(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_v2, &mut ctx.stack_v2),
                LinalgOpCode::DotV2 => eval::eval_dot_v2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v2, &ctx.stack_v2),
                LinalgOpCode::NormV2 => eval::eval_norm_v2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v2, &ctx.stack_v2),

                LinalgOpCode::AddV3 => eval::eval_add_v3(&mut ctx.sp_v3, &mut ctx.stack_v3),
                LinalgOpCode::SubV3 => eval::eval_sub_v3(&mut ctx.sp_v3, &mut ctx.stack_v3),
                LinalgOpCode::ScaleV3 => eval::eval_scale_v3(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_v3, &mut ctx.stack_v3),
                LinalgOpCode::DotV3 => eval::eval_dot_v3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v3, &ctx.stack_v3),
                LinalgOpCode::NormV3 => eval::eval_norm_v3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_v3, &ctx.stack_v3),
                LinalgOpCode::CrossV3 => eval::eval_cross_v3(&mut ctx.sp_v3, &mut ctx.stack_v3),

                LinalgOpCode::MakeMat2 => eval::eval_make_mat2(&mut ctx.sp_v2, &ctx.stack_v2, &mut ctx.sp_m2, &mut ctx.stack_m2),
                LinalgOpCode::AddM2 => eval::eval_add_m2(&mut ctx.sp_m2, &mut ctx.stack_m2),
                LinalgOpCode::SubM2 => eval::eval_sub_m2(&mut ctx.sp_m2, &mut ctx.stack_m2),
                LinalgOpCode::ScaleM2 => eval::eval_scale_m2(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_m2, &mut ctx.stack_m2),
                LinalgOpCode::MulM2 => eval::eval_mul_m2(&mut ctx.sp_m2, &mut ctx.stack_m2),
                LinalgOpCode::MulM2V2 => eval::eval_mul_m2v2(&mut ctx.sp_m2, &ctx.stack_m2, &mut ctx.sp_v2, &mut ctx.stack_v2),
                LinalgOpCode::DetM2 => eval::eval_det_m2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m2, &ctx.stack_m2),
                LinalgOpCode::TraceM2 => eval::eval_trace_m2(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m2, &ctx.stack_m2),
                LinalgOpCode::TransposeM2 => eval::eval_transpose_m2(&mut ctx.sp_m2, &mut ctx.stack_m2),
                LinalgOpCode::InverseM2 => eval::eval_inverse_m2(&mut ctx.sp_m2, &mut ctx.stack_m2),

                LinalgOpCode::MakeMat3 => eval::eval_make_mat3(&mut ctx.sp_v3, &ctx.stack_v3, &mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::AddM3 => eval::eval_add_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::SubM3 => eval::eval_sub_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::ScaleM3 => eval::eval_scale_m3(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::MulM3 => eval::eval_mul_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::MulM3V3 => eval::eval_mul_m3v3(&mut ctx.sp_m3, &ctx.stack_m3, &mut ctx.sp_v3, &mut ctx.stack_v3),
                LinalgOpCode::DetM3 => eval::eval_det_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                LinalgOpCode::TraceM3 => eval::eval_trace_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                LinalgOpCode::TransposeM3 => eval::eval_transpose_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::InverseM3 => eval::eval_inverse_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
            }
        }
    }

    fn try_simplify(op: Self::OpCode, const_vals: &[Option<Scalar>], args_equal: bool) -> SimplifyAction {
        let all_const = const_vals.iter().all(|c| c.is_some());
        if all_const && !const_vals.is_empty() {
            if let Some(folded) = fold_linalg_constants(op, const_vals) {
                return SimplifyAction::ReplaceWithConstant(folded);
            }
        }

        if const_vals.len() == 2 {
            let a_is_zero = const_vals[0].as_ref().map_or(false, |c| c.is_zero());
            let b_is_zero = const_vals[1].as_ref().map_or(false, |c| c.is_zero());
            let a_is_one = const_vals[0].as_ref().map_or(false, |c| c.is_one());

            match op {
                LinalgOpCode::AddV2 | LinalgOpCode::AddV3 | LinalgOpCode::AddM2 | LinalgOpCode::AddM3 => {
                    if b_is_zero { return SimplifyAction::KeepArg(0); }
                    if a_is_zero { return SimplifyAction::KeepArg(1); }
                }
                LinalgOpCode::SubV2 | LinalgOpCode::SubV3 | LinalgOpCode::SubM2 | LinalgOpCode::SubM3 => {
                    if b_is_zero { return SimplifyAction::KeepArg(0); }
                    if args_equal { return SimplifyAction::ReplaceWithConstant(zero_for_op(op)); }
                }
                LinalgOpCode::ScaleV2 | LinalgOpCode::ScaleV3 | LinalgOpCode::ScaleM2 | LinalgOpCode::ScaleM3 => {
                    if a_is_zero { return SimplifyAction::ReplaceWithConstant(zero_for_op(op)); }
                    if a_is_one { return SimplifyAction::KeepArg(1); }
                }
                LinalgOpCode::DotV2 | LinalgOpCode::DotV3 => {
                    if a_is_zero || b_is_zero { return SimplifyAction::ReplaceWithConstant(Scalar::Float(0.0)); }
                }
                LinalgOpCode::CrossV3 => {
                    if a_is_zero || b_is_zero || args_equal {
                        return SimplifyAction::ReplaceWithConstant(Scalar::Vec3([0.0; 3]));
                    }
                }
                LinalgOpCode::MulM2 | LinalgOpCode::MulM3 => {
                    let a_is_ident = const_vals[0].as_ref().map_or(false, |c| c.is_identity());
                    let b_is_ident = const_vals[1].as_ref().map_or(false, |c| c.is_identity());
                    if a_is_ident { return SimplifyAction::KeepArg(1); }
                    if b_is_ident { return SimplifyAction::KeepArg(0); }
                    if a_is_zero || b_is_zero { return SimplifyAction::ReplaceWithConstant(zero_for_op(op)); }
                }
                LinalgOpCode::MulM2V2 | LinalgOpCode::MulM3V3 => {
                    let a_is_ident = const_vals[0].as_ref().map_or(false, |c| c.is_identity());
                    if a_is_ident { return SimplifyAction::KeepArg(1); }
                    if a_is_zero || b_is_zero { return SimplifyAction::ReplaceWithConstant(zero_for_op(op)); }
                }
                _ => {}
            }
        }
        SimplifyAction::None
    }

    fn arity(op: Self::OpCode) -> usize {
        match op {
            LinalgOpCode::MakeVec3 | LinalgOpCode::MakeMat3 => 3,
            LinalgOpCode::MakeVec2 | LinalgOpCode::MakeMat2 |
            LinalgOpCode::AddV2 | LinalgOpCode::SubV2 | LinalgOpCode::ScaleV2 | LinalgOpCode::DotV2 |
            LinalgOpCode::AddV3 | LinalgOpCode::SubV3 | LinalgOpCode::ScaleV3 | LinalgOpCode::DotV3 | LinalgOpCode::CrossV3 |
            LinalgOpCode::AddM2 | LinalgOpCode::SubM2 | LinalgOpCode::ScaleM2 | LinalgOpCode::MulM2 | LinalgOpCode::MulM2V2 |
            LinalgOpCode::AddM3 | LinalgOpCode::SubM3 | LinalgOpCode::ScaleM3 | LinalgOpCode::MulM3 | LinalgOpCode::MulM3V3 => 2,
            _ => 1,
        }
    }

    fn return_type(op: Self::OpCode) -> crate::engine::eval::types::ValueType {
        use crate::engine::eval::types::ValueType;
        match op {
            LinalgOpCode::GetXV2 | LinalgOpCode::GetYV2 | LinalgOpCode::DotV2 | LinalgOpCode::NormV2 |
            LinalgOpCode::GetXV3 | LinalgOpCode::GetYV3 | LinalgOpCode::GetZV3 | LinalgOpCode::DotV3 | LinalgOpCode::NormV3 |
            LinalgOpCode::DetM2 | LinalgOpCode::TraceM2 | LinalgOpCode::DetM3 | LinalgOpCode::TraceM3 => ValueType::Float,
            
            LinalgOpCode::MakeVec2 | LinalgOpCode::AddV2 | LinalgOpCode::SubV2 | LinalgOpCode::ScaleV2 | LinalgOpCode::MulM2V2 => ValueType::Vec2,
            LinalgOpCode::MakeVec3 | LinalgOpCode::AddV3 | LinalgOpCode::SubV3 | LinalgOpCode::ScaleV3 | LinalgOpCode::CrossV3 | LinalgOpCode::MulM3V3 => ValueType::Vec3,
            LinalgOpCode::MakeMat2 | LinalgOpCode::AddM2 | LinalgOpCode::SubM2 | LinalgOpCode::ScaleM2 | LinalgOpCode::MulM2 | LinalgOpCode::TransposeM2 | LinalgOpCode::InverseM2 => ValueType::Mat2,
            LinalgOpCode::MakeMat3 | LinalgOpCode::AddM3 | LinalgOpCode::SubM3 | LinalgOpCode::ScaleM3 | LinalgOpCode::MulM3 | LinalgOpCode::TransposeM3 | LinalgOpCode::InverseM3 => ValueType::Mat3,
        }
    }

    fn expected_types(op: Self::OpCode) -> &'static [crate::engine::eval::types::ValueType] {
        use crate::engine::eval::types::ValueType;
        match op {
            LinalgOpCode::MakeVec2 => &[ValueType::Float, ValueType::Float],
            LinalgOpCode::GetXV2 | LinalgOpCode::GetYV2 | LinalgOpCode::NormV2 => &[ValueType::Vec2],
            LinalgOpCode::AddV2 | LinalgOpCode::SubV2 | LinalgOpCode::DotV2 => &[ValueType::Vec2, ValueType::Vec2],
            LinalgOpCode::ScaleV2 => &[ValueType::Float, ValueType::Vec2],
            LinalgOpCode::MakeVec3 => &[ValueType::Float, ValueType::Float, ValueType::Float],
            LinalgOpCode::GetXV3 | LinalgOpCode::GetYV3 | LinalgOpCode::GetZV3 | LinalgOpCode::NormV3 => &[ValueType::Vec3],
            LinalgOpCode::AddV3 | LinalgOpCode::SubV3 | LinalgOpCode::DotV3 | LinalgOpCode::CrossV3 => &[ValueType::Vec3, ValueType::Vec3],
            LinalgOpCode::ScaleV3 => &[ValueType::Float, ValueType::Vec3],
            LinalgOpCode::MakeMat2 => &[ValueType::Vec2, ValueType::Vec2],
            LinalgOpCode::AddM2 | LinalgOpCode::SubM2 | LinalgOpCode::MulM2 => &[ValueType::Mat2, ValueType::Mat2],
            LinalgOpCode::ScaleM2 => &[ValueType::Float, ValueType::Mat2],
            LinalgOpCode::MulM2V2 => &[ValueType::Mat2, ValueType::Vec2],
            LinalgOpCode::DetM2 | LinalgOpCode::TraceM2 | LinalgOpCode::TransposeM2 | LinalgOpCode::InverseM2 => &[ValueType::Mat2],
            LinalgOpCode::MakeMat3 => &[ValueType::Vec3, ValueType::Vec3, ValueType::Vec3],
            LinalgOpCode::AddM3 | LinalgOpCode::SubM3 | LinalgOpCode::MulM3 => &[ValueType::Mat3, ValueType::Mat3],
            LinalgOpCode::ScaleM3 => &[ValueType::Float, ValueType::Mat3],
            LinalgOpCode::MulM3V3 => &[ValueType::Mat3, ValueType::Vec3],
            LinalgOpCode::DetM3 | LinalgOpCode::TraceM3 | LinalgOpCode::TransposeM3 | LinalgOpCode::InverseM3 => &[ValueType::Mat3],
        }
    }

    fn weight(op: Self::OpCode) -> usize {
        match op {
            LinalgOpCode::AddV2 | LinalgOpCode::SubV2 | LinalgOpCode::AddV3 | LinalgOpCode::SubV3 | 
            LinalgOpCode::AddM2 | LinalgOpCode::SubM2 | LinalgOpCode::AddM3 | LinalgOpCode::SubM3 => 1,
            
            LinalgOpCode::MakeVec2 | LinalgOpCode::GetXV2 | LinalgOpCode::GetYV2 | LinalgOpCode::ScaleV2 | LinalgOpCode::DotV2 | 
            LinalgOpCode::MakeVec3 | LinalgOpCode::GetXV3 | LinalgOpCode::GetYV3 | LinalgOpCode::GetZV3 | LinalgOpCode::ScaleV3 | LinalgOpCode::DotV3 | 
            LinalgOpCode::ScaleM2 | LinalgOpCode::ScaleM3 => 2,
            
            LinalgOpCode::NormV2 | LinalgOpCode::NormV3 | LinalgOpCode::MakeMat2 | LinalgOpCode::MulM2 | LinalgOpCode::MulM2V2 | 
            LinalgOpCode::TraceM2 | LinalgOpCode::TransposeM2 | LinalgOpCode::MulM3V3 | LinalgOpCode::TraceM3 | LinalgOpCode::TransposeM3 => 3,
            
            LinalgOpCode::CrossV3 | LinalgOpCode::DetM2 | LinalgOpCode::InverseM2 | LinalgOpCode::MulM3 => 4,
            LinalgOpCode::MakeMat3 | LinalgOpCode::DetM3 => 5,
            LinalgOpCode::InverseM3 => 6,
        }
    }

    fn is_forbidden_child(parent: Self::OpCode, child: Self::OpCode) -> bool {
        match parent {
            LinalgOpCode::TransposeM2 => matches!(child, LinalgOpCode::TransposeM2),
            LinalgOpCode::TransposeM3 => matches!(child, LinalgOpCode::TransposeM3),
            LinalgOpCode::InverseM2 => matches!(child, LinalgOpCode::InverseM2),
            LinalgOpCode::InverseM3 => matches!(child, LinalgOpCode::InverseM3),
            LinalgOpCode::GetXV2 | LinalgOpCode::GetYV2 => matches!(child, LinalgOpCode::MakeVec2),
            LinalgOpCode::GetXV3 | LinalgOpCode::GetYV3 | LinalgOpCode::GetZV3 => matches!(child, LinalgOpCode::MakeVec3),
            _ => false,
        }
    }

    fn is_differentiable(op: Self::OpCode) -> bool {
        !matches!(op, LinalgOpCode::InverseM2 | LinalgOpCode::InverseM3)
    }

    fn requires_cmaes(_op: Self::OpCode) -> bool { 
        false 
    }

    #[inline(always)]
    fn eval_dual(op: Self::OpCode, ctx: &mut DualVmState) {
        unsafe {
            match op {
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
                LinalgOpCode::InverseM2 => {},

                LinalgOpCode::MakeMat3 => autodiff::eval_make_dual_mat3(&mut ctx.sp_v3, &ctx.stack_v3, &mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::AddM3 => autodiff::eval_add_dual_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::SubM3 => autodiff::eval_sub_dual_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::ScaleM3 => autodiff::eval_scale_dual_m3(&mut ctx.sp_f, &ctx.stack_f, &mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::MulM3 => autodiff::eval_mul_dual_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::MulM3V3 => autodiff::eval_mul_dual_m3v3(&mut ctx.sp_m3, &ctx.stack_m3, &mut ctx.sp_v3, &mut ctx.stack_v3),
                LinalgOpCode::DetM3 => autodiff::eval_det_dual_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                LinalgOpCode::TraceM3 => autodiff::eval_trace_dual_m3(&mut ctx.sp_f, &mut ctx.stack_f, &mut ctx.sp_m3, &ctx.stack_m3),
                LinalgOpCode::TransposeM3 => autodiff::eval_transpose_dual_m3(&mut ctx.sp_m3, &mut ctx.stack_m3),
                LinalgOpCode::InverseM3 => {},
            }
        }
    }

    fn format_op(op: Self::OpCode, args: &[String]) -> String {
        match op {
            LinalgOpCode::MakeVec2 => format!("({}, {})", args[0], args[1]),
            LinalgOpCode::MakeVec3 => format!("({}, {}, {})", args[0], args[1], args[2]),
            LinalgOpCode::GetXV2 | LinalgOpCode::GetXV3 => format!("{}.x", args[0]),
            LinalgOpCode::GetYV2 | LinalgOpCode::GetYV3 => format!("{}.y", args[0]),
            LinalgOpCode::GetZV3 => format!("{}.z", args[0]),
            LinalgOpCode::DotV2 | LinalgOpCode::DotV3 => format!("<{} • {}>", args[0], args[1]),
            LinalgOpCode::CrossV3 => format!("({} x {})", args[0], args[1]),
            LinalgOpCode::NormV2 | LinalgOpCode::NormV3 => format!("||{}||", args[0]),
            LinalgOpCode::AddV2 | LinalgOpCode::AddV3 | LinalgOpCode::AddM2 | LinalgOpCode::AddM3 => format!("({} + {})", args[0], args[1]),
            LinalgOpCode::SubV2 | LinalgOpCode::SubV3 | LinalgOpCode::SubM2 | LinalgOpCode::SubM3 => format!("({} - {})", args[0], args[1]),
            LinalgOpCode::ScaleV2 | LinalgOpCode::ScaleV3 | LinalgOpCode::ScaleM2 | LinalgOpCode::ScaleM3 |
            LinalgOpCode::MulM2 | LinalgOpCode::MulM3 | LinalgOpCode::MulM2V2 | LinalgOpCode::MulM3V3 => format!("({} * {})", args[0], args[1]),
            LinalgOpCode::InverseM2 | LinalgOpCode::InverseM3 => format!("{}^-1", args[0]),
            LinalgOpCode::TransposeM2 | LinalgOpCode::TransposeM3 => format!("{}^T", args[0]),
            LinalgOpCode::DetM2 | LinalgOpCode::DetM3 => format!("det({})", args[0]),
            LinalgOpCode::TraceM2 | LinalgOpCode::TraceM3 => format!("tr({})", args[0]),
            LinalgOpCode::MakeMat2 => format!("({}, {})", args[0], args[1]),
            LinalgOpCode::MakeMat3 => format!("({}, {}, {})", args[0], args[1], args[2]),
        }
    }
}

fn zero_for_op(op: LinalgOpCode) -> Scalar {
    match op {
        LinalgOpCode::SubV2 | LinalgOpCode::ScaleV2 | LinalgOpCode::MulM2V2 => Scalar::Vec2([0.0; 2]),
        LinalgOpCode::SubV3 | LinalgOpCode::ScaleV3 | LinalgOpCode::CrossV3 | LinalgOpCode::MulM3V3 => Scalar::Vec3([0.0; 3]),
        LinalgOpCode::SubM2 | LinalgOpCode::ScaleM2 | LinalgOpCode::MulM2 => Scalar::Mat2([0.0; 4]),
        LinalgOpCode::SubM3 | LinalgOpCode::ScaleM3 | LinalgOpCode::MulM3 => Scalar::Mat3([0.0; 9]),
        _ => Scalar::Float(0.0),
    }
}

fn fold_linalg_constants(op: LinalgOpCode, const_vals: &[Option<Scalar>]) -> Option<Scalar> {
    use Scalar::*;
    let args: Vec<Scalar> = const_vals.iter().map(|c| c.unwrap()).collect();
    match (op, args.as_slice()) {
        (LinalgOpCode::MakeVec2, [Float(x), Float(y)]) => Some(Vec2([*x, *y])),
        (LinalgOpCode::MakeVec3, [Float(x), Float(y), Float(z)]) => Some(Vec3([*x, *y, *z])),
        (LinalgOpCode::GetXV2, [Vec2(v)]) => Some(Float(v[0])),
        (LinalgOpCode::GetYV2, [Vec2(v)]) => Some(Float(v[1])),
        (LinalgOpCode::GetXV3, [Vec3(v)]) => Some(Float(v[0])),
        (LinalgOpCode::GetYV3, [Vec3(v)]) => Some(Float(v[1])),
        (LinalgOpCode::GetZV3, [Vec3(v)]) => Some(Float(v[2])),
        (LinalgOpCode::AddV2, [Vec2(a), Vec2(b)]) => Some(Vec2([a[0]+b[0], a[1]+b[1]])),
        (LinalgOpCode::SubV2, [Vec2(a), Vec2(b)]) => Some(Vec2([a[0]-b[0], a[1]-b[1]])),
        (LinalgOpCode::ScaleV2, [Float(s), Vec2(v)]) => Some(Vec2([s*v[0], s*v[1]])),
        (LinalgOpCode::DotV2, [Vec2(a), Vec2(b)]) => Some(Float(a[0]*b[0] + a[1]*b[1])),
        (LinalgOpCode::NormV2, [Vec2(v)]) => Some(Float((v[0]*v[0] + v[1]*v[1]).sqrt())),
        (LinalgOpCode::AddV3, [Vec3(a), Vec3(b)]) => Some(Vec3([a[0]+b[0], a[1]+b[1], a[2]+b[2]])),
        (LinalgOpCode::SubV3, [Vec3(a), Vec3(b)]) => Some(Vec3([a[0]-b[0], a[1]-b[1], a[2]-b[2]])),
        (LinalgOpCode::ScaleV3, [Float(s), Vec3(v)]) => Some(Vec3([s*v[0], s*v[1], s*v[2]])),
        (LinalgOpCode::DotV3, [Vec3(a), Vec3(b)]) => Some(Float(a[0]*b[0] + a[1]*b[1] + a[2]*b[2])),
        (LinalgOpCode::NormV3, [Vec3(v)]) => Some(Float((v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).sqrt())),
        (LinalgOpCode::CrossV3, [Vec3(a), Vec3(b)]) => Some(Vec3([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])),
        (LinalgOpCode::MakeMat2, [Vec2(c0), Vec2(c1)]) => Some(Mat2([c0[0], c0[1], c1[0], c1[1]])),
        (LinalgOpCode::AddM2, [Mat2(a), Mat2(b)]) => Some(Mat2([a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]])),
        (LinalgOpCode::SubM2, [Mat2(a), Mat2(b)]) => Some(Mat2([a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]])),
        (LinalgOpCode::ScaleM2, [Float(s), Mat2(m)]) => Some(Mat2([s*m[0], s*m[1], s*m[2], s*m[3]])),
        (LinalgOpCode::MulM2, [Mat2(a), Mat2(b)]) => Some(Mat2([a[0]*b[0]+a[2]*b[1], a[1]*b[0]+a[3]*b[1], a[0]*b[2]+a[2]*b[3], a[1]*b[2]+a[3]*b[3]])),
        (LinalgOpCode::MulM2V2, [Mat2(m), Vec2(v)]) => Some(Vec2([m[0]*v[0]+m[2]*v[1], m[1]*v[0]+m[3]*v[1]])),
        (LinalgOpCode::DetM2, [Mat2(m)]) => Some(Float(m[0]*m[3] - m[1]*m[2])),
        (LinalgOpCode::TraceM2, [Mat2(m)]) => Some(Float(m[0]+m[3])),
        (LinalgOpCode::TransposeM2, [Mat2(m)]) => Some(Mat2([m[0], m[2], m[1], m[3]])),
        (LinalgOpCode::InverseM2, [Mat2(m)]) => {
            let det = m[0]*m[3] - m[1]*m[2];
            if det.abs() > 1e-9 { Some(Mat2([m[3]/det, -m[1]/det, -m[2]/det, m[0]/det])) } else { None }
        },
        (LinalgOpCode::MakeMat3, [Vec3(c0), Vec3(c1), Vec3(c2)]) => Some(Mat3([c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]])),
        (LinalgOpCode::AddM3, [Mat3(a), Mat3(b)]) => Some(Mat3([a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4], a[5]+b[5], a[6]+b[6], a[7]+b[7], a[8]+b[8]])),
        (LinalgOpCode::SubM3, [Mat3(a), Mat3(b)]) => Some(Mat3([a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3], a[4]-b[4], a[5]-b[5], a[6]-b[6], a[7]-b[7], a[8]-b[8]])),
        (LinalgOpCode::ScaleM3, [Float(s), Mat3(m)]) => Some(Mat3([s*m[0], s*m[1], s*m[2], s*m[3], s*m[4], s*m[5], s*m[6], s*m[7], s*m[8]])),
        (LinalgOpCode::MulM3, [Mat3(a), Mat3(b)]) => Some(Mat3([
            a[0]*b[0]+a[3]*b[1]+a[6]*b[2], a[1]*b[0]+a[4]*b[1]+a[7]*b[2], a[2]*b[0]+a[5]*b[1]+a[8]*b[2],
            a[0]*b[3]+a[3]*b[4]+a[6]*b[5], a[1]*b[3]+a[4]*b[4]+a[7]*b[5], a[2]*b[3]+a[5]*b[4]+a[8]*b[5],
            a[0]*b[6]+a[3]*b[7]+a[6]*b[8], a[1]*b[6]+a[4]*b[7]+a[7]*b[8], a[2]*b[6]+a[5]*b[7]+a[8]*b[8]
        ])),
        (LinalgOpCode::MulM3V3, [Mat3(m), Vec3(v)]) => Some(Vec3([
            m[0]*v[0]+m[3]*v[1]+m[6]*v[2], m[1]*v[0]+m[4]*v[1]+m[7]*v[2], m[2]*v[0]+m[5]*v[1]+m[8]*v[2]
        ])),
        (LinalgOpCode::TraceM3, [Mat3(m)]) => Some(Float(m[0]+m[4]+m[8])),
        (LinalgOpCode::TransposeM3, [Mat3(m)]) => Some(Mat3([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]])),
        (LinalgOpCode::DetM3, [Mat3(m)]) => Some(Float(
            m[0]*(m[4]*m[8]-m[5]*m[7]) - m[3]*(m[1]*m[8]-m[2]*m[7]) + m[6]*(m[1]*m[5]-m[2]*m[4])
        )),
        (LinalgOpCode::InverseM3, [Mat3(m)]) => {
            let det = m[0]*(m[4]*m[8]-m[5]*m[7]) - m[3]*(m[1]*m[8]-m[2]*m[7]) + m[6]*(m[1]*m[5]-m[2]*m[4]);
            if det.abs() > 1e-9 {
                let inv_d = 1.0 / det;
                Some(Mat3([
                    (m[4]*m[8]-m[5]*m[7])*inv_d, -(m[1]*m[8]-m[2]*m[7])*inv_d, (m[1]*m[5]-m[2]*m[4])*inv_d,
                    -(m[3]*m[8]-m[5]*m[6])*inv_d, (m[0]*m[8]-m[2]*m[6])*inv_d, -(m[0]*m[5]-m[2]*m[3])*inv_d,
                    (m[3]*m[7]-m[4]*m[6])*inv_d, -(m[0]*m[7]-m[1]*m[6])*inv_d, (m[0]*m[4]-m[1]*m[3])*inv_d
                ]))
            } else { None }
        }
        _ => None,
    }
}