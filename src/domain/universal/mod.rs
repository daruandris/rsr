#![allow(unsafe_op_in_unsafe_fn)]
mod basic;
mod linalg;

use crate::domain::dual::*; 
use crate::domain::Domain;
use crate::ast::node::Node;
use wide::{f32x4};
use rand::RngExt;
use std::fmt;

pub enum SimplifyAction {
    ReplaceWithConstant(UniversalScalar),
    KeepArg(usize),
    None,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalType {
    Float, Int, Bool, Vec2, Vec3, Mat2, Mat3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalOp {
    AddF, SubF, MulF, DivF, SinF, CosF, ExpF, SqrF, LnF, SqrtF,
    MakeVec2, MakeVec3, GetXV2, GetYV2, GetXV3, GetYV3, GetZV3,
    AddV2, SubV2, ScaleV2, DotV2, NormV2, AddV3, SubV3, ScaleV3, DotV3, NormV3, CrossV3,
    MakeMat2, AddM2, SubM2, ScaleM2, MulM2, MulM2V2, DetM2, TraceM2, TransposeM2, InverseM2,
    MakeMat3, AddM3, SubM3, ScaleM3, MulM3, MulM3V3, DetM3, TraceM3, TransposeM3, InverseM3,
    IfElseF,
}

impl UniversalOp {
    pub fn forbidden_children(&self) -> &'static [UniversalOp] {
        match self {
            UniversalOp::SinF | UniversalOp::CosF => &[UniversalOp::SinF, UniversalOp::CosF, UniversalOp::ExpF],
            UniversalOp::ExpF => &[UniversalOp::ExpF, UniversalOp::SinF, UniversalOp::CosF, UniversalOp::SqrF, UniversalOp::LnF],
            UniversalOp::SqrtF => &[UniversalOp::SqrtF, UniversalOp::SqrF],
            UniversalOp::SqrF => &[UniversalOp::SqrF, UniversalOp::SqrtF],
            UniversalOp::LnF => &[UniversalOp::LnF, UniversalOp::ExpF],
            UniversalOp::TransposeM2 => &[UniversalOp::TransposeM2],
            UniversalOp::TransposeM3 => &[UniversalOp::TransposeM3],
            UniversalOp::InverseM2 => &[UniversalOp::InverseM2],
            UniversalOp::InverseM3 => &[UniversalOp::InverseM3],
            UniversalOp::GetXV2 | UniversalOp::GetYV2 => &[UniversalOp::MakeVec2],
            UniversalOp::GetXV3 | UniversalOp::GetYV3 | UniversalOp::GetZV3 => &[UniversalOp::MakeVec3],
            _ => &[],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalInstruction {
    LoadVarF(u8), LoadConstF(u16), LoadVarB(u8), LoadConstB(u16),
    LoadVarI(u8), LoadConstI(u16), LoadVarV2(u8), LoadConstV2(u16),
    LoadVarV3(u8), LoadConstV3(u16), LoadVarM2(u8), LoadConstM2(u16), LoadVarM3(u8), LoadConstM3(u16),
    
    AddF, SubF, MulF, DivF, SinF, CosF, ExpF, SqrF, SqrtF, LnF,
    MakeVec2, MakeVec3, GetXV2, GetYV2, GetXV3, GetYV3, GetZV3,
    AddV2, SubV2, ScaleV2, DotV2, NormV2, AddV3, SubV3, ScaleV3, DotV3, NormV3, CrossV3,
    MakeMat2, AddM2, SubM2, ScaleM2, MulM2, MulM2V2, DetM2, TraceM2, TransposeM2, InverseM2,
    MakeMat3, AddM3, SubM3, ScaleM3, MulM3, MulM3V3, DetM3, TraceM3, TransposeM3, InverseM3,
    IfElseF,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalScalar {
    Float(f32), Int(i32), Bool(bool),
    Vec2([f32; 2]), Vec3([f32; 3]), Mat2([f32; 4]), Mat3([f32; 9]),
}

impl UniversalScalar {
    pub fn is_zero(&self) -> bool {
        match self {
            Self::Float(f) => f.abs() < 1e-6,
            Self::Vec2(v) => v.iter().all(|&x| x.abs() < 1e-6),
            Self::Vec3(v) => v.iter().all(|&x| x.abs() < 1e-6),
            Self::Mat2(m) => m.iter().all(|&x| x.abs() < 1e-6),
            Self::Mat3(m) => m.iter().all(|&x| x.abs() < 1e-6),
            _ => false,
        }
    }
    pub fn is_one(&self) -> bool {
        match self { Self::Float(f) => (f - 1.0).abs() < 1e-6, _ => false }
    }
    pub fn is_identity(&self) -> bool {
        match self {
            Self::Mat2(m) => (m[0]-1.0).abs()<1e-6 && m[1].abs()<1e-6 && m[2].abs()<1e-6 && (m[3]-1.0).abs()<1e-6,
            Self::Mat3(m) => (m[0]-1.0).abs()<1e-6 && m[1].abs()<1e-6 && m[2].abs()<1e-6 &&
                             m[3].abs()<1e-6 && (m[4]-1.0).abs()<1e-6 && m[5].abs()<1e-6 &&
                             m[6].abs()<1e-6 && m[7].abs()<1e-6 && (m[8]-1.0).abs()<1e-6,
            _ => false,
        }
    }

    pub fn apply_threshold(&mut self, threshold: f32) {
        match self {
            Self::Float(f) => { 
                if f.abs() < threshold { *f = 0.0; } 
            },
            Self::Vec2(v) => { 
                for x in v.iter_mut() { if x.abs() < threshold { *x = 0.0; } } 
            },
            Self::Vec3(v) => { 
                for x in v.iter_mut() { if x.abs() < threshold { *x = 0.0; } } 
            },
            Self::Mat2(m) => { 
                for x in m.iter_mut() { if x.abs() < threshold { *x = 0.0; } } 
            },
            Self::Mat3(m) => { 
                for x in m.iter_mut() { if x.abs() < threshold { *x = 0.0; } } 
            },
            _ => {}
        }
    }
}

impl fmt::Display for UniversalScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UniversalScalar::Float(val) => write!(f, "{:.4}", val),
            UniversalScalar::Int(val) => write!(f, "{}", val),
            UniversalScalar::Bool(val) => write!(f, "{}", val),
            UniversalScalar::Vec2(arr) => write!(f, "[{:.2}, {:.2}]", arr[0], arr[1]),
            UniversalScalar::Vec3(arr) => write!(f, "[{:.2}, {:.2}, {:.2}]", arr[0], arr[1], arr[2]),
            UniversalScalar::Mat2(m) => write!(
                f, 
                "[{:.2}, {:.2}; {:.2}, {:.2}]", 
                m[0], m[2],
                m[1], m[3]
            ),
            UniversalScalar::Mat3(m) => write!(
                f, 
                "[{:.2}, {:.2}, {:.2}; {:.2}, {:.2}, {:.2}; {:.2}, {:.2}, {:.2}]", 
                m[0], m[3], m[6],
                m[1], m[4], m[7],
                m[2], m[5], m[8]
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ExprInfo { start_idx: usize, const_val: Option<UniversalScalar>, }

macro_rules! generate_domain_metadata {
    (
        $(
            $op:ident => { arity: $arity:expr, weight: $weight:expr, ret_type: $ret_type:expr, expected: $expected:expr }
        ),* $(,)?
    ) => {
        #[inline(always)] fn operator_arity(op: &Self::Operator) -> usize {
            match op { $( UniversalOp::$op => $arity, )* }
        }
        #[inline(always)] fn operator_weight(op: &Self::Operator) -> usize {
            match op { $( UniversalOp::$op => $weight, )* }
        }
        #[inline(always)] fn return_type(op: &Self::Operator) -> Self::TypeId {
            match op { $( UniversalOp::$op => $ret_type, )* }
        }
        #[inline(always)] fn expected_types(op: &Self::Operator) -> &'static [Self::TypeId] {
            match op { $( UniversalOp::$op => $expected, )* }
        }
        #[inline(always)] fn compile_operator(op: &Self::Operator) -> Self::Instruction {
            match op { $( UniversalOp::$op => UniversalInstruction::$op, )* }
        }
    };
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct UniversalDomain;

impl Domain for UniversalDomain {
    type Operator = UniversalOp; type Instruction = UniversalInstruction;
    type SimdValue = f32x4; type ScalarValue = UniversalScalar; type TypeId = UniversalType;

    generate_domain_metadata! {
        AddF => { arity: 2, weight: 1, ret_type: UniversalType::Float, expected: &[UniversalType::Float, UniversalType::Float] },
        SubF => { arity: 2, weight: 1, ret_type: UniversalType::Float, expected: &[UniversalType::Float, UniversalType::Float] },
        MulF => { arity: 2, weight: 1, ret_type: UniversalType::Float, expected: &[UniversalType::Float, UniversalType::Float] },
        DivF => { arity: 2, weight: 2, ret_type: UniversalType::Float, expected: &[UniversalType::Float, UniversalType::Float] },
        SinF => { arity: 1, weight: 3, ret_type: UniversalType::Float, expected: &[UniversalType::Float] },
        CosF => { arity: 1, weight: 3, ret_type: UniversalType::Float, expected: &[UniversalType::Float] },
        ExpF => { arity: 1, weight: 4, ret_type: UniversalType::Float, expected: &[UniversalType::Float] },
        SqrF => { arity: 1, weight: 2, ret_type: UniversalType::Float, expected: &[UniversalType::Float] },
        SqrtF => { arity: 1, weight: 2, ret_type: UniversalType::Float, expected: &[UniversalType::Float] },
        LnF => { arity: 1, weight: 4, ret_type: UniversalType::Float, expected: &[UniversalType::Float] },
        
        MakeVec2 => { arity: 2, weight: 2, ret_type: UniversalType::Vec2, expected: &[UniversalType::Float, UniversalType::Float] },
        GetXV2 => { arity: 1, weight: 2, ret_type: UniversalType::Float, expected: &[UniversalType::Vec2] },
        GetYV2 => { arity: 1, weight: 2, ret_type: UniversalType::Float, expected: &[UniversalType::Vec2] },
        AddV2 => { arity: 2, weight: 1, ret_type: UniversalType::Vec2, expected: &[UniversalType::Vec2, UniversalType::Vec2] },
        SubV2 => { arity: 2, weight: 1, ret_type: UniversalType::Vec2, expected: &[UniversalType::Vec2, UniversalType::Vec2] },
        ScaleV2 => { arity: 2, weight: 2, ret_type: UniversalType::Vec2, expected: &[UniversalType::Float, UniversalType::Vec2] },
        DotV2 => { arity: 2, weight: 2, ret_type: UniversalType::Float, expected: &[UniversalType::Vec2, UniversalType::Vec2] },
        NormV2 => { arity: 1, weight: 3, ret_type: UniversalType::Float, expected: &[UniversalType::Vec2] },
        
        MakeVec3 => { arity: 3, weight: 2, ret_type: UniversalType::Vec3, expected: &[UniversalType::Float, UniversalType::Float, UniversalType::Float] },
        GetXV3 => { arity: 1, weight: 2, ret_type: UniversalType::Float, expected: &[UniversalType::Vec3] },
        GetYV3 => { arity: 1, weight: 2, ret_type: UniversalType::Float, expected: &[UniversalType::Vec3] },
        GetZV3 => { arity: 1, weight: 2, ret_type: UniversalType::Float, expected: &[UniversalType::Vec3] },
        AddV3 => { arity: 2, weight: 1, ret_type: UniversalType::Vec3, expected: &[UniversalType::Vec3, UniversalType::Vec3] },
        SubV3 => { arity: 2, weight: 1, ret_type: UniversalType::Vec3, expected: &[UniversalType::Vec3, UniversalType::Vec3] },
        ScaleV3 => { arity: 2, weight: 2, ret_type: UniversalType::Vec3, expected: &[UniversalType::Float, UniversalType::Vec3] },
        DotV3 => { arity: 2, weight: 2, ret_type: UniversalType::Float, expected: &[UniversalType::Vec3, UniversalType::Vec3] },
        NormV3 => { arity: 1, weight: 3, ret_type: UniversalType::Float, expected: &[UniversalType::Vec3] },
        CrossV3 => { arity: 2, weight: 4, ret_type: UniversalType::Vec3, expected: &[UniversalType::Vec3, UniversalType::Vec3] },
        
        MakeMat2 => { arity: 2, weight: 3, ret_type: UniversalType::Mat2, expected: &[UniversalType::Vec2, UniversalType::Vec2] },
        AddM2 => { arity: 2, weight: 1, ret_type: UniversalType::Mat2, expected: &[UniversalType::Mat2, UniversalType::Mat2] },
        SubM2 => { arity: 2, weight: 1, ret_type: UniversalType::Mat2, expected: &[UniversalType::Mat2, UniversalType::Mat2] },
        ScaleM2 => { arity: 2, weight: 2, ret_type: UniversalType::Mat2, expected: &[UniversalType::Float, UniversalType::Mat2] },
        MulM2 => { arity: 2, weight: 3, ret_type: UniversalType::Mat2, expected: &[UniversalType::Mat2, UniversalType::Mat2] },
        MulM2V2 => { arity: 2, weight: 3, ret_type: UniversalType::Vec2, expected: &[UniversalType::Mat2, UniversalType::Vec2] },
        DetM2 => { arity: 1, weight: 4, ret_type: UniversalType::Float, expected: &[UniversalType::Mat2] },
        TraceM2 => { arity: 1, weight: 3, ret_type: UniversalType::Float, expected: &[UniversalType::Mat2] },
        TransposeM2 => { arity: 1, weight: 3, ret_type: UniversalType::Mat2, expected: &[UniversalType::Mat2] },
        InverseM2 => { arity: 1, weight: 4, ret_type: UniversalType::Mat2, expected: &[UniversalType::Mat2] },

        MakeMat3 => { arity: 3, weight: 5, ret_type: UniversalType::Mat3, expected: &[UniversalType::Vec3, UniversalType::Vec3, UniversalType::Vec3] },
        AddM3 => { arity: 2, weight: 1, ret_type: UniversalType::Mat3, expected: &[UniversalType::Mat3, UniversalType::Mat3] },
        SubM3 => { arity: 2, weight: 1, ret_type: UniversalType::Mat3, expected: &[UniversalType::Mat3, UniversalType::Mat3] },
        ScaleM3 => { arity: 2, weight: 2, ret_type: UniversalType::Mat3, expected: &[UniversalType::Float, UniversalType::Mat3] },
        MulM3 => { arity: 2, weight: 4, ret_type: UniversalType::Mat3, expected: &[UniversalType::Mat3, UniversalType::Mat3] },
        MulM3V3 => { arity: 2, weight: 3, ret_type: UniversalType::Vec3, expected: &[UniversalType::Mat3, UniversalType::Vec3] },
        DetM3 => { arity: 1, weight: 5, ret_type: UniversalType::Float, expected: &[UniversalType::Mat3] },
        TraceM3 => { arity: 1, weight: 3, ret_type: UniversalType::Float, expected: &[UniversalType::Mat3] },
        TransposeM3 => { arity: 1, weight: 3, ret_type: UniversalType::Mat3, expected: &[UniversalType::Mat3] },
        InverseM3 => { arity: 1, weight: 6, ret_type: UniversalType::Mat3, expected: &[UniversalType::Mat3] },

        IfElseF => { arity: 3, weight: 4, ret_type: UniversalType::Float, expected: &[UniversalType::Bool, UniversalType::Float, UniversalType::Float] },
    }

    fn format_operator(op: &Self::Operator, args: &[String]) -> String {
        if let Some(fmt) = basic::format_op(*op, args) { return fmt; }
        if let Some(fmt) = linalg::format_op(*op, args) { return fmt; }
        if *op == UniversalOp::IfElseF { return format!("(if {} then {} else {})", args[0], args[1], args[2]); }
        format!("{:?}({})", op, args.join(", "))
    }

    #[inline(always)] fn constant_type() -> Self::TypeId { UniversalType::Float }

    fn random_operator(
        target_type: Self::TypeId, 
        allowed_ops: &[Self::Operator], 
        parent_op: Option<Self::Operator>,
        rng: &mut impl RngExt
    ) -> Option<Self::Operator> {
        let forbidden = parent_op.map(|p| p.forbidden_children()).unwrap_or(&[]);
        let valid_count = allowed_ops.iter()
            .copied()
            .filter(|op| Self::return_type(op) == target_type && !forbidden.contains(op))
            .count();

        if valid_count == 0 { return None; }
        let chosen_idx = rng.random_range(0..valid_count);

        allowed_ops.iter()
            .copied()
            .filter(|op| Self::return_type(op) == target_type && !forbidden.contains(op))
            .nth(chosen_idx)
    }

    fn random_constant(target_type: Self::TypeId, rng: &mut impl RngExt) -> Option<Self::ScalarValue> {
        match target_type {
            UniversalType::Float => Some(UniversalScalar::Float(rng.random_range(-5.0..5.0))),
            UniversalType::Vec2 => Some(UniversalScalar::Vec2([rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)])),
            UniversalType::Vec3 => Some(UniversalScalar::Vec3([rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)])),
            UniversalType::Mat2 => Some(UniversalScalar::Mat2([rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)])),
            UniversalType::Mat3 => Some(UniversalScalar::Mat3([
                rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)
            ])),
            UniversalType::Bool => Some(UniversalScalar::Bool(rng.random::<bool>())),
            UniversalType::Int => Some(UniversalScalar::Int(rng.random_range(-10..10))),
        }
    }

    fn perturb_constant(val: &mut Self::ScalarValue, rng: &mut impl RngExt) {
        let r = rng.random::<f32>();
        match val {
            UniversalScalar::Float(f) => {
                if r < 0.8 { *f *= rng.random_range(0.9..1.1); } 
                else if r < 0.9 { *f += rng.random_range(-0.1..0.1); } 
                else { *f = rng.random_range(-5.0..5.0); }
            },
            UniversalScalar::Vec2(v) => { for i in 0..2 { v[i] += rng.random_range(-0.5..0.5); } },
            UniversalScalar::Vec3(v) => { for i in 0..3 { v[i] += rng.random_range(-0.5..0.5); } },
            UniversalScalar::Mat2(m) => { for i in 0..4 { m[i] += rng.random_range(-0.5..0.5); } },
            UniversalScalar::Mat3(m) => { for i in 0..9 { m[i] += rng.random_range(-0.5..0.5); } },
            _ => {} 
        }
    }

    #[inline(always)] fn load_var_instruction(idx: u8, target_type: Self::TypeId) -> Self::Instruction {
        match target_type {
            UniversalType::Float => UniversalInstruction::LoadVarF(idx), UniversalType::Vec2 => UniversalInstruction::LoadVarV2(idx),
            UniversalType::Vec3 => UniversalInstruction::LoadVarV3(idx), UniversalType::Mat2 => UniversalInstruction::LoadVarM2(idx),
            UniversalType::Mat3 => UniversalInstruction::LoadVarM3(idx), UniversalType::Bool => UniversalInstruction::LoadVarB(idx),
            UniversalType::Int => UniversalInstruction::LoadVarI(idx),
        }
    }
    
    #[inline(always)] fn load_const_instruction(idx: u16, target_type: Self::TypeId) -> Self::Instruction {
        match target_type {
            UniversalType::Float => UniversalInstruction::LoadConstF(idx), UniversalType::Vec2 => UniversalInstruction::LoadConstV2(idx),
            UniversalType::Vec3 => UniversalInstruction::LoadConstV3(idx), UniversalType::Mat2 => UniversalInstruction::LoadConstM2(idx),
            UniversalType::Mat3 => UniversalInstruction::LoadConstM3(idx), UniversalType::Bool => UniversalInstruction::LoadConstB(idx),
            UniversalType::Int => UniversalInstruction::LoadConstI(idx),
        }
    }

    #[inline(always)] fn scalar_to_f32(val: &Self::ScalarValue) ->Option<f32> {
         if let UniversalScalar::Float(f) = val { Some(*f) } else { None } }
    #[inline(always)] fn scalar_from_f32(val: f32) -> Self::ScalarValue { UniversalScalar::Float(val) }

    fn simplify(nodes: &[Node<Self>]) -> Vec<Node<Self>> {
        if nodes.is_empty() { return vec![]; }
        let mut output = Vec::with_capacity(nodes.len());
        let mut stack: Vec<ExprInfo> = Vec::with_capacity(32);

        for &node in nodes {
            match node {
                Node::Constant(val, _type_id) => {
                    let start_idx = output.len(); output.push(node);
                    stack.push(ExprInfo { start_idx, const_val: Some(val) });
                },
                Node::Variable(_, _) => {
                    let start_idx = output.len(); output.push(node);
                    stack.push(ExprInfo { start_idx, const_val: None });
                },
                Node::Operator(op) => {
                    let arity = Self::operator_arity(&op);
                    if stack.len() < arity {
                        let start_idx = output.len(); output.push(node);
                        stack.push(ExprInfo { start_idx, const_val: None });
                        continue;
                    }

                    let mut args = Vec::with_capacity(arity);
                    for _ in 0..arity { args.push(stack.pop().unwrap()); }
                    args.reverse();

                    let mut args_equal = false;
                    if arity == 2 {
                        let a = &args[0]; let b = &args[1];
                        if a.start_idx < b.start_idx && b.start_idx <= output.len() {
                            args_equal = output[a.start_idx..b.start_idx] == output[b.start_idx..];
                        }
                    }

                    let const_vals: Vec<Option<UniversalScalar>> = args.iter().map(|a| a.const_val).collect();

                    let mut action = linalg::try_simplify(op, &const_vals, args_equal);
                    if let SimplifyAction::None = action {
                        action = basic::try_simplify(op, &const_vals, args_equal);
                    }

                    match action {
                        SimplifyAction::ReplaceWithConstant(val) => {
                            output.truncate(args[0].start_idx);
                            let new_start = output.len();
                            output.push(Node::Constant(val, Self::return_type(&op)));
                            stack.push(ExprInfo { start_idx: new_start, const_val: Some(val) });
                        },
                        SimplifyAction::KeepArg(idx) => {
                            let target_arg = &args[idx];
                            let start_of_args = args[0].start_idx;
                            
                            let target_end = if idx == arity - 1 { output.len() } else { args[idx + 1].start_idx };
                            let target_len = target_end - target_arg.start_idx;
                            
                            if target_arg.start_idx > start_of_args {
                                output.copy_within(target_arg.start_idx..target_end, start_of_args);
                            }
                            output.truncate(start_of_args + target_len);
                            
                            let mut kept_info = target_arg.clone();
                            kept_info.start_idx = start_of_args;
                            stack.push(kept_info);
                        },
                        SimplifyAction::None => {
                            if arity == 1 && output.len() > args[0].start_idx {
                                if let Node::Operator(child_op) = output[output.len() - 1] {
                                    match (op, child_op) {
                                        (UniversalOp::LnF, UniversalOp::ExpF) | (UniversalOp::ExpF, UniversalOp::LnF) |
                                        (UniversalOp::SqrtF, UniversalOp::SqrF) | (UniversalOp::SqrF, UniversalOp::SqrtF) => {
                                            output.pop();
                                            stack.push(ExprInfo { start_idx: args[0].start_idx, const_val: None });
                                            continue;
                                        },
                                        _ => {}
                                    }
                                }
                            }
                            output.push(node);
                            stack.push(ExprInfo { start_idx: args[0].start_idx, const_val: None });
                        }
                    }
                }
            }
        }
        output
    }

    #[inline(always)]
    fn eval_simd(code: &[Self::Instruction], constants: &[Self::ScalarValue], features: &[Self::SimdValue]) -> Self::SimdValue {
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
                UniversalInstruction::LoadVarF(idx) => unsafe { *stack_f.get_unchecked_mut(sp_f) = *features.get_unchecked(*idx as usize); sp_f += 1; },
                UniversalInstruction::LoadConstF(idx) => unsafe {
                    if let UniversalScalar::Float(val) = constants.get_unchecked(*idx as usize) { *stack_f.get_unchecked_mut(sp_f) = f32x4::splat(*val); }
                    sp_f += 1;
                },
                UniversalInstruction::LoadConstV2(idx) => unsafe {
                    if let UniversalScalar::Vec2(val) = constants.get_unchecked(*idx as usize) { *stack_v2.get_unchecked_mut(sp_v2) = [f32x4::splat(val[0]), f32x4::splat(val[1])]; }
                    sp_v2 += 1;
                },
                UniversalInstruction::LoadConstV3(idx) => unsafe {
                    if let UniversalScalar::Vec3(val) = constants.get_unchecked(*idx as usize) { *stack_v3.get_unchecked_mut(sp_v3) = [f32x4::splat(val[0]), f32x4::splat(val[1]), f32x4::splat(val[2])]; }
                    sp_v3 += 1;
                },
                UniversalInstruction::LoadConstM2(idx) => unsafe {
                    if let UniversalScalar::Mat2(val) = constants.get_unchecked(*idx as usize) { 
                        *stack_m2.get_unchecked_mut(sp_m2) = [f32x4::splat(val[0]), f32x4::splat(val[1]), f32x4::splat(val[2]), f32x4::splat(val[3])]; 
                    }
                    sp_m2 += 1;
                },
                UniversalInstruction::LoadConstM3(idx) => unsafe {
                    if let UniversalScalar::Mat3(val) = constants.get_unchecked(*idx as usize) { 
                        *stack_m3.get_unchecked_mut(sp_m3) = [f32x4::splat(val[0]), f32x4::splat(val[1]), f32x4::splat(val[2]), f32x4::splat(val[3]), f32x4::splat(val[4]), f32x4::splat(val[5]), f32x4::splat(val[6]), f32x4::splat(val[7]), f32x4::splat(val[8])]; 
                    }
                    sp_m3 += 1;
                },

                UniversalInstruction::AddF => unsafe { basic::eval_add_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::SubF => unsafe { basic::eval_sub_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::MulF => unsafe { basic::eval_mul_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::DivF => unsafe { basic::eval_div_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::SinF => unsafe { basic::eval_sin_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::CosF => unsafe { basic::eval_cos_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::ExpF => unsafe { basic::eval_exp_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::SqrF => unsafe { basic::eval_sqr_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::SqrtF => unsafe { basic::eval_sqrt_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::LnF => unsafe { basic::eval_ln_f(&mut sp_f, &mut stack_f) },

                UniversalInstruction::MakeVec2 => unsafe { linalg::eval_make_vec2(&mut sp_f, &stack_f, &mut sp_v2, &mut stack_v2) },
                UniversalInstruction::MakeVec3 => unsafe { linalg::eval_make_vec3(&mut sp_f, &stack_f, &mut sp_v3, &mut stack_v3) },
                UniversalInstruction::GetXV2 => unsafe { linalg::eval_get_x_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2) },
                UniversalInstruction::GetYV2 => unsafe { linalg::eval_get_y_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2) },
                UniversalInstruction::GetXV3 => unsafe { linalg::eval_get_x_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3) },
                UniversalInstruction::GetYV3 => unsafe { linalg::eval_get_y_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3) },
                UniversalInstruction::GetZV3 => unsafe { linalg::eval_get_z_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3) },
                UniversalInstruction::AddV2 => unsafe { linalg::eval_add_v2(&mut sp_v2, &mut stack_v2) },
                UniversalInstruction::SubV2 => unsafe { linalg::eval_sub_v2(&mut sp_v2, &mut stack_v2) },
                UniversalInstruction::ScaleV2 => unsafe { linalg::eval_scale_v2(&mut sp_f, &stack_f, &mut sp_v2, &mut stack_v2) },
                UniversalInstruction::DotV2 => unsafe { linalg::eval_dot_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2) },
                UniversalInstruction::NormV2 => unsafe { linalg::eval_norm_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2) },
                UniversalInstruction::AddV3 => unsafe { linalg::eval_add_v3(&mut sp_v3, &mut stack_v3) },
                UniversalInstruction::SubV3 => unsafe { linalg::eval_sub_v3(&mut sp_v3, &mut stack_v3) },
                UniversalInstruction::ScaleV3 => unsafe { linalg::eval_scale_v3(&mut sp_f, &stack_f, &mut sp_v3, &mut stack_v3) },
                UniversalInstruction::DotV3 => unsafe { linalg::eval_dot_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3) },
                UniversalInstruction::NormV3 => unsafe { linalg::eval_norm_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3) },
                UniversalInstruction::CrossV3 => unsafe { linalg::eval_cross_v3(&mut sp_v3, &mut stack_v3) },
                UniversalInstruction::MakeMat2 => unsafe { linalg::eval_make_mat2(&mut sp_v2, &stack_v2, &mut sp_m2, &mut stack_m2) },
                UniversalInstruction::AddM2 => unsafe { linalg::eval_add_m2(&mut sp_m2, &mut stack_m2) },
                UniversalInstruction::SubM2 => unsafe { linalg::eval_sub_m2(&mut sp_m2, &mut stack_m2) },
                UniversalInstruction::ScaleM2 => unsafe { linalg::eval_scale_m2(&mut sp_f, &stack_f, &mut sp_m2, &mut stack_m2) },
                UniversalInstruction::MulM2 => unsafe { linalg::eval_mul_m2(&mut sp_m2, &mut stack_m2) },
                UniversalInstruction::MulM2V2 => unsafe { linalg::eval_mul_m2v2(&mut sp_m2, &stack_m2, &mut sp_v2, &mut stack_v2) },
                UniversalInstruction::DetM2 => unsafe { linalg::eval_det_m2(&mut sp_f, &mut stack_f, &mut sp_m2, &stack_m2) },
                UniversalInstruction::TraceM2 => unsafe { linalg::eval_trace_m2(&mut sp_f, &mut stack_f, &mut sp_m2, &stack_m2) },
                UniversalInstruction::TransposeM2 => unsafe { linalg::eval_transpose_m2(&mut sp_m2, &mut stack_m2) },
                UniversalInstruction::InverseM2 => unsafe { linalg::eval_inverse_m2(&mut sp_m2, &mut stack_m2) },
                UniversalInstruction::MakeMat3 => unsafe { linalg::eval_make_mat3(&mut sp_v3, &stack_v3, &mut sp_m3, &mut stack_m3) },
                UniversalInstruction::AddM3 => unsafe { linalg::eval_add_m3(&mut sp_m3, &mut stack_m3) },
                UniversalInstruction::SubM3 => unsafe { linalg::eval_sub_m3(&mut sp_m3, &mut stack_m3) },
                UniversalInstruction::ScaleM3 => unsafe { linalg::eval_scale_m3(&mut sp_f, &stack_f, &mut sp_m3, &mut stack_m3) },
                UniversalInstruction::MulM3 => unsafe { linalg::eval_mul_m3(&mut sp_m3, &mut stack_m3) },
                UniversalInstruction::MulM3V3 => unsafe { linalg::eval_mul_m3v3(&mut sp_m3, &stack_m3, &mut sp_v3, &mut stack_v3) },
                UniversalInstruction::DetM3 => unsafe { linalg::eval_det_m3(&mut sp_f, &mut stack_f, &mut sp_m3, &stack_m3) },
                UniversalInstruction::TraceM3 => unsafe { linalg::eval_trace_m3(&mut sp_f, &mut stack_f, &mut sp_m3, &stack_m3) },
                UniversalInstruction::TransposeM3 => unsafe { linalg::eval_transpose_m3(&mut sp_m3, &mut stack_m3) },
                UniversalInstruction::InverseM3 => unsafe { linalg::eval_inverse_m3(&mut sp_m3, &mut stack_m3) },

                UniversalInstruction::LoadVarV2(idx) => unsafe {
                    let i = *idx as usize;
                    *stack_v2.get_unchecked_mut(sp_v2) = [
                        *features.get_unchecked(i), 
                        *features.get_unchecked(i + 1)
                    ];
                    sp_v2 += 1;
                },
                UniversalInstruction::LoadVarV3(idx) => unsafe {
                    let i = *idx as usize;
                    *stack_v3.get_unchecked_mut(sp_v3) = [
                        *features.get_unchecked(i), 
                        *features.get_unchecked(i + 1), 
                        *features.get_unchecked(i + 2)
                    ];
                    sp_v3 += 1;
                },
                UniversalInstruction::LoadVarM2(idx) => unsafe {
                    let i = *idx as usize;
                    *stack_m2.get_unchecked_mut(sp_m2) = [
                        *features.get_unchecked(i), *features.get_unchecked(i + 1),
                        *features.get_unchecked(i + 2), *features.get_unchecked(i + 3)
                    ];
                    sp_m2 += 1;
                },
                UniversalInstruction::LoadVarM3(idx) => unsafe {
                    let i = *idx as usize;
                    *stack_m3.get_unchecked_mut(sp_m3) = [
                        *features.get_unchecked(i), *features.get_unchecked(i + 1), *features.get_unchecked(i + 2),
                        *features.get_unchecked(i + 3), *features.get_unchecked(i + 4), *features.get_unchecked(i + 5),
                        *features.get_unchecked(i + 6), *features.get_unchecked(i + 7), *features.get_unchecked(i + 8)
                    ];
                    sp_m3 += 1;
                },
                _ => {}
            }
        }
        unsafe { *stack_f.get_unchecked(0) }
    }

    fn compute_mse(code: &[Self::Instruction], constants: &[Self::ScalarValue], dataset: &crate::metrics::dataset::SimdDataset) -> f32 {
        let mut sum_squared_error = 0.0;
        let num_features = dataset.num_features as usize;
        let flat_features = &dataset.feature_flat;
        let targets = &dataset.target_batches;
        
        for i in 0..dataset.num_batches {
            let start = i * num_features;
            let input_batch = unsafe { flat_features.get_unchecked(start..start + num_features) };
            let prediction = Self::eval_simd(code, constants, input_batch);
            let target = unsafe { *targets.get_unchecked(i) };
            let diff = prediction - target;
            sum_squared_error += (diff * diff).reduce_add();
        }
        if !sum_squared_error.is_finite() { return f32::MAX; }
        sum_squared_error / (dataset.num_samples as f32)
    }

    #[inline(always)]
    fn type_weight(type_id: &Self::TypeId) -> usize {
        match type_id {
            UniversalType::Float | UniversalType::Int | UniversalType::Bool => 1,
            UniversalType::Vec2 => 2,
            UniversalType::Vec3 => 3,
            UniversalType::Mat2 => 4,
            UniversalType::Mat3 => 9,
        }
    }

    fn compute_mse_with_gradient(
        code: &[Self::Instruction], 
        constants: &[Self::ScalarValue], 
        dataset: &crate::metrics::dataset::SimdDataset
    ) -> (f32, [f32; 32]) {
        
        let mut sum_squared_error = f32x4::splat(0.0);
        let mut grad_sum = [f32x4::splat(0.0); 32];
        
        let num_features = dataset.num_features as usize;
        let flat_features = &dataset.feature_flat;
        let targets = &dataset.target_batches;
        
        let mut active_params_count = 0;
        for c in constants {
            active_params_count += match c {
                UniversalScalar::Float(_) => 1,
                UniversalScalar::Vec2(_) => 2,
                UniversalScalar::Vec3(_) => 3,
                UniversalScalar::Mat2(_) => 4,
                UniversalScalar::Mat3(_) => 9,
                _ => 0,
            };
        }
        active_params_count = active_params_count.min(32);
        
        if active_params_count == 0 {
            return (Self::compute_mse(code, constants, dataset), [0.0; 32]);
        }

        for i in 0..dataset.num_batches {
            let start = i * num_features;
            let input_batch = unsafe { flat_features.get_unchecked(start..start + num_features) };
            let target = unsafe { *targets.get_unchecked(i) };

            let mut diff = f32x4::splat(0.0);

            for k in 0..active_params_count {
                let dual_result = Self::eval_simd_dual(code, constants, input_batch, k);
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
}

#[inline(always)] 
pub unsafe fn eval_add_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    *sp_f -= 2;
    let a = *stack_f.get_unchecked(*sp_f);
    let b = *stack_f.get_unchecked(*sp_f + 1);
    *stack_f.get_unchecked_mut(*sp_f) = a + b;
    *sp_f += 1;
}

#[inline(always)] 
pub unsafe fn eval_sin_dual_f(sp_f: &mut usize, stack_f: &mut [DualSimd; 32]) {
    let idx = *sp_f - 1;
    *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).sin();
}

#[inline(always)] 
pub unsafe fn eval_add_dual_v3(sp_v3: &mut usize, stack_v3: &mut [[DualSimd; 3]; 32]) {
    *sp_v3 -= 2;
    let a = *stack_v3.get_unchecked(*sp_v3); 
    let b = *stack_v3.get_unchecked(*sp_v3 + 1);
    *stack_v3.get_unchecked_mut(*sp_v3) = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    *sp_v3 += 1;
}

#[inline(always)] 
pub unsafe fn eval_dot_dual_v3(sp_f: &mut usize, stack_f: &mut [DualSimd; 32], sp_v3: &mut usize, stack_v3: &[[DualSimd; 3]; 32]) {
    *sp_v3 -= 2;
    let a = *stack_v3.get_unchecked(*sp_v3); 
    let b = *stack_v3.get_unchecked(*sp_v3 + 1);
    *stack_f.get_unchecked_mut(*sp_f) = dual_dot_v3(&a, &b);
    *sp_f += 1;
}

impl UniversalDomain {
    #[inline(always)]
    pub fn eval_simd_dual(
        code: &[UniversalInstruction], 
        constants: &[UniversalScalar], 
        features: &[f32x4],
        active_const_idx: usize,
    ) -> DualSimd {
        let mut stack_f: [DualSimd; 32] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut sp_f: usize = 0;
        let mut stack_v2: [[DualSimd; 2]; 32] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut sp_v2: usize = 0;
        let mut stack_v3: [[DualSimd; 3]; 32] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut sp_v3: usize = 0;
        let mut stack_m2: [[DualSimd; 4]; 32] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut sp_m2: usize = 0;
        let mut stack_m3: [[DualSimd; 9]; 32] = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        let mut sp_m3: usize = 0;

        let get_flat_start_idx = |target_c_idx: usize| -> usize {
            let mut flat_idx = 0;
            for c in constants.iter().take(target_c_idx) {
                flat_idx += match c {
                    UniversalScalar::Float(_) => 1,
                    UniversalScalar::Vec2(_) => 2,
                    UniversalScalar::Vec3(_) => 3,
                    UniversalScalar::Mat2(_) => 4,
                    UniversalScalar::Mat3(_) => 9,
                    _ => 0,
                };
            }
            flat_idx
        };

        let get_grad = |flat_idx: usize| -> f32x4 {
            if flat_idx == active_const_idx { f32x4::splat(1.0) } else { f32x4::splat(0.0) }
        };

        for op in code {
            match op {
                UniversalInstruction::LoadVarF(idx) => unsafe { 
                    *stack_f.get_unchecked_mut(sp_f) = DualSimd::constant(*features.get_unchecked(*idx as usize));
                    sp_f += 1; 
                },
                UniversalInstruction::LoadVarV2(idx) => unsafe {
                    let i = *idx as usize;
                    *stack_v2.get_unchecked_mut(sp_v2) = [
                        DualSimd::constant(*features.get_unchecked(i)), 
                        DualSimd::constant(*features.get_unchecked(i + 1))
                    ];
                    sp_v2 += 1;
                },
                UniversalInstruction::LoadVarV3(idx) => unsafe {
                    let i = *idx as usize;
                    *stack_v3.get_unchecked_mut(sp_v3) = [
                        DualSimd::constant(*features.get_unchecked(i)), 
                        DualSimd::constant(*features.get_unchecked(i + 1)),
                        DualSimd::constant(*features.get_unchecked(i + 2))
                    ];
                    sp_v3 += 1;
                },
                UniversalInstruction::LoadConstF(idx) => unsafe {
                    if let UniversalScalar::Float(val) = constants.get_unchecked(*idx as usize) {
                        let flat_idx = get_flat_start_idx(*idx as usize);
                        *stack_f.get_unchecked_mut(sp_f) = DualSimd::new(f32x4::splat(*val), get_grad(flat_idx));
                    }
                    sp_f += 1;
                },
                UniversalInstruction::LoadConstV2(idx) => unsafe {
                    if let UniversalScalar::Vec2(val) = constants.get_unchecked(*idx as usize) {
                        let flat_idx = get_flat_start_idx(*idx as usize);
                        *stack_v2.get_unchecked_mut(sp_v2) = [
                            DualSimd::new(f32x4::splat(val[0]), get_grad(flat_idx)),
                            DualSimd::new(f32x4::splat(val[1]), get_grad(flat_idx + 1))
                        ];
                    }
                    sp_v2 += 1;
                },
                UniversalInstruction::LoadConstV3(idx) => unsafe {
                    if let UniversalScalar::Vec3(val) = constants.get_unchecked(*idx as usize) {
                        let flat_idx = get_flat_start_idx(*idx as usize);
                        *stack_v3.get_unchecked_mut(sp_v3) = [
                            DualSimd::new(f32x4::splat(val[0]), get_grad(flat_idx)),
                            DualSimd::new(f32x4::splat(val[1]), get_grad(flat_idx + 1)),
                            DualSimd::new(f32x4::splat(val[2]), get_grad(flat_idx + 2))
                        ];
                    }
                    sp_v3 += 1;
                },
                UniversalInstruction::AddF => unsafe { eval_add_dual_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::SubF => unsafe { eval_sub_dual_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::MulF => unsafe { eval_mul_dual_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::DivF => unsafe { eval_div_dual_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::SinF => unsafe { eval_sin_dual_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::CosF => unsafe { eval_cos_dual_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::ExpF => unsafe { eval_exp_dual_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::SqrF => unsafe { eval_sqr_dual_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::SqrtF => unsafe { eval_sqrt_dual_f(&mut sp_f, &mut stack_f) },
                UniversalInstruction::LnF => unsafe { eval_ln_dual_f(&mut sp_f, &mut stack_f) },

                UniversalInstruction::MakeVec2 => unsafe { eval_make_dual_vec2(&mut sp_f, &stack_f, &mut sp_v2, &mut stack_v2) },
                UniversalInstruction::MakeVec3 => unsafe { eval_make_dual_vec3(&mut sp_f, &stack_f, &mut sp_v3, &mut stack_v3) },
                UniversalInstruction::GetXV2 => unsafe { eval_get_x_dual_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2) },
                UniversalInstruction::GetYV2 => unsafe { eval_get_y_dual_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2) },
                UniversalInstruction::GetXV3 => unsafe { eval_get_x_dual_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3) },
                UniversalInstruction::GetYV3 => unsafe { eval_get_y_dual_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3) },
                UniversalInstruction::GetZV3 => unsafe { eval_get_z_dual_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3) },
                
                UniversalInstruction::AddV2 => unsafe { eval_add_dual_v2(&mut sp_v2, &mut stack_v2) },
                UniversalInstruction::SubV2 => unsafe { eval_sub_dual_v2(&mut sp_v2, &mut stack_v2) },
                UniversalInstruction::ScaleV2 => unsafe { eval_scale_dual_v2(&mut sp_f, &stack_f, &mut sp_v2, &mut stack_v2) },
                UniversalInstruction::DotV2 => unsafe { eval_dot_dual_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2) },
                UniversalInstruction::NormV2 => unsafe { eval_norm_dual_v2(&mut sp_f, &mut stack_f, &mut sp_v2, &stack_v2) },
                
                UniversalInstruction::AddV3 => unsafe { eval_add_dual_v3(&mut sp_v3, &mut stack_v3) },
                UniversalInstruction::SubV3 => unsafe { eval_sub_dual_v3(&mut sp_v3, &mut stack_v3) },
                UniversalInstruction::ScaleV3 => unsafe { eval_scale_dual_v3(&mut sp_f, &stack_f, &mut sp_v3, &mut stack_v3) },
                UniversalInstruction::DotV3 => unsafe { eval_dot_dual_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3) },
                UniversalInstruction::NormV3 => unsafe { eval_norm_dual_v3(&mut sp_f, &mut stack_f, &mut sp_v3, &stack_v3) },
                UniversalInstruction::CrossV3 => unsafe { eval_cross_dual_v3(&mut sp_v3, &mut stack_v3) },

                UniversalInstruction::MakeMat2 => unsafe { eval_make_dual_mat2(&mut sp_v2, &stack_v2, &mut sp_m2, &mut stack_m2) },
                UniversalInstruction::AddM2 => unsafe { eval_add_dual_m2(&mut sp_m2, &mut stack_m2) },
                UniversalInstruction::SubM2 => unsafe { eval_sub_dual_m2(&mut sp_m2, &mut stack_m2) },
                UniversalInstruction::ScaleM2 => unsafe { eval_scale_dual_m2(&mut sp_f, &stack_f, &mut sp_m2, &mut stack_m2) },
                UniversalInstruction::MulM2 => unsafe { eval_mul_dual_m2(&mut sp_m2, &mut stack_m2) },
                UniversalInstruction::MulM2V2 => unsafe { eval_mul_dual_m2v2(&mut sp_m2, &stack_m2, &mut sp_v2, &mut stack_v2) },
                UniversalInstruction::DetM2 => unsafe { eval_det_dual_m2(&mut sp_f, &mut stack_f, &mut sp_m2, &stack_m2) },
                UniversalInstruction::TraceM2 => unsafe { eval_trace_dual_m2(&mut sp_f, &mut stack_f, &mut sp_m2, &stack_m2) },
                UniversalInstruction::TransposeM2 => unsafe { eval_transpose_dual_m2(&mut sp_m2, &mut stack_m2) },

                UniversalInstruction::MakeMat3 => unsafe { eval_make_dual_mat3(&mut sp_v3, &stack_v3, &mut sp_m3, &mut stack_m3) },
                UniversalInstruction::AddM3 => unsafe { eval_add_dual_m3(&mut sp_m3, &mut stack_m3) },
                UniversalInstruction::SubM3 => unsafe { eval_sub_dual_m3(&mut sp_m3, &mut stack_m3) },
                UniversalInstruction::ScaleM3 => unsafe { eval_scale_dual_m3(&mut sp_f, &stack_f, &mut sp_m3, &mut stack_m3) },
                UniversalInstruction::MulM3 => unsafe { eval_mul_dual_m3(&mut sp_m3, &mut stack_m3) },
                UniversalInstruction::MulM3V3 => unsafe { eval_mul_dual_m3v3(&mut sp_m3, &stack_m3, &mut sp_v3, &mut stack_v3) },
                UniversalInstruction::DetM3 => unsafe { eval_det_dual_m3(&mut sp_f, &mut stack_f, &mut sp_m3, &stack_m3) },
                UniversalInstruction::TraceM3 => unsafe { eval_trace_dual_m3(&mut sp_f, &mut stack_f, &mut sp_m3, &stack_m3) },
                UniversalInstruction::TransposeM3 => unsafe { eval_transpose_dual_m3(&mut sp_m3, &mut stack_m3) },
                
                _ => {}
            }
        }
        
        unsafe { *stack_f.get_unchecked(0) }
    }
}