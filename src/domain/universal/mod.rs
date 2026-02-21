use crate::domain::Domain;
use crate::ast::node::Node;
use wide::{f32x4, CmpLt};
use rand::RngExt;
use std::fmt;

// --- 1. TÍPUSRENDSZER ---
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalType {
    Float, Int, Bool,
    Vec2, Vec3, Mat2, Mat3,
}

// --- 2. OPERÁTOROK ---
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalOp {
    // Basic
    AddF, SubF, MulF, DivF, SinF, CosF, ExpF, SqrF, LnF, SqrtF,
    // Linalg Vektorok
    MakeVec2, MakeVec3,
    GetXV2, GetYV2,
    GetXV3, GetYV3, GetZV3,
    AddV2, SubV2, ScaleV2, DotV2, NormV2,
    AddV3, SubV3, ScaleV3, DotV3, NormV3, CrossV3,
    // Linalg Mátrixok (2x2 és 3x3)
    MakeMat2, AddM2, SubM2, ScaleM2, MulM2, MulM2V2, 
    DetM2, TraceM2, TransposeM2, InverseM2,
    MakeMat3, AddM3, SubM3, ScaleM3, MulM3, MulM3V3, 
    DetM3, TraceM3, TransposeM3, InverseM3,
    // Logic
    IfElseF,
}

impl UniversalOp {
    pub fn forbidden_children(&self) -> &'static [UniversalOp] {
        match self {
            UniversalOp::SinF | UniversalOp::CosF => &[
                UniversalOp::SinF, UniversalOp::CosF, UniversalOp::ExpF
            ],
            UniversalOp::ExpF => &[
                UniversalOp::ExpF, UniversalOp::SinF, UniversalOp::CosF, 
                UniversalOp::SqrF, UniversalOp::LnF
            ],
            UniversalOp::SqrtF => &[UniversalOp::SqrtF, UniversalOp::SqrF],
            UniversalOp::SqrF => &[UniversalOp::SqrF, UniversalOp::SqrtF],
            UniversalOp::LnF => &[UniversalOp::LnF, UniversalOp::ExpF],
            _ => &[],
        }
    }
}

// --- 3. BYTECODE UTASÍTÁSOK ---
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalInstruction {
    LoadVarF(u8), LoadConstF(u16),
    LoadVarB(u8), LoadConstB(u16),
    LoadVarI(u8), LoadConstI(u16),
    LoadVarV2(u8), LoadConstV2(u16),
    LoadVarV3(u8), LoadConstV3(u16),
    LoadVarM2(u8), LoadConstM2(u16),
    LoadVarM3(u8), LoadConstM3(u16),
    
    AddF, SubF, MulF, DivF, SinF, CosF, ExpF, SqrF, SqrtF, LnF,
    MakeVec2, MakeVec3, GetXV2, GetYV2, GetXV3, GetYV3, GetZV3,
    AddV2, SubV2, ScaleV2, DotV2, NormV2,
    AddV3, SubV3, ScaleV3, DotV3, NormV3, CrossV3,
    MakeMat2, AddM2, SubM2, ScaleM2, MulM2, MulM2V2, DetM2, TraceM2, TransposeM2, InverseM2,
    MakeMat3, AddM3, SubM3, ScaleM3, MulM3, MulM3V3, DetM3, TraceM3, TransposeM3, InverseM3,
    IfElseF,
}

// --- 4. UNIVERZÁLIS KONSTANSOK ---
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalScalar {
    Float(f32),
    Int(i32),
    Bool(bool),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Mat2([f32; 4]),
    Mat3([f32; 9]),
}

impl fmt::Display for UniversalScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UniversalScalar::Float(val) => write!(f, "{:.4}", val),
            UniversalScalar::Int(val) => write!(f, "{}", val),
            UniversalScalar::Bool(val) => write!(f, "{}", val),
            UniversalScalar::Vec2(arr) => write!(f, "[{:.2}, {:.2}]", arr[0], arr[1]),
            UniversalScalar::Vec3(arr) => write!(f, "[{:.2}, {:.2}, {:.2}]", arr[0], arr[1], arr[2]),
            UniversalScalar::Mat2(_) => write!(f, "[Mat2]"),
            UniversalScalar::Mat3(_) => write!(f, "[Mat3]"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ExprInfo {
    start_idx: usize,
    const_val: Option<f32>,
}

// --- 5. A DOMAIN IMPLEMENTÁCIÓJA ---
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct UniversalDomain;

impl Domain for UniversalDomain {
    type Operator = UniversalOp;
    type Instruction = UniversalInstruction;
    type SimdValue = f32x4;
    type ScalarValue = UniversalScalar;
    type TypeId = UniversalType;

    #[inline(always)]
    fn operator_arity(op: &Self::Operator) -> usize {
        match op {
            UniversalOp::SinF | UniversalOp::CosF | UniversalOp::ExpF | 
            UniversalOp::SqrF | UniversalOp::LnF | UniversalOp::SqrtF |
            UniversalOp::GetXV2 | UniversalOp::GetYV2 | 
            UniversalOp::GetXV3 | UniversalOp::GetYV3 | UniversalOp::GetZV3 |
            UniversalOp::NormV2 | UniversalOp::NormV3 | 
            UniversalOp::DetM2 | UniversalOp::TraceM2 | UniversalOp::TransposeM2 | UniversalOp::InverseM2 |
            UniversalOp::DetM3 | UniversalOp::TraceM3 | UniversalOp::TransposeM3 | UniversalOp::InverseM3 => 1,
            
            UniversalOp::AddF | UniversalOp::SubF | UniversalOp::MulF | UniversalOp::DivF | 
            UniversalOp::AddV2 | UniversalOp::SubV2 | UniversalOp::ScaleV2 | UniversalOp::DotV2 |
            UniversalOp::AddV3 | UniversalOp::SubV3 | UniversalOp::ScaleV3 | UniversalOp::DotV3 | UniversalOp::CrossV3 |
            UniversalOp::MakeVec2 | UniversalOp::MakeMat2 | 
            UniversalOp::AddM2 | UniversalOp::SubM2 | UniversalOp::ScaleM2 | UniversalOp::MulM2 | UniversalOp::MulM2V2 |
            UniversalOp::AddM3 | UniversalOp::SubM3 | UniversalOp::ScaleM3 | UniversalOp::MulM3 | UniversalOp::MulM3V3 => 2,
            
            UniversalOp::MakeVec3 | UniversalOp::MakeMat3 | UniversalOp::IfElseF => 3,
        }
    }

    #[inline(always)]
    fn operator_weight(op: &Self::Operator) -> usize {
        match op {
            UniversalOp::AddF | UniversalOp::SubF | UniversalOp::MulF | 
            UniversalOp::AddV2 | UniversalOp::SubV2 | UniversalOp::AddV3 | UniversalOp::SubV3 |
            UniversalOp::AddM2 | UniversalOp::SubM2 | UniversalOp::AddM3 | UniversalOp::SubM3 => 1,
            
            UniversalOp::DivF | UniversalOp::SqrF | UniversalOp::SqrtF | 
            UniversalOp::DotV2 | UniversalOp::DotV3 | UniversalOp::CrossV3 |
            UniversalOp::ScaleV2 | UniversalOp::ScaleV3 | UniversalOp::ScaleM2 | UniversalOp::ScaleM3 |
            UniversalOp::GetXV2 | UniversalOp::GetYV2 | UniversalOp::GetXV3 | UniversalOp::GetYV3 | UniversalOp::GetZV3 => 2,
            
            UniversalOp::SinF | UniversalOp::CosF | UniversalOp::NormV2 | UniversalOp::NormV3 |
            UniversalOp::MulM2 | UniversalOp::MulM2V2 | UniversalOp::MulM3 | UniversalOp::MulM3V3 |
            UniversalOp::TransposeM2 | UniversalOp::TransposeM3 | UniversalOp::TraceM2 | UniversalOp::TraceM3 => 3,
            
            UniversalOp::IfElseF | UniversalOp::ExpF | UniversalOp::LnF | 
            UniversalOp::DetM2 | UniversalOp::DetM3 | UniversalOp::InverseM2 | UniversalOp::InverseM3 => 4,
            
            _ => 1,
        }
    }

    fn format_operator(op: &Self::Operator, args: &[String]) -> String {
        match op {
            UniversalOp::AddF => format!("({} + {})", args[0], args[1]),
            UniversalOp::SubF => format!("({} - {})", args[0], args[1]),
            UniversalOp::MulF => format!("({} * {})", args[0], args[1]),
            UniversalOp::DivF => format!("({} / {})", args[0], args[1]),
            UniversalOp::SinF => format!("sin({})", args[0]),
            UniversalOp::CosF => format!("cos({})", args[0]),
            UniversalOp::ExpF => format!("exp({})", args[0]),
            UniversalOp::SqrF => format!("({})^2", args[0]),
            UniversalOp::SqrtF => format!("sqrt(|{}|)", args[0]),
            UniversalOp::LnF => format!("ln(|{}|)", args[0]),
            
            UniversalOp::MakeVec2 => format!("vec2({}, {})", args[0], args[1]),
            UniversalOp::MakeVec3 => format!("vec3({}, {}, {})", args[0], args[1], args[2]),
            UniversalOp::GetXV2 | UniversalOp::GetXV3 => format!("{}.x", args[0]),
            UniversalOp::GetYV2 | UniversalOp::GetYV3 => format!("{}.y", args[0]),
            UniversalOp::GetZV3 => format!("{}.z", args[0]),
            UniversalOp::DotV2 | UniversalOp::DotV3 => format!("({} • {})", args[0], args[1]),
            UniversalOp::CrossV3 => format!("({} x {})", args[0], args[1]),
            UniversalOp::NormV2 | UniversalOp::NormV3 => format!("||{}||", args[0]),
            UniversalOp::ScaleV2 | UniversalOp::ScaleV3 | UniversalOp::ScaleM2 | UniversalOp::ScaleM3 => format!("({} * {})", args[0], args[1]),
            
            UniversalOp::InverseM2 | UniversalOp::InverseM3 => format!("{}^-1", args[0]),
            UniversalOp::TransposeM2 | UniversalOp::TransposeM3 => format!("{}^T", args[0]),
            UniversalOp::DetM2 | UniversalOp::DetM3 => format!("det({})", args[0]),
            UniversalOp::TraceM2 | UniversalOp::TraceM3 => format!("tr({})", args[0]),
            
            UniversalOp::IfElseF => format!("(if {} then {} else {})", args[0], args[1], args[2]),
            _ => format!("{:?}({})", op, args.join(", ")),
        }
    }

    #[inline(always)]
    fn return_type(op: &Self::Operator) -> Self::TypeId {
        use UniversalType::*;
        match op {
            UniversalOp::MakeVec2 | UniversalOp::AddV2 | UniversalOp::SubV2 | 
            UniversalOp::ScaleV2 | UniversalOp::MulM2V2 => Vec2,
            
            UniversalOp::MakeVec3 | UniversalOp::AddV3 | UniversalOp::SubV3 | 
            UniversalOp::ScaleV3 | UniversalOp::CrossV3 | UniversalOp::MulM3V3 => Vec3,
            
            UniversalOp::MakeMat2 | UniversalOp::AddM2 | UniversalOp::SubM2 | 
            UniversalOp::ScaleM2 | UniversalOp::MulM2 | UniversalOp::TransposeM2 | 
            UniversalOp::InverseM2 => Mat2,
            
            UniversalOp::MakeMat3 | UniversalOp::AddM3 | UniversalOp::SubM3 | 
            UniversalOp::ScaleM3 | UniversalOp::MulM3 | UniversalOp::TransposeM3 | 
            UniversalOp::InverseM3 => Mat3,
            
            _ => Float, 
        }
    }

    #[inline(always)]
    fn expected_types(op: &Self::Operator) -> Vec<Self::TypeId> {
        use UniversalType::*;
        match op {
            UniversalOp::MakeVec2 => vec![Float, Float],
            UniversalOp::MakeVec3 => vec![Float, Float, Float],
            UniversalOp::GetXV2 | UniversalOp::GetYV2 | UniversalOp::NormV2 => vec![Vec2],
            UniversalOp::GetXV3 | UniversalOp::GetYV3 | UniversalOp::GetZV3 | UniversalOp::NormV3 => vec![Vec3],
            
            UniversalOp::AddV2 | UniversalOp::SubV2 => vec![Vec2, Vec2],
            UniversalOp::AddV3 | UniversalOp::SubV3 => vec![Vec3, Vec3],
            UniversalOp::ScaleV2 => vec![Float, Vec2],
            UniversalOp::ScaleV3 => vec![Float, Vec3],
            UniversalOp::DotV2 => vec![Vec2, Vec2],
            UniversalOp::DotV3 | UniversalOp::CrossV3 => vec![Vec3, Vec3],
            
            UniversalOp::MakeMat2 => vec![Vec2, Vec2], 
            UniversalOp::AddM2 | UniversalOp::SubM2 | UniversalOp::MulM2 => vec![Mat2, Mat2],
            UniversalOp::ScaleM2 => vec![Float, Mat2],
            UniversalOp::MulM2V2 => vec![Mat2, Vec2],
            UniversalOp::DetM2 | UniversalOp::TraceM2 | UniversalOp::TransposeM2 | UniversalOp::InverseM2 => vec![Mat2],
            
            UniversalOp::MakeMat3 => vec![Vec3, Vec3, Vec3],
            UniversalOp::AddM3 | UniversalOp::SubM3 | UniversalOp::MulM3 => vec![Mat3, Mat3],
            UniversalOp::ScaleM3 => vec![Float, Mat3],
            UniversalOp::MulM3V3 => vec![Mat3, Vec3],
            UniversalOp::DetM3 | UniversalOp::TraceM3 | UniversalOp::TransposeM3 | UniversalOp::InverseM3 => vec![Mat3],
            
            UniversalOp::IfElseF => vec![Bool, Float, Float],
            UniversalOp::SinF | UniversalOp::CosF | UniversalOp::ExpF | UniversalOp::SqrF | UniversalOp::LnF | UniversalOp::SqrtF => vec![Float],
            UniversalOp::AddF | UniversalOp::SubF | UniversalOp::MulF | UniversalOp::DivF => vec![Float, Float],
        }
    }

    #[inline(always)] fn variable_type() -> Self::TypeId { UniversalType::Float }
    #[inline(always)] fn constant_type() -> Self::TypeId { UniversalType::Float }

    fn random_operator(
        target_type: Self::TypeId, 
        allowed_ops: &[Self::Operator], 
        parent_op: Option<Self::Operator>,
        rng: &mut impl RngExt
    ) -> Option<Self::Operator> {
        let forbidden = parent_op.map(|p| p.forbidden_children()).unwrap_or(&[]);
        let valid_ops: Vec<Self::Operator> = allowed_ops.iter()
            .copied()
            .filter(|op| Self::return_type(op) == target_type && !forbidden.contains(op))
            .collect();

        if valid_ops.is_empty() { return None; }
        Some(valid_ops[rng.random_range(0..valid_ops.len())])
    }

    fn random_constant(target_type: Self::TypeId, rng: &mut impl RngExt) -> Option<Self::ScalarValue> {
        match target_type {
            UniversalType::Float => Some(UniversalScalar::Float(rng.random_range(-5.0..5.0))),
            UniversalType::Vec2 => Some(UniversalScalar::Vec2([rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)])),
            UniversalType::Vec3 => Some(UniversalScalar::Vec3([rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)])),
            
            // ÚJ: Mat2 Konstans generálása
            UniversalType::Mat2 => Some(UniversalScalar::Mat2([
                rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0),
                rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)
            ])),
            
            // ÚJ: Mat3 Konstans generálása
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
            UniversalScalar::Vec2(v) => {
                for i in 0..2 { v[i] += rng.random_range(-0.5..0.5); }
            },
            UniversalScalar::Vec3(v) => {
                for i in 0..3 { v[i] += rng.random_range(-0.5..0.5); }
            },
            // ÚJ: Mátrixok perturbációja a Nelder-Mead és a Mutáció számára
            UniversalScalar::Mat2(m) => {
                for i in 0..4 { m[i] += rng.random_range(-0.5..0.5); }
            },
            UniversalScalar::Mat3(m) => {
                for i in 0..9 { m[i] += rng.random_range(-0.5..0.5); }
            },
            _ => {} 
        }
    }

    #[inline(always)]
    fn compile_operator(op: &Self::Operator) -> Self::Instruction {
        match op {
            UniversalOp::AddF => UniversalInstruction::AddF,
            UniversalOp::SubF => UniversalInstruction::SubF,
            UniversalOp::MulF => UniversalInstruction::MulF,
            UniversalOp::DivF => UniversalInstruction::DivF,
            UniversalOp::SinF => UniversalInstruction::SinF,
            UniversalOp::CosF => UniversalInstruction::CosF,
            UniversalOp::ExpF => UniversalInstruction::ExpF,
            UniversalOp::SqrF => UniversalInstruction::SqrF,
            UniversalOp::LnF => UniversalInstruction::LnF,
            UniversalOp::SqrtF => UniversalInstruction::SqrtF,
            
            UniversalOp::MakeVec2 => UniversalInstruction::MakeVec2,
            UniversalOp::MakeVec3 => UniversalInstruction::MakeVec3,
            UniversalOp::GetXV2 => UniversalInstruction::GetXV2,
            UniversalOp::GetYV2 => UniversalInstruction::GetYV2,
            UniversalOp::GetXV3 => UniversalInstruction::GetXV3,
            UniversalOp::GetYV3 => UniversalInstruction::GetYV3,
            UniversalOp::GetZV3 => UniversalInstruction::GetZV3,
            UniversalOp::AddV2 => UniversalInstruction::AddV2,
            UniversalOp::SubV2 => UniversalInstruction::SubV2,
            UniversalOp::ScaleV2 => UniversalInstruction::ScaleV2,
            UniversalOp::DotV2 => UniversalInstruction::DotV2,
            UniversalOp::NormV2 => UniversalInstruction::NormV2,
            UniversalOp::AddV3 => UniversalInstruction::AddV3,
            UniversalOp::SubV3 => UniversalInstruction::SubV3,
            UniversalOp::ScaleV3 => UniversalInstruction::ScaleV3,
            UniversalOp::DotV3 => UniversalInstruction::DotV3,
            UniversalOp::NormV3 => UniversalInstruction::NormV3,
            UniversalOp::CrossV3 => UniversalInstruction::CrossV3,
            
            UniversalOp::MakeMat2 => UniversalInstruction::MakeMat2,
            UniversalOp::AddM2 => UniversalInstruction::AddM2,
            UniversalOp::SubM2 => UniversalInstruction::SubM2,
            UniversalOp::ScaleM2 => UniversalInstruction::ScaleM2,
            UniversalOp::MulM2 => UniversalInstruction::MulM2,
            UniversalOp::MulM2V2 => UniversalInstruction::MulM2V2,
            UniversalOp::DetM2 => UniversalInstruction::DetM2,
            UniversalOp::TraceM2 => UniversalInstruction::TraceM2,
            UniversalOp::TransposeM2 => UniversalInstruction::TransposeM2,
            UniversalOp::InverseM2 => UniversalInstruction::InverseM2,
            
            UniversalOp::MakeMat3 => UniversalInstruction::MakeMat3,
            UniversalOp::AddM3 => UniversalInstruction::AddM3,
            UniversalOp::SubM3 => UniversalInstruction::SubM3,
            UniversalOp::ScaleM3 => UniversalInstruction::ScaleM3,
            UniversalOp::MulM3 => UniversalInstruction::MulM3,
            UniversalOp::MulM3V3 => UniversalInstruction::MulM3V3,
            UniversalOp::DetM3 => UniversalInstruction::DetM3,
            UniversalOp::TraceM3 => UniversalInstruction::TraceM3,
            UniversalOp::TransposeM3 => UniversalInstruction::TransposeM3,
            UniversalOp::InverseM3 => UniversalInstruction::InverseM3,
            
            UniversalOp::IfElseF => UniversalInstruction::IfElseF,
        }
    }

    #[inline(always)]
    fn load_var_instruction(idx: u8, target_type: Self::TypeId) -> Self::Instruction {
        match target_type {
            UniversalType::Float => UniversalInstruction::LoadVarF(idx),
            UniversalType::Vec2 => UniversalInstruction::LoadVarV2(idx),
            UniversalType::Vec3 => UniversalInstruction::LoadVarV3(idx),
            UniversalType::Mat2 => UniversalInstruction::LoadVarM2(idx),
            UniversalType::Mat3 => UniversalInstruction::LoadVarM3(idx),
            UniversalType::Bool => UniversalInstruction::LoadVarB(idx),
            UniversalType::Int => UniversalInstruction::LoadVarI(idx),
        }
    }
    
    #[inline(always)]
    fn load_const_instruction(idx: u16, target_type: Self::TypeId) -> Self::Instruction {
        match target_type {
            UniversalType::Float => UniversalInstruction::LoadConstF(idx),
            UniversalType::Vec2 => UniversalInstruction::LoadConstV2(idx),
            UniversalType::Vec3 => UniversalInstruction::LoadConstV3(idx),
            UniversalType::Mat2 => UniversalInstruction::LoadConstM2(idx),
            UniversalType::Mat3 => UniversalInstruction::LoadConstM3(idx),
            UniversalType::Bool => UniversalInstruction::LoadConstB(idx),
            UniversalType::Int => UniversalInstruction::LoadConstI(idx),
        }
    }

    #[inline(always)] 
    fn scalar_to_f32(val: &Self::ScalarValue) -> f32 { 
        if let UniversalScalar::Float(f) = val { *f } else { 0.0 } 
    }
    
    #[inline(always)] 
    fn scalar_from_f32(val: f32) -> Self::ScalarValue { 
        UniversalScalar::Float(val) 
    }

    fn simplify(nodes: &[Node<Self>]) -> Vec<Node<Self>> {
        if nodes.is_empty() { return vec![]; }
        
        let mut output = Vec::with_capacity(nodes.len());
        let mut stack: Vec<ExprInfo> = Vec::with_capacity(32);

        for &node in nodes {
            match node {
                Node::Constant(val, type_id) => {
                    let start_idx = output.len();
                    output.push(node);
                    let float_val = if let UniversalScalar::Float(f) = val { Some(f) } else { None };
                    stack.push(ExprInfo { start_idx, const_val: float_val });
                },
                Node::Variable(_, _) => {
                    let start_idx = output.len();
                    output.push(node);
                    stack.push(ExprInfo { start_idx, const_val: None });
                },
                Node::Operator(op) => {
                    let arity = Self::operator_arity(&op);
                    
                    if stack.len() < arity {
                        let start_idx = output.len();
                        output.push(node);
                        stack.push(ExprInfo { start_idx, const_val: None });
                        continue;
                    }

                    // --- UNÁRIS OPERÁTOROK ---
                    if arity == 1 {
                        let arg = stack.pop().unwrap();
                        
                        // Konstans folding
                        if let Some(val) = arg.const_val {
                            let folded = match op {
                                UniversalOp::SinF => Some(val.sin()),
                                UniversalOp::CosF => Some(val.cos()),
                                UniversalOp::ExpF => Some(val.exp()),
                                UniversalOp::SqrF => Some(val * val),
                                UniversalOp::SqrtF => Some(val.abs().sqrt()),
                                UniversalOp::LnF => Some((val.abs() + 1e-9).ln()),
                                _ => None,
                            };
                            
                            if let Some(f) = folded {
                                if f.is_finite() {
                                    output.truncate(arg.start_idx); // Zero-cost visszavágás!
                                    let new_start = output.len();
                                    output.push(Node::Constant(UniversalScalar::Float(f), UniversalType::Float));
                                    stack.push(ExprInfo { start_idx: new_start, const_val: Some(f) });
                                    continue;
                                }
                            }
                        }
                        
                        // Algebrai egyszerűsítés (Pl: sqrt(sqr(x)))
                        // Megnézzük az output utolsó előtti elemét
                        if output.len() > arg.start_idx {
                            if let Node::Operator(child_op) = output[output.len() - 1] {
                                match (op, child_op) {
                                    (UniversalOp::LnF, UniversalOp::ExpF) |
                                    (UniversalOp::ExpF, UniversalOp::LnF) |
                                    (UniversalOp::SqrtF, UniversalOp::SqrF) |
                                    (UniversalOp::SqrF, UniversalOp::SqrtF) => {
                                        // Visszavágjuk az operátort
                                        output.pop();
                                        // Nem pusholjuk a jelenlegit
                                        stack.push(ExprInfo { start_idx: arg.start_idx, const_val: None });
                                        continue;
                                    },
                                    _ => {}
                                }
                            }
                        }

                        output.push(node);
                        stack.push(ExprInfo { start_idx: arg.start_idx, const_val: None });
                    }
                    // --- BINÁRIS OPERÁTOROK ---
                    else if arity == 2 {
                        let b = stack.pop().unwrap();
                        let a = stack.pop().unwrap();

                        // Konstans folding
                        if let (Some(val_a), Some(val_b)) = (a.const_val, b.const_val) {
                            let folded = match op {
                                UniversalOp::AddF => Some(val_a + val_b),
                                UniversalOp::SubF => Some(val_a - val_b),
                                UniversalOp::MulF => Some(val_a * val_b),
                                UniversalOp::DivF => if val_b.abs() > 1e-9 { Some(val_a / val_b) } else { None },
                                _ => None,
                            };
                            
                            if let Some(f) = folded {
                                if f.is_finite() {
                                    output.truncate(a.start_idx); // Zero-cost visszavágás
                                    let new_start = output.len();
                                    output.push(Node::Constant(UniversalScalar::Float(f), UniversalType::Float));
                                    stack.push(ExprInfo { start_idx: new_start, const_val: Some(f) });
                                    continue;
                                }
                            }
                        }

                        // Algebrai egyszerűsítések (Pl: x * 0 = 0)
                        let b_is_zero = b.const_val.map_or(false, |v| v.abs() < 1e-6);
                        let a_is_zero = a.const_val.map_or(false, |v| v.abs() < 1e-6);
                        let b_is_one = b.const_val.map_or(false, |v| (v - 1.0).abs() < 1e-6);
                        let a_is_one = a.const_val.map_or(false, |v| (v - 1.0).abs() < 1e-6);

                        let are_equal = || output[a.start_idx..b.start_idx] == output[b.start_idx..];

                        match op {
                            UniversalOp::AddF => {
                                if b_is_zero { output.truncate(b.start_idx); stack.push(a); continue; }
                                // Mivel RPN, az 'a' törlése bonyolultabb (memmove), azt most skipeljük, csak a tailt vágjuk
                            },
                            UniversalOp::SubF => {
                                if b_is_zero { output.truncate(b.start_idx); stack.push(a); continue; }
                                if are_equal() {
                                    output.truncate(a.start_idx);
                                    let new_start = output.len();
                                    output.push(Node::Constant(UniversalScalar::Float(0.0), UniversalType::Float));
                                    stack.push(ExprInfo { start_idx: new_start, const_val: Some(0.0) });
                                    continue;
                                }
                            },
                            UniversalOp::MulF => {
                                if b_is_one { output.truncate(b.start_idx); stack.push(a); continue; }
                                if b_is_zero || a_is_zero {
                                    output.truncate(a.start_idx);
                                    let new_start = output.len();
                                    output.push(Node::Constant(UniversalScalar::Float(0.0), UniversalType::Float));
                                    stack.push(ExprInfo { start_idx: new_start, const_val: Some(0.0) });
                                    continue;
                                }
                            },
                            UniversalOp::DivF => {
                                if b_is_one { output.truncate(b.start_idx); stack.push(a); continue; }
                                if a_is_zero {
                                    output.truncate(a.start_idx);
                                    let new_start = output.len();
                                    output.push(Node::Constant(UniversalScalar::Float(0.0), UniversalType::Float));
                                    stack.push(ExprInfo { start_idx: new_start, const_val: Some(0.0) });
                                    continue;
                                }
                                if are_equal() {
                                    output.truncate(a.start_idx);
                                    let new_start = output.len();
                                    output.push(Node::Constant(UniversalScalar::Float(1.0), UniversalType::Float));
                                    stack.push(ExprInfo { start_idx: new_start, const_val: Some(1.0) });
                                    continue;
                                }
                            },
                            _ => {}
                        }

                        output.push(node);
                        stack.push(ExprInfo { start_idx: a.start_idx, const_val: None });
                    } 
                    // --- 3 ARITY OPERÁTOROK ---
                    else {
                        for _ in 0..arity { stack.pop(); }
                        let start_idx = if output.len() > arity { output.len() - arity } else { 0 }; // Ez durva becslés, ha kell 3 arity egyszerűsítés, ide jöhet
                        output.push(node);
                        stack.push(ExprInfo { start_idx, const_val: None });
                    }
                }
            }
        }
        
        output
    }

    #[inline(always)]
    fn eval_simd(
        code: &[Self::Instruction], 
        constants: &[Self::ScalarValue], 
        features: &[Self::SimdValue]
    ) -> Self::SimdValue {
        
        let mut stack_f: [f32x4; 32] = [f32x4::splat(0.0); 32];
        let mut sp_f: usize = 0;
        
        let mut stack_v2: [[f32x4; 2]; 32] = [[f32x4::splat(0.0); 2]; 32];
        let mut sp_v2: usize = 0;
        
        let mut stack_v3: [[f32x4; 3]; 32] = [[f32x4::splat(0.0); 3]; 32];
        let mut sp_v3: usize = 0;

        let mut stack_m2: [[f32x4; 4]; 32] = [[f32x4::splat(0.0); 4]; 32];
        let mut sp_m2: usize = 0;
        
        let mut stack_m3: [[f32x4; 9]; 32] = [[f32x4::splat(0.0); 9]; 32];
        let mut sp_m3: usize = 0;

        let mut stack_b: [f32x4; 32] = [f32x4::splat(0.0); 32];
        let mut sp_b: usize = 0;

        for op in code {
            match op {
                // --- LOAD ---
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
                // (LoadVarV2, LoadVarV3 stb. ide jöhet később, ha a dataset támogatja)

                // --- BASIC FLOAT ---
                UniversalInstruction::AddF => unsafe { sp_f -= 2; *stack_f.get_unchecked_mut(sp_f) = *stack_f.get_unchecked(sp_f) + *stack_f.get_unchecked(sp_f + 1); sp_f += 1; },
                UniversalInstruction::SubF => unsafe { sp_f -= 2; *stack_f.get_unchecked_mut(sp_f) = *stack_f.get_unchecked(sp_f) - *stack_f.get_unchecked(sp_f + 1); sp_f += 1; },
                UniversalInstruction::MulF => unsafe { sp_f -= 2; *stack_f.get_unchecked_mut(sp_f) = *stack_f.get_unchecked(sp_f) * *stack_f.get_unchecked(sp_f + 1); sp_f += 1; },
                UniversalInstruction::DivF => unsafe {
                    sp_f -= 2; let a = *stack_f.get_unchecked(sp_f); let b = *stack_f.get_unchecked(sp_f + 1);
                    let safe_b = b.abs().simd_lt(f32x4::splat(1e-9)).blend(f32x4::splat(1.0), b);
                    *stack_f.get_unchecked_mut(sp_f) = a / safe_b; sp_f += 1;
                },
                UniversalInstruction::SinF => unsafe { let idx = sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).sin(); },
                UniversalInstruction::CosF => unsafe { let idx = sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).cos(); },
                UniversalInstruction::ExpF => unsafe { let idx = sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).exp(); },
                UniversalInstruction::SqrF => unsafe { let idx = sp_f - 1; let a = *stack_f.get_unchecked(idx); *stack_f.get_unchecked_mut(idx) = a * a; },
                UniversalInstruction::SqrtF => unsafe { let idx = sp_f - 1; *stack_f.get_unchecked_mut(idx) = stack_f.get_unchecked(idx).abs().sqrt(); },
                UniversalInstruction::LnF => unsafe { let idx = sp_f - 1; *stack_f.get_unchecked_mut(idx) = (stack_f.get_unchecked(idx).abs() + f32x4::splat(1e-9)).ln(); },

                // --- VEKTOR KONSTRUKTOROK ---
                UniversalInstruction::MakeVec2 => unsafe {
                    sp_f -= 2;
                    *stack_v2.get_unchecked_mut(sp_v2) = [*stack_f.get_unchecked(sp_f), *stack_f.get_unchecked(sp_f + 1)];
                    sp_v2 += 1;
                },
                UniversalInstruction::MakeVec3 => unsafe {
                    sp_f -= 3;
                    *stack_v3.get_unchecked_mut(sp_v3) = [*stack_f.get_unchecked(sp_f), *stack_f.get_unchecked(sp_f + 1), *stack_f.get_unchecked(sp_f + 2)];
                    sp_v3 += 1;
                },
                UniversalInstruction::GetXV2 => unsafe { sp_v2 -= 1; *stack_f.get_unchecked_mut(sp_f) = stack_v2.get_unchecked(sp_v2)[0]; sp_f += 1; },
                UniversalInstruction::GetYV2 => unsafe { sp_v2 -= 1; *stack_f.get_unchecked_mut(sp_f) = stack_v2.get_unchecked(sp_v2)[1]; sp_f += 1; },
                UniversalInstruction::GetXV3 => unsafe { sp_v3 -= 1; *stack_f.get_unchecked_mut(sp_f) = stack_v3.get_unchecked(sp_v3)[0]; sp_f += 1; },
                UniversalInstruction::GetYV3 => unsafe { sp_v3 -= 1; *stack_f.get_unchecked_mut(sp_f) = stack_v3.get_unchecked(sp_v3)[1]; sp_f += 1; },
                UniversalInstruction::GetZV3 => unsafe { sp_v3 -= 1; *stack_f.get_unchecked_mut(sp_f) = stack_v3.get_unchecked(sp_v3)[2]; sp_f += 1; },

                // --- 2D VEKTOR MATEMATIKA ---
                UniversalInstruction::AddV2 => unsafe { sp_v2 -= 2; let a = *stack_v2.get_unchecked(sp_v2); let b = *stack_v2.get_unchecked(sp_v2 + 1); *stack_v2.get_unchecked_mut(sp_v2) = [a[0]+b[0], a[1]+b[1]]; sp_v2 += 1; },
                UniversalInstruction::SubV2 => unsafe { sp_v2 -= 2; let a = *stack_v2.get_unchecked(sp_v2); let b = *stack_v2.get_unchecked(sp_v2 + 1); *stack_v2.get_unchecked_mut(sp_v2) = [a[0]-b[0], a[1]-b[1]]; sp_v2 += 1; },
                UniversalInstruction::ScaleV2 => unsafe { sp_f -= 1; sp_v2 -= 1; let s = *stack_f.get_unchecked(sp_f); let v = *stack_v2.get_unchecked(sp_v2); *stack_v2.get_unchecked_mut(sp_v2) = [v[0]*s, v[1]*s]; sp_v2 += 1; },
                UniversalInstruction::DotV2 => unsafe { sp_v2 -= 2; let a = *stack_v2.get_unchecked(sp_v2); let b = *stack_v2.get_unchecked(sp_v2 + 1); *stack_f.get_unchecked_mut(sp_f) = (a[0]*b[0]) + (a[1]*b[1]); sp_f += 1; },
                UniversalInstruction::NormV2 => unsafe { let idx = sp_v2 - 1; let v = *stack_v2.get_unchecked(idx); *stack_f.get_unchecked_mut(sp_f) = ((v[0]*v[0]) + (v[1]*v[1])).sqrt(); sp_f += 1; sp_v2 -= 1; },

                // --- 3D VEKTOR MATEMATIKA ---
                UniversalInstruction::AddV3 => unsafe { sp_v3 -= 2; let a = *stack_v3.get_unchecked(sp_v3); let b = *stack_v3.get_unchecked(sp_v3 + 1); *stack_v3.get_unchecked_mut(sp_v3) = [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; sp_v3 += 1; },
                UniversalInstruction::SubV3 => unsafe { sp_v3 -= 2; let a = *stack_v3.get_unchecked(sp_v3); let b = *stack_v3.get_unchecked(sp_v3 + 1); *stack_v3.get_unchecked_mut(sp_v3) = [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; sp_v3 += 1; },
                UniversalInstruction::ScaleV3 => unsafe { sp_f -= 1; sp_v3 -= 1; let s = *stack_f.get_unchecked(sp_f); let v = *stack_v3.get_unchecked(sp_v3); *stack_v3.get_unchecked_mut(sp_v3) = [v[0]*s, v[1]*s, v[2]*s]; sp_v3 += 1; },
                UniversalInstruction::DotV3 => unsafe { sp_v3 -= 2; let a = *stack_v3.get_unchecked(sp_v3); let b = *stack_v3.get_unchecked(sp_v3 + 1); *stack_f.get_unchecked_mut(sp_f) = (a[0]*b[0]) + (a[1]*b[1]) + (a[2]*b[2]); sp_f += 1; },
                UniversalInstruction::NormV3 => unsafe { let idx = sp_v3 - 1; let v = *stack_v3.get_unchecked(idx); *stack_f.get_unchecked_mut(sp_f) = ((v[0]*v[0]) + (v[1]*v[1]) + (v[2]*v[2])).sqrt(); sp_f += 1; sp_v3 -= 1; },
                UniversalInstruction::CrossV3 => unsafe { 
                    sp_v3 -= 2; let a = *stack_v3.get_unchecked(sp_v3); let b = *stack_v3.get_unchecked(sp_v3 + 1);
                    *stack_v3.get_unchecked_mut(sp_v3) = [ a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0] ];
                    sp_v3 += 1;
                },

                // --- 2x2 MÁTRIX MŰVELETEK ---
                UniversalInstruction::MakeMat2 => unsafe { sp_v2 -= 2; let c0 = *stack_v2.get_unchecked(sp_v2); let c1 = *stack_v2.get_unchecked(sp_v2 + 1); *stack_m2.get_unchecked_mut(sp_m2) = [c0[0], c0[1], c1[0], c1[1]]; sp_m2 += 1; },
                UniversalInstruction::AddM2 => unsafe { sp_m2 -= 2; let a = *stack_m2.get_unchecked(sp_m2); let b = *stack_m2.get_unchecked(sp_m2 + 1); *stack_m2.get_unchecked_mut(sp_m2) = [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]]; sp_m2 += 1; },
                UniversalInstruction::SubM2 => unsafe { sp_m2 -= 2; let a = *stack_m2.get_unchecked(sp_m2); let b = *stack_m2.get_unchecked(sp_m2 + 1); *stack_m2.get_unchecked_mut(sp_m2) = [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]]; sp_m2 += 1; },
                UniversalInstruction::ScaleM2 => unsafe { sp_f -= 1; sp_m2 -= 1; let s = *stack_f.get_unchecked(sp_f); let m = *stack_m2.get_unchecked(sp_m2); *stack_m2.get_unchecked_mut(sp_m2) = [m[0]*s, m[1]*s, m[2]*s, m[3]*s]; sp_m2 += 1; },
                UniversalInstruction::MulM2 => unsafe { 
                    sp_m2 -= 2; let a = *stack_m2.get_unchecked(sp_m2); let b = *stack_m2.get_unchecked(sp_m2 + 1);
                    *stack_m2.get_unchecked_mut(sp_m2) = [
                        a[0]*b[0] + a[2]*b[1], a[1]*b[0] + a[3]*b[1],
                        a[0]*b[2] + a[2]*b[3], a[1]*b[2] + a[3]*b[3]
                    ]; sp_m2 += 1;
                },
                UniversalInstruction::MulM2V2 => unsafe {
                    sp_m2 -= 1; sp_v2 -= 1; let m = *stack_m2.get_unchecked(sp_m2); let v = *stack_v2.get_unchecked(sp_v2);
                    *stack_v2.get_unchecked_mut(sp_v2) = [ m[0]*v[0] + m[2]*v[1], m[1]*v[0] + m[3]*v[1] ];
                    sp_v2 += 1;
                },
                UniversalInstruction::DetM2 => unsafe { sp_m2 -= 1; let m = *stack_m2.get_unchecked(sp_m2); *stack_f.get_unchecked_mut(sp_f) = (m[0]*m[3]) - (m[1]*m[2]); sp_f += 1; },
                UniversalInstruction::TraceM2 => unsafe { sp_m2 -= 1; let m = *stack_m2.get_unchecked(sp_m2); *stack_f.get_unchecked_mut(sp_f) = m[0] + m[3]; sp_f += 1; },
                UniversalInstruction::TransposeM2 => unsafe { let idx = sp_m2 - 1; let m = *stack_m2.get_unchecked(idx); *stack_m2.get_unchecked_mut(idx) = [m[0], m[2], m[1], m[3]]; },
                UniversalInstruction::InverseM2 => unsafe {
                    let idx = sp_m2 - 1; let m = *stack_m2.get_unchecked(idx); let det = (m[0]*m[3]) - (m[1]*m[2]);
                    let is_singular = det.abs().simd_lt(f32x4::splat(1e-9)); let safe_det = is_singular.blend(f32x4::splat(1.0), det); let inv_d = f32x4::splat(1.0) / safe_det;
                    *stack_m2.get_unchecked_mut(idx) = [
                        is_singular.blend(f32x4::splat(1.0), m[3]*inv_d), is_singular.blend(f32x4::splat(0.0), -m[1]*inv_d),
                        is_singular.blend(f32x4::splat(0.0), -m[2]*inv_d), is_singular.blend(f32x4::splat(1.0), m[0]*inv_d)
                    ];
                },

                // --- 3x3 MÁTRIX MŰVELETEK ---
                UniversalInstruction::MakeMat3 => unsafe {
                    sp_v3 -= 3;
                    let c0 = *stack_v3.get_unchecked(sp_v3); let c1 = *stack_v3.get_unchecked(sp_v3 + 1); let c2 = *stack_v3.get_unchecked(sp_v3 + 2);
                    *stack_m3.get_unchecked_mut(sp_m3) = [c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]];
                    sp_m3 += 1;
                },
                UniversalInstruction::AddM3 => unsafe {
                    sp_m3 -= 2; let a = *stack_m3.get_unchecked(sp_m3); let b = *stack_m3.get_unchecked(sp_m3 + 1);
                    *stack_m3.get_unchecked_mut(sp_m3) = [ a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4], a[5]+b[5], a[6]+b[6], a[7]+b[7], a[8]+b[8] ];
                    sp_m3 += 1;
                },
                UniversalInstruction::SubM3 => unsafe {
                    sp_m3 -= 2; let a = *stack_m3.get_unchecked(sp_m3); let b = *stack_m3.get_unchecked(sp_m3 + 1);
                    *stack_m3.get_unchecked_mut(sp_m3) = [ a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3], a[4]-b[4], a[5]-b[5], a[6]-b[6], a[7]-b[7], a[8]-b[8] ];
                    sp_m3 += 1;
                },
                UniversalInstruction::ScaleM3 => unsafe {
                    sp_f -= 1; sp_m3 -= 1; let s = *stack_f.get_unchecked(sp_f); let m = *stack_m3.get_unchecked(sp_m3);
                    *stack_m3.get_unchecked_mut(sp_m3) = [ m[0]*s, m[1]*s, m[2]*s, m[3]*s, m[4]*s, m[5]*s, m[6]*s, m[7]*s, m[8]*s ];
                    sp_m3 += 1;
                },
                UniversalInstruction::MulM3 => unsafe {
                    sp_m3 -= 2; let a = *stack_m3.get_unchecked(sp_m3); let b = *stack_m3.get_unchecked(sp_m3 + 1);
                    *stack_m3.get_unchecked_mut(sp_m3) = [
                        a[0]*b[0] + a[3]*b[1] + a[6]*b[2], a[1]*b[0] + a[4]*b[1] + a[7]*b[2], a[2]*b[0] + a[5]*b[1] + a[8]*b[2],
                        a[0]*b[3] + a[3]*b[4] + a[6]*b[5], a[1]*b[3] + a[4]*b[4] + a[7]*b[5], a[2]*b[3] + a[5]*b[4] + a[8]*b[5],
                        a[0]*b[6] + a[3]*b[7] + a[6]*b[8], a[1]*b[6] + a[4]*b[7] + a[7]*b[8], a[2]*b[6] + a[5]*b[7] + a[8]*b[8]
                    ];
                    sp_m3 += 1;
                },
                UniversalInstruction::MulM3V3 => unsafe {
                    sp_m3 -= 1; sp_v3 -= 1; let m = *stack_m3.get_unchecked(sp_m3); let v = *stack_v3.get_unchecked(sp_v3);
                    *stack_v3.get_unchecked_mut(sp_v3) = [
                        m[0]*v[0] + m[3]*v[1] + m[6]*v[2],
                        m[1]*v[0] + m[4]*v[1] + m[7]*v[2],
                        m[2]*v[0] + m[5]*v[1] + m[8]*v[2]
                    ];
                    sp_v3 += 1;
                },
                UniversalInstruction::TraceM3 => unsafe { sp_m3 -= 1; let m = *stack_m3.get_unchecked(sp_m3); *stack_f.get_unchecked_mut(sp_f) = m[0] + m[4] + m[8]; sp_f += 1; },
                UniversalInstruction::TransposeM3 => unsafe {
                    let idx = sp_m3 - 1; let m = *stack_m3.get_unchecked(idx);
                    *stack_m3.get_unchecked_mut(idx) = [ m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8] ];
                },
                UniversalInstruction::DetM3 => unsafe {
                    sp_m3 -= 1; let m = *stack_m3.get_unchecked(sp_m3);
                    *stack_f.get_unchecked_mut(sp_f) = 
                        m[0] * (m[4]*m[8] - m[5]*m[7]) - 
                        m[3] * (m[1]*m[8] - m[2]*m[7]) + 
                        m[6] * (m[1]*m[5] - m[2]*m[4]);
                    sp_f += 1;
                },
                UniversalInstruction::InverseM3 => unsafe {
                    let idx = sp_m3 - 1; let m = *stack_m3.get_unchecked(idx);
                    let det = m[0]*(m[4]*m[8] - m[5]*m[7]) - m[3]*(m[1]*m[8] - m[2]*m[7]) + m[6]*(m[1]*m[5] - m[2]*m[4]);
                    
                    let is_singular = det.abs().simd_lt(f32x4::splat(1e-9));
                    let safe_det = is_singular.blend(f32x4::splat(1.0), det);
                    let inv_d = f32x4::splat(1.0) / safe_det;
                    
                    let adj = [
                         (m[4]*m[8] - m[5]*m[7])*inv_d, -(m[1]*m[8] - m[2]*m[7])*inv_d,  (m[1]*m[5] - m[2]*m[4])*inv_d,
                        -(m[3]*m[8] - m[5]*m[6])*inv_d,  (m[0]*m[8] - m[2]*m[6])*inv_d, -(m[0]*m[5] - m[2]*m[3])*inv_d,
                         (m[3]*m[7] - m[4]*m[6])*inv_d, -(m[0]*m[7] - m[1]*m[6])*inv_d,  (m[0]*m[4] - m[1]*m[3])*inv_d
                    ];
                    
                    // Ha szinguláris, identitást adunk vissza
                    *stack_m3.get_unchecked_mut(idx) = [
                        is_singular.blend(f32x4::splat(1.0), adj[0]), is_singular.blend(f32x4::splat(0.0), adj[1]), is_singular.blend(f32x4::splat(0.0), adj[2]),
                        is_singular.blend(f32x4::splat(0.0), adj[3]), is_singular.blend(f32x4::splat(1.0), adj[4]), is_singular.blend(f32x4::splat(0.0), adj[5]),
                        is_singular.blend(f32x4::splat(0.0), adj[6]), is_singular.blend(f32x4::splat(0.0), adj[7]), is_singular.blend(f32x4::splat(1.0), adj[8])
                    ];
                },
                _ => {} // Logic
            }
        }
        
        unsafe { *stack_f.get_unchecked(0) }
    }

    fn compute_mse(
        code: &[Self::Instruction], 
        constants: &[Self::ScalarValue], 
        dataset: &crate::metrics::dataset::SimdDataset
    ) -> f32 {
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
            let sqr = diff * diff;
            sum_squared_error += sqr.reduce_add();
            if !sum_squared_error.is_finite() { return f32::MAX; }
        }

        
        sum_squared_error / (dataset.num_samples as f32)
    }
}