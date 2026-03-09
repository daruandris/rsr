use super::instruction::Instruction;
use super::types::ValueType;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    AddF,
    SubF,
    MulF,
    DivF,
    SinF,
    CosF,
    ExpF,
    SqrF,
    LnF,
    SqrtF,
    MakeVec2,
    MakeVec3,
    GetXV2,
    GetYV2,
    GetXV3,
    GetYV3,
    GetZV3,
    AddV2,
    SubV2,
    ScaleV2,
    DotV2,
    NormV2,
    AddV3,
    SubV3,
    ScaleV3,
    DotV3,
    NormV3,
    CrossV3,
    MakeMat2,
    AddM2,
    SubM2,
    ScaleM2,
    MulM2,
    MulM2V2,
    DetM2,
    TraceM2,
    TransposeM2,
    InverseM2,
    MakeMat3,
    AddM3,
    SubM3,
    ScaleM3,
    MulM3,
    MulM3V3,
    DetM3,
    TraceM3,
    TransposeM3,
    InverseM3,
    IfElseF,
}

impl Op {
    pub fn forbidden_children(&self) -> &'static [Op] {
        match self {
            Op::SinF | Op::CosF => &[Op::SinF, Op::CosF, Op::ExpF],
            Op::ExpF => &[Op::ExpF, Op::SinF, Op::CosF, Op::SqrF, Op::LnF],
            Op::SqrtF => &[Op::SqrtF, Op::SqrF],
            Op::SqrF => &[Op::SqrF, Op::SqrtF],
            Op::LnF => &[Op::LnF, Op::ExpF],
            Op::TransposeM2 => &[Op::TransposeM2],
            Op::TransposeM3 => &[Op::TransposeM3],
            Op::InverseM2 => &[Op::InverseM2],
            Op::InverseM3 => &[Op::InverseM3],
            Op::GetXV2 | Op::GetYV2 => &[Op::MakeVec2],
            Op::GetXV3 | Op::GetYV3 | Op::GetZV3 => &[Op::MakeVec3],
            _ => &[],
        }
    }

    #[inline(always)]
    pub fn arity(&self) -> usize {
        match self {
            Op::AddF
            | Op::SubF
            | Op::MulF
            | Op::DivF
            | Op::MakeVec2
            | Op::AddV2
            | Op::SubV2
            | Op::ScaleV2
            | Op::DotV2
            | Op::AddV3
            | Op::SubV3
            | Op::ScaleV3
            | Op::DotV3
            | Op::CrossV3
            | Op::AddM2
            | Op::SubM2
            | Op::ScaleM2
            | Op::MulM2
            | Op::MulM2V2
            | Op::AddM3
            | Op::SubM3
            | Op::ScaleM3
            | Op::MulM3
            | Op::MulM3V3 => 2,
            Op::MakeVec3 | Op::MakeMat3 | Op::IfElseF => 3,
            _ => 1,
        }
    }

    #[inline(always)]
    pub fn weight(&self) -> usize {
        match self {
            Op::AddF
            | Op::SubF
            | Op::MulF
            | Op::AddV2
            | Op::SubV2
            | Op::AddV3
            | Op::SubV3
            | Op::AddM2
            | Op::SubM2
            | Op::AddM3
            | Op::SubM3 => 1,
            Op::DivF
            | Op::SqrF
            | Op::SqrtF
            | Op::MakeVec2
            | Op::GetXV2
            | Op::GetYV2
            | Op::ScaleV2
            | Op::DotV2
            | Op::MakeVec3
            | Op::GetXV3
            | Op::GetYV3
            | Op::GetZV3
            | Op::ScaleV3
            | Op::DotV3
            | Op::ScaleM2
            | Op::ScaleM3 => 2,
            Op::SinF
            | Op::CosF
            | Op::NormV2
            | Op::NormV3
            | Op::MakeMat2
            | Op::MulM2
            | Op::MulM2V2
            | Op::TraceM2
            | Op::TransposeM2
            | Op::MulM3V3
            | Op::TraceM3
            | Op::TransposeM3 => 3,
            Op::ExpF
            | Op::LnF
            | Op::CrossV3
            | Op::DetM2
            | Op::InverseM2
            | Op::MulM3
            | Op::IfElseF => 4,
            Op::MakeMat3 | Op::DetM3 => 5,
            Op::InverseM3 => 6,
        }
    }

    #[inline(always)]
    pub fn return_type(&self) -> ValueType {
        match self {
            Op::AddF
            | Op::SubF
            | Op::MulF
            | Op::DivF
            | Op::SinF
            | Op::CosF
            | Op::ExpF
            | Op::SqrF
            | Op::SqrtF
            | Op::LnF
            | Op::GetXV2
            | Op::GetYV2
            | Op::DotV2
            | Op::NormV2
            | Op::GetXV3
            | Op::GetYV3
            | Op::GetZV3
            | Op::DotV3
            | Op::NormV3
            | Op::DetM2
            | Op::TraceM2
            | Op::DetM3
            | Op::TraceM3
            | Op::IfElseF => ValueType::Float,
            Op::MakeVec2 | Op::AddV2 | Op::SubV2 | Op::ScaleV2 | Op::MulM2V2 => ValueType::Vec2,
            Op::MakeVec3 | Op::AddV3 | Op::SubV3 | Op::ScaleV3 | Op::CrossV3 | Op::MulM3V3 => {
                ValueType::Vec3
            }
            Op::MakeMat2
            | Op::AddM2
            | Op::SubM2
            | Op::ScaleM2
            | Op::MulM2
            | Op::TransposeM2
            | Op::InverseM2 => ValueType::Mat2,
            Op::MakeMat3
            | Op::AddM3
            | Op::SubM3
            | Op::ScaleM3
            | Op::MulM3
            | Op::TransposeM3
            | Op::InverseM3 => ValueType::Mat3,
        }
    }

    #[inline(always)]
    pub fn expected_types(&self) -> &'static [ValueType] {
        match self {
            Op::AddF | Op::SubF | Op::MulF | Op::DivF => &[ValueType::Float, ValueType::Float],
            Op::SinF | Op::CosF | Op::ExpF | Op::SqrF | Op::SqrtF | Op::LnF => &[ValueType::Float],
            Op::MakeVec2 => &[ValueType::Float, ValueType::Float],
            Op::GetXV2 | Op::GetYV2 | Op::NormV2 => &[ValueType::Vec2],
            Op::AddV2 | Op::SubV2 | Op::DotV2 => &[ValueType::Vec2, ValueType::Vec2],
            Op::ScaleV2 => &[ValueType::Float, ValueType::Vec2],
            Op::MakeVec3 => &[ValueType::Float, ValueType::Float, ValueType::Float],
            Op::GetXV3 | Op::GetYV3 | Op::GetZV3 | Op::NormV3 => &[ValueType::Vec3],
            Op::AddV3 | Op::SubV3 | Op::DotV3 | Op::CrossV3 => &[ValueType::Vec3, ValueType::Vec3],
            Op::ScaleV3 => &[ValueType::Float, ValueType::Vec3],
            Op::MakeMat2 => &[ValueType::Vec2, ValueType::Vec2],
            Op::AddM2 | Op::SubM2 | Op::MulM2 => &[ValueType::Mat2, ValueType::Mat2],
            Op::ScaleM2 => &[ValueType::Float, ValueType::Mat2],
            Op::MulM2V2 => &[ValueType::Mat2, ValueType::Vec2],
            Op::DetM2 | Op::TraceM2 | Op::TransposeM2 | Op::InverseM2 => &[ValueType::Mat2],
            Op::MakeMat3 => &[ValueType::Vec3, ValueType::Vec3, ValueType::Vec3],
            Op::AddM3 | Op::SubM3 | Op::MulM3 => &[ValueType::Mat3, ValueType::Mat3],
            Op::ScaleM3 => &[ValueType::Float, ValueType::Mat3],
            Op::MulM3V3 => &[ValueType::Mat3, ValueType::Vec3],
            Op::DetM3 | Op::TraceM3 | Op::TransposeM3 | Op::InverseM3 => &[ValueType::Mat3],
            Op::IfElseF => &[ValueType::Bool, ValueType::Float, ValueType::Float],
        }
    }

    #[inline(always)]
    pub fn compile(&self) -> Instruction {
        // Mivel a nevek megegyeznek, egyszerűen konvertálunk.
        // Ezt lehetne makróval is, de a biztonság és az autocomplete miatt a match itt tökéletes.
        match self {
            Op::AddF => Instruction::AddF,
            Op::SubF => Instruction::SubF,
            Op::MulF => Instruction::MulF,
            Op::DivF => Instruction::DivF,
            Op::SinF => Instruction::SinF,
            Op::CosF => Instruction::CosF,
            Op::ExpF => Instruction::ExpF,
            Op::SqrF => Instruction::SqrF,
            Op::SqrtF => Instruction::SqrtF,
            Op::LnF => Instruction::LnF,
            Op::MakeVec2 => Instruction::MakeVec2,
            Op::MakeVec3 => Instruction::MakeVec3,
            Op::GetXV2 => Instruction::GetXV2,
            Op::GetYV2 => Instruction::GetYV2,
            Op::GetXV3 => Instruction::GetXV3,
            Op::GetYV3 => Instruction::GetYV3,
            Op::GetZV3 => Instruction::GetZV3,
            Op::AddV2 => Instruction::AddV2,
            Op::SubV2 => Instruction::SubV2,
            Op::ScaleV2 => Instruction::ScaleV2,
            Op::DotV2 => Instruction::DotV2,
            Op::NormV2 => Instruction::NormV2,
            Op::AddV3 => Instruction::AddV3,
            Op::SubV3 => Instruction::SubV3,
            Op::ScaleV3 => Instruction::ScaleV3,
            Op::DotV3 => Instruction::DotV3,
            Op::NormV3 => Instruction::NormV3,
            Op::CrossV3 => Instruction::CrossV3,
            Op::MakeMat2 => Instruction::MakeMat2,
            Op::AddM2 => Instruction::AddM2,
            Op::SubM2 => Instruction::SubM2,
            Op::ScaleM2 => Instruction::ScaleM2,
            Op::MulM2 => Instruction::MulM2,
            Op::MulM2V2 => Instruction::MulM2V2,
            Op::DetM2 => Instruction::DetM2,
            Op::TraceM2 => Instruction::TraceM2,
            Op::TransposeM2 => Instruction::TransposeM2,
            Op::InverseM2 => Instruction::InverseM2,
            Op::MakeMat3 => Instruction::MakeMat3,
            Op::AddM3 => Instruction::AddM3,
            Op::SubM3 => Instruction::SubM3,
            Op::ScaleM3 => Instruction::ScaleM3,
            Op::MulM3 => Instruction::MulM3,
            Op::MulM3V3 => Instruction::MulM3V3,
            Op::DetM3 => Instruction::DetM3,
            Op::TraceM3 => Instruction::TraceM3,
            Op::TransposeM3 => Instruction::TransposeM3,
            Op::InverseM3 => Instruction::InverseM3,
            Op::IfElseF => Instruction::IfElseF,
        }
    }
}
