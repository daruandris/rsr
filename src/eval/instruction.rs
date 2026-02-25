#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Instruction {
    LoadVarF(u8), LoadConstF(u16),
    LoadVarB(u8), LoadConstB(u16),
    LoadVarI(u8), LoadConstI(u16),
    LoadVarV2(u8), LoadConstV2(u16),
    LoadVarV3(u8), LoadConstV3(u16),
    LoadVarM2(u8), LoadConstM2(u16),
    LoadVarM3(u8), LoadConstM3(u16),

    AddF, SubF, MulF, DivF, SinF, CosF, ExpF, SqrF, SqrtF, LnF,
    
    MakeVec2, MakeVec3, GetXV2, GetYV2, GetXV3, GetYV3, GetZV3,
    AddV2, SubV2, ScaleV2, DotV2, NormV2, AddV3, SubV3, ScaleV3, DotV3, NormV3, CrossV3,
    
    MakeMat2, AddM2, SubM2, ScaleM2, MulM2, MulM2V2, DetM2, TraceM2, TransposeM2, InverseM2,
    MakeMat3, AddM3, SubM3, ScaleM3, MulM3, MulM3V3, DetM3, TraceM3, TransposeM3, InverseM3,
    
    IfElseF,
}