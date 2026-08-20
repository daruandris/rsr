use crate::eval::scalar::Scalar;
use crate::eval::types::ValueType;
use crate::eval::state::VmState;

pub enum SimplifyAction {
    ReplaceWithConstant(Scalar),
    KeepArg(usize),
    None,
}

pub trait Domain {
    type OpCode: Clone + Copy + std::fmt::Debug + PartialEq + Eq;
    
    fn eval(op: Self::OpCode, ctx: &mut VmState);
    fn try_simplify(op: Self::OpCode, const_vals: &[Option<Scalar>], args_equal: bool) -> SimplifyAction;
    fn arity(op: Self::OpCode) -> usize;
    fn return_type(op: Self::OpCode) -> ValueType;
    fn expected_types(op: Self::OpCode) -> &'static [ValueType];
    fn weight(op: Self::OpCode) -> usize;
    fn forbidden_children(op: Self::OpCode) -> &'static [crate::Instruction];
}

#[macro_export]
macro_rules! compose_engine {
    ($engine_name:ident, $($domain_name:ident => $domain_type:ty),+) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Instruction {
            LoadVarF(u8),
            LoadConstF(u16),
            LoadVarB(u8),
            LoadConstB(u16),
            LoadVarI(u8),
            LoadConstI(u16),
            LoadVarV2(u8),
            LoadConstV2(u16),
            LoadVarV3(u8),
            LoadConstV3(u16),
            LoadVarM2(u8),
            LoadConstM2(u16),
            LoadVarM3(u8),
            LoadConstM3(u16),
            
            $( $domain_name(<$domain_type as $crate::domain::Domain>::OpCode), )+
        }

        pub struct $engine_name;

        impl $engine_name {
            #[inline(always)]
            pub fn eval_simd(code: &[Instruction], ctx: &mut $crate::eval::state::VmState) {
                for op in code {
                    match op {
                        $( Instruction::$domain_name(domain_op) => {
                            <$domain_type as $crate::domain::Domain>::eval(*domain_op, ctx);
                        })+
                        _ => {}
                    }
                }
            }

            pub fn try_simplify(op: Instruction, const_vals: &[Option<$crate::eval::scalar::Scalar>], args_equal: bool) -> $crate::domain::SimplifyAction {
                match op {
                    $( Instruction::$domain_name(domain_op) => {
                        <$domain_type as $crate::domain::Domain>::try_simplify(domain_op, const_vals, args_equal)
                    })+
                    _ => $crate::domain::SimplifyAction::None,
                }
            }
        }

        impl Instruction {
            #[inline(always)]
            pub fn arity(&self) -> usize {
                match self {
                    Instruction::LoadVarF(_) | Instruction::LoadConstF(_) |
                    Instruction::LoadVarB(_) | Instruction::LoadConstB(_) |
                    Instruction::LoadVarI(_) | Instruction::LoadConstI(_) |
                    Instruction::LoadVarV2(_) | Instruction::LoadConstV2(_) |
                    Instruction::LoadVarV3(_) | Instruction::LoadConstV3(_) |
                    Instruction::LoadVarM2(_) | Instruction::LoadConstM2(_) |
                    Instruction::LoadVarM3(_) | Instruction::LoadConstM3(_) => 0,
                    
                    $( Instruction::$domain_name(op) => <$domain_type as $crate::domain::Domain>::arity(*op), )+
                }
            }

            #[inline(always)]
            pub fn return_type(&self) -> $crate::eval::types::ValueType {
                use $crate::eval::types::ValueType;
                match self {
                    Instruction::LoadVarF(_) | Instruction::LoadConstF(_) => ValueType::Float,
                    Instruction::LoadVarB(_) | Instruction::LoadConstB(_) => ValueType::Bool,
                    Instruction::LoadVarI(_) | Instruction::LoadConstI(_) => ValueType::Int,
                    Instruction::LoadVarV2(_) | Instruction::LoadConstV2(_) => ValueType::Vec2,
                    Instruction::LoadVarV3(_) | Instruction::LoadConstV3(_) => ValueType::Vec3,
                    Instruction::LoadVarM2(_) | Instruction::LoadConstM2(_) => ValueType::Mat2,
                    Instruction::LoadVarM3(_) | Instruction::LoadConstM3(_) => ValueType::Mat3,
                    
                    $( Instruction::$domain_name(op) => <$domain_type as $crate::domain::Domain>::return_type(*op), )+
                }
            }

            #[inline(always)]
            pub fn expected_types(&self) -> &'static [$crate::eval::types::ValueType] {
                match self {
                    Instruction::LoadVarF(_) | Instruction::LoadConstF(_) |
                    Instruction::LoadVarB(_) | Instruction::LoadConstB(_) |
                    Instruction::LoadVarI(_) | Instruction::LoadConstI(_) |
                    Instruction::LoadVarV2(_) | Instruction::LoadConstV2(_) |
                    Instruction::LoadVarV3(_) | Instruction::LoadConstV3(_) |
                    Instruction::LoadVarM2(_) | Instruction::LoadConstM2(_) |
                    Instruction::LoadVarM3(_) | Instruction::LoadConstM3(_) => &[],
                    
                    $( Instruction::$domain_name(op) => <$domain_type as $crate::domain::Domain>::expected_types(*op), )+
                }
            }

            #[inline(always)]
            pub fn weight(&self) -> usize {
                match self {
                    Instruction::LoadVarF(_) | Instruction::LoadConstF(_) |
                    Instruction::LoadVarB(_) | Instruction::LoadConstB(_) |
                    Instruction::LoadVarI(_) | Instruction::LoadConstI(_) => 1,
                    Instruction::LoadVarV2(_) | Instruction::LoadConstV2(_) => 2,
                    Instruction::LoadVarV3(_) | Instruction::LoadConstV3(_) => 3,
                    Instruction::LoadVarM2(_) | Instruction::LoadConstM2(_) => 4,
                    Instruction::LoadVarM3(_) | Instruction::LoadConstM3(_) => 9,
                    
                    $( Instruction::$domain_name(op) => <$domain_type as $crate::domain::Domain>::weight(*op), )+
                }
            }

            #[inline(always)]
            pub fn forbidden_children(&self) -> &'static [Instruction] {
                match self {
                    Instruction::LoadVarF(_) | Instruction::LoadConstF(_) |
                    Instruction::LoadVarB(_) | Instruction::LoadConstB(_) |
                    Instruction::LoadVarI(_) | Instruction::LoadConstI(_) |
                    Instruction::LoadVarV2(_) | Instruction::LoadConstV2(_) |
                    Instruction::LoadVarV3(_) | Instruction::LoadConstV3(_) |
                    Instruction::LoadVarM2(_) | Instruction::LoadConstM2(_) |
                    Instruction::LoadVarM3(_) | Instruction::LoadConstM3(_) => &[],
                    $( Instruction::$domain_name(op) => <$domain_type as $crate::domain::Domain>::forbidden_children(*op), )+
                }
            }
        }
    };
}