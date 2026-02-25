use crate::eval::instruction::Instruction;
use crate::eval::scalar::Scalar;
use crate::eval::types::ValueType;
use super::node::Node;

#[derive(Clone, Debug)]
pub struct Program {
    pub code: Vec<Instruction>,
    pub constants: Vec<Scalar>,
}

impl Program {
    pub fn from_nodes(nodes: &[Node]) -> Self {
        let mut code = Vec::with_capacity(nodes.len());
        let mut constants = Vec::new();

        for node in nodes {
            match node {
                Node::Variable(idx, type_id) => {
                    code.push(Self::load_var_instruction(*idx, *type_id));
                }
                Node::Constant(val, type_id) => {
                    let c_idx = constants.len();
                    constants.push(*val);
                    code.push(Self::load_const_instruction(c_idx as u16, *type_id));
                }
                Node::Operator(op) => {
                    code.push(op.compile());
                }
            }
        }

        Program { code, constants }
    }

    #[inline(always)]
    fn load_var_instruction(idx: u8, target_type: ValueType) -> Instruction {
        match target_type {
            ValueType::Float => Instruction::LoadVarF(idx),
            ValueType::Vec2 => Instruction::LoadVarV2(idx),
            ValueType::Vec3 => Instruction::LoadVarV3(idx),
            ValueType::Mat2 => Instruction::LoadVarM2(idx),
            ValueType::Mat3 => Instruction::LoadVarM3(idx),
            ValueType::Bool => Instruction::LoadVarB(idx),
            ValueType::Int => Instruction::LoadVarI(idx),
        }
    }

    #[inline(always)]
    fn load_const_instruction(idx: u16, target_type: ValueType) -> Instruction {
        match target_type {
            ValueType::Float => Instruction::LoadConstF(idx),
            ValueType::Vec2 => Instruction::LoadConstV2(idx),
            ValueType::Vec3 => Instruction::LoadConstV3(idx),
            ValueType::Mat2 => Instruction::LoadConstM2(idx),
            ValueType::Mat3 => Instruction::LoadConstM3(idx),
            ValueType::Bool => Instruction::LoadConstB(idx),
            ValueType::Int => Instruction::LoadConstI(idx),
        }
    }
}