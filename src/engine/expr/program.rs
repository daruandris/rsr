//! Executable representation of mathematical expressions.

use super::node::Node;
use crate::Instruction;
use crate::engine::eval::scalar::Scalar;
use crate::engine::eval::types::ValueType;
use crate::engine::optimize::Parameterized;

// A flattened, compiled version of an AST.
///
/// While the genetic algorithm operates on tree structures (see [`crate::engine::expr::node::Node`]),
/// the evaluation engine requires a linear sequence of instructions for maximum performance.
///
/// `Program` separates the structural operations (`code`) from the numerical parameters (`constants`).
/// This architecture makes it incredibly fast to evaluate, and allows continuous optimizers
/// (like L-BFGS or CMA-ES) to update the weights in-place without rebuilding the expression tree.
#[derive(Clone, Debug)]
pub struct Program {
    /// The linear sequence of opcodes and memory load instructions in Postfix notation.
    pub code: Vec<Instruction>,
    /// The numerical parameters (weights) extracted from the AST.
    pub constants: Vec<Scalar>,
}

impl Program {
    /// Compiles a hierarchical AST into a linear executable program.
    ///
    /// This method traverses the provided nodes, extracting constants into a separate buffer
    /// and converting variables and operators into direct virtual machine instructions.
    ///
    /// # Arguments
    ///
    /// * `nodes` - A slice of [`crate::engine::expr::node::Node`] representing the expression tree.
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
                    code.push(*op);
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

impl Parameterized for Program {
    fn param_count(&self) -> usize {
        let mut n = 0;
        for c in &self.constants {
            n += match c {
                Scalar::Float(_) => 1,
                Scalar::Vec2(_) => 2,
                Scalar::Vec3(_) => 3,
                Scalar::Mat2(_) => 4,
                Scalar::Mat3(_) => 9,
                _ => 0,
            };
        }
        n
    }

    fn flatten_params(&self, buffer: &mut [f32]) {
        let mut ptr = 0;
        for c in &self.constants {
            match c {
                Scalar::Float(f) => {
                    if ptr < buffer.len() {
                        buffer[ptr] = *f;
                        ptr += 1;
                    }
                }
                Scalar::Vec2(v) => {
                    for val in v.iter().take(2) {
                        if ptr < buffer.len() {
                            buffer[ptr] = *val;
                            ptr += 1;
                        }
                    }
                }
                Scalar::Vec3(v) => {
                    for val in v.iter().take(3) {
                        if ptr < buffer.len() {
                            buffer[ptr] = *val;
                            ptr += 1;
                        }
                    }
                }
                Scalar::Mat2(m) => {
                    for val in m.iter().take(4) {
                        if ptr < buffer.len() {
                            buffer[ptr] = *val;
                            ptr += 1;
                        }
                    }
                }
                Scalar::Mat3(m) => {
                    for val in m.iter().take(9) {
                        if ptr < buffer.len() {
                            buffer[ptr] = *val;
                            ptr += 1;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn unflatten_params(&mut self, buffer: &[f32]) {
        let mut ptr = 0;
        for c in self.constants.iter_mut() {
            match c {
                Scalar::Float(f) => {
                    if ptr < buffer.len() {
                        *f = buffer[ptr];
                        ptr += 1;
                    }
                }
                Scalar::Vec2(v) => {
                    for val in v.iter_mut().take(2) {
                        if ptr < buffer.len() {
                            *val = buffer[ptr];
                            ptr += 1;
                        }
                    }
                }
                Scalar::Vec3(v) => {
                    for val in v.iter_mut().take(3) {
                        if ptr < buffer.len() {
                            *val = buffer[ptr];
                            ptr += 1;
                        }
                    }
                }
                Scalar::Mat2(m) => {
                    for val in m.iter_mut().take(4) {
                        if ptr < buffer.len() {
                            *val = buffer[ptr];
                            ptr += 1;
                        }
                    }
                }
                Scalar::Mat3(m) => {
                    for val in m.iter_mut().take(9) {
                        if ptr < buffer.len() {
                            *val = buffer[ptr];
                            ptr += 1;
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
