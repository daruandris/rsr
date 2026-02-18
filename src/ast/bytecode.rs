use wide::{f32x4};
use crate::ast::node::{Node, Op};

const EVALUATION_ARRAY_SIZE: usize = 32;

#[derive(Clone, Copy, Debug)]
pub enum OpCode {
    LoadVar(u8),       // Max 255 feature. Ha több kell, írd át u16-ra!
    LoadConst(u16),    // Max 65535 konstans
    Add, Sub, Mul, Div,
    Sin, Cos, Exp, Sqr
}

#[derive(Clone, Debug)]
pub struct Program {
    pub code: Vec<OpCode>,
    pub constants: Vec<f32>,
}

impl Program {
    pub fn from_nodes(nodes: &[Node]) -> Self {
        let mut code = Vec::with_capacity(nodes.len());
        let mut constants = Vec::new();

        for node in nodes {
            match node {
                Node::Variable(idx) => {
                    code.push(OpCode::LoadVar(*idx as u8));
                },
                Node::Constant(val) => {
                    // Itt lehetne deduplikálni (ha már van ilyen konstans, ne vedd fel újra),
                    // de a sebesség miatt most csak hozzáadjuk.
                    let c_idx = constants.len();
                    constants.push(*val);
                    code.push(OpCode::LoadConst(c_idx as u16));
                },
                Node::Operator(op) => {
                    let opcode = match op {
                        Op::Add => OpCode::Add,
                        Op::Sub => OpCode::Sub,
                        Op::Mul => OpCode::Mul,
                        Op::Div => OpCode::Div,
                        Op::Sin => OpCode::Sin,
                        Op::Cos => OpCode::Cos,
                        Op::Exp => OpCode::Exp,
                        Op::Sqr => OpCode::Sqr,
                    };
                    code.push(opcode);
                }
            }
        }

        Program { code, constants }
    }

    #[inline(always)]
    pub fn eval_simd(&self, features: &[f32x4]) -> f32x4 {
        let mut stack: [f32x4; EVALUATION_ARRAY_SIZE] = [f32x4::splat(0.0); EVALUATION_ARRAY_SIZE];
        let mut sp: usize = 0; 

        for op in &self.code {
            match op {
                OpCode::LoadVar(idx) => {
                    unsafe {
                        let val = *features.get_unchecked(*idx as usize);
                        *stack.get_unchecked_mut(sp) = val;
                        sp += 1;
                    }
                },
                OpCode::LoadConst(idx) => {
                    unsafe {
                        let val = f32x4::splat(*self.constants.get_unchecked(*idx as usize));
                        *stack.get_unchecked_mut(sp) = val;
                        sp += 1;
                    }
                },
                OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div => {
                    unsafe {
                        sp -= 2;
                        let a = *stack.get_unchecked(sp);
                        let b = *stack.get_unchecked(sp + 1);

                        let res = match op {
                            OpCode::Add => a + b,
                            OpCode::Sub => a - b,
                            OpCode::Mul => a * b,
                            OpCode::Div => a / b,
                            _ => std::hint::unreachable_unchecked(),
                        };
                        *stack.get_unchecked_mut(sp) = res;
                        sp += 1;
                    }
                },
                OpCode::Sin | OpCode::Cos | OpCode::Exp | OpCode::Sqr => {
                    unsafe {
                        let idx = sp - 1;
                        let a = *stack.get_unchecked(idx);
                        let res = match op {
                            OpCode::Sin => a.sin(),
                            OpCode::Cos => a.cos(),
                            OpCode::Exp => a.exp(),
                            OpCode::Sqr => a * a,
                            _ => std::hint::unreachable_unchecked(),
                        };
                        *stack.get_unchecked_mut(idx) = res;
                    }
                }
            }
        }

        unsafe { *stack.get_unchecked(0) }
    }
}