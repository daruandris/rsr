// src/domain/basic/mod.rs
use crate::domain::Domain;
use crate::ast::node::Node;
use wide::f32x4;
use rand::RngExt;

pub mod heuristic;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BasicOp { Add, Sub, Mul, Div, Sin, Cos, Exp, Sqr, Sqrt, Ln }

#[derive(Clone, Copy, Debug)]
pub enum BasicInstruction { LoadVar(u8), LoadConst(u16), Add, Sub, Mul, Div, Sin, Cos, Exp, Sqr, Sqrt, Ln }

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BasicDomain;

impl BasicOp {
    fn forbidden_children(&self) -> &'static [BasicOp] {
        match self {
            BasicOp::Add | BasicOp::Sub | BasicOp::Mul | BasicOp::Div | BasicOp::Sqr => &[],
            BasicOp::Sin | BasicOp::Cos => &[BasicOp::Sin, BasicOp::Cos, BasicOp::Exp],
            BasicOp::Exp => &[BasicOp::Exp, BasicOp::Sin, BasicOp::Cos, BasicOp::Sqr, BasicOp::Ln],
            BasicOp::Sqrt => &[BasicOp::Sqrt, BasicOp::Sqr],
            BasicOp::Ln => &[BasicOp::Ln, BasicOp::Exp],
        }
    }
}

impl Domain for BasicDomain {
    type Operator = BasicOp;
    type Instruction = BasicInstruction;
    type SimdValue = f32x4;
    type ScalarValue = f32;

    #[inline(always)]
    fn operator_arity(op: &Self::Operator) -> usize {
        match op {
            BasicOp::Sin | BasicOp::Cos | BasicOp::Exp | BasicOp::Sqr | BasicOp::Sqrt |BasicOp::Ln => 1,
            BasicOp::Add | BasicOp::Sub | BasicOp::Mul | BasicOp::Div => 2,
        }
    }

    #[inline(always)]
    fn operator_weight(op: &Self::Operator) -> usize {
        match op {
            BasicOp::Add | BasicOp::Sub | BasicOp::Mul => 1,
            BasicOp::Div => 2,
            BasicOp::Sqr => 2,
            BasicOp::Sqrt => 2,

            BasicOp::Sin | BasicOp::Cos => 3,

            BasicOp::Ln | BasicOp::Exp => 4,
        }
    }

    fn format_operator(op: &Self::Operator, args: &[String]) -> String {
        match op {
            BasicOp::Add => format!("({} + {})", args[0], args[1]),
            BasicOp::Sub => format!("({} - {})", args[0], args[1]),
            BasicOp::Mul => format!("({} * {})", args[0], args[1]),
            BasicOp::Div => format!("({} / {})", args[0], args[1]),
            BasicOp::Sin => format!("sin({})", args[0]),
            BasicOp::Cos => format!("cos({})", args[0]),
            BasicOp::Exp => format!("exp({})", args[0]),
            BasicOp::Sqr => format!("({})^2", args[0]),
            BasicOp::Sqrt => format!("sqrt(|{}|)", args[0]),
            BasicOp::Ln => format!("ln(|{}|)", args[0]),
        }
    }

    fn random_operator(arity: usize, parent_op: Option<Self::Operator>, rng: &mut impl RngExt) -> Self::Operator {
        let all_ops_arity1 = [BasicOp::Sin, BasicOp::Cos, BasicOp::Exp, BasicOp::Sqr, BasicOp::Sqrt, BasicOp::Ln];        let all_ops_arity2 = [BasicOp::Add, BasicOp::Sub, BasicOp::Mul, BasicOp::Div];
        let candidates: &[BasicOp] = if arity == 1 { &all_ops_arity1 } else { &all_ops_arity2 };
        
        let forbidden = match parent_op {
            Some(op) => op.forbidden_children(),
            None => &[],
        };
        
        let allowed: Vec<BasicOp> = candidates.iter().filter(|op| !forbidden.contains(op)).cloned().collect();
        if allowed.is_empty() {
            candidates[rng.random_range(0..candidates.len())]
        } else {
            allowed[rng.random_range(0..allowed.len())]
        }
    }

    #[inline(always)]
    fn compile_operator(op: &Self::Operator) -> Self::Instruction {
        match op {
            BasicOp::Add => BasicInstruction::Add,
            BasicOp::Sub => BasicInstruction::Sub,
            BasicOp::Mul => BasicInstruction::Mul,
            BasicOp::Div => BasicInstruction::Div,
            BasicOp::Sin => BasicInstruction::Sin,
            BasicOp::Cos => BasicInstruction::Cos,
            BasicOp::Exp => BasicInstruction::Exp,
            BasicOp::Sqr => BasicInstruction::Sqr,
            BasicOp::Sqrt => BasicInstruction::Sqrt,
            BasicOp::Ln => BasicInstruction::Ln,
        }
    }
    
    #[inline(always)] fn load_var_instruction(idx: u8) -> Self::Instruction { BasicInstruction::LoadVar(idx) }
    #[inline(always)] fn load_const_instruction(idx: u16) -> Self::Instruction { BasicInstruction::LoadConst(idx) }

    #[inline(always)]
    fn eval_simd(code: &[Self::Instruction], constants: &[Self::ScalarValue], features: &[Self::SimdValue]) -> Self::SimdValue {
        let mut stack: [f32x4; 32] = [f32x4::splat(0.0); 32];
        let mut sp: usize = 0; 

        for op in code {
            match op {
                BasicInstruction::LoadVar(idx) => unsafe {
                    *stack.get_unchecked_mut(sp) = *features.get_unchecked(*idx as usize);
                    sp += 1;
                },
                BasicInstruction::LoadConst(idx) => unsafe {
                    *stack.get_unchecked_mut(sp) = f32x4::splat(*constants.get_unchecked(*idx as usize));
                    sp += 1;
                },
                BasicInstruction::Add | BasicInstruction::Sub | BasicInstruction::Mul | BasicInstruction::Div => unsafe {
                    sp -= 2;
                    let a = *stack.get_unchecked(sp);
                    let b = *stack.get_unchecked(sp + 1);
                    let res = match op {
                        BasicInstruction::Add => a + b,
                        BasicInstruction::Sub => a - b,
                        BasicInstruction::Mul => a * b,
                        BasicInstruction::Div => a / b,
                        _ => std::hint::unreachable_unchecked(),
                    };
                    *stack.get_unchecked_mut(sp) = res;
                    sp += 1;
                },
                BasicInstruction::Sin | BasicInstruction::Cos | BasicInstruction::Exp | 
                BasicInstruction::Sqr | BasicInstruction::Sqrt | BasicInstruction::Ln => unsafe {
                    let idx = sp - 1;
                    let a = *stack.get_unchecked(idx);
                    let res = match op {
                        BasicInstruction::Sin => a.sin(),
                        BasicInstruction::Cos => a.cos(),
                        BasicInstruction::Exp => a.exp(),
                        BasicInstruction::Sqr => a * a,
                        BasicInstruction::Sqrt => a.abs().sqrt(),
                        BasicInstruction::Ln => {
                            let safe_a = a.abs() + f32x4::splat(1e-9);
                            safe_a.ln()
                        },
                        _ => std::hint::unreachable_unchecked(),
                    };
                    *stack.get_unchecked_mut(idx) = res;
                },
            }
        }
        unsafe { *stack.get_unchecked(0) }
    }

    fn simplify(nodes: &[Node<Self>]) -> Vec<Node<Self>> {
        heuristic::simplify_ast(nodes)
    }

    #[inline(always)] fn scalar_to_f32(val: &Self::ScalarValue) -> f32 { *val }
    #[inline(always)] fn scalar_from_f32(val: f32) -> Self::ScalarValue { val }

    fn random_constant(rng: &mut impl RngExt) -> Self::ScalarValue {
        rng.random_range(-5.0..5.0)
    }

    fn perturb_constant(val: &mut Self::ScalarValue, rng: &mut impl RngExt) {
        let r = rng.random::<f32>();
        if r < 0.8 {
            *val *= rng.random_range(0.9..1.1); 
        } else if r < 0.9 {
            *val += rng.random_range(-0.1..0.1);
        } else {
            *val = rng.random_range(-5.0..5.0);
        }
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
        }

        if !sum_squared_error.is_finite() { return f32::MAX; }
        sum_squared_error / (dataset.num_samples as f32)
    }
}