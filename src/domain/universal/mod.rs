use crate::domain::Domain;
use crate::ast::node::Node;
use wide::{f32x4, CmpLt}; // Itt a i32x4 is kellene majd a teljes verzióban
use rand::RngExt;
use std::fmt;

// --- 1. TÍPUSRENDSZER ---
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalType {
    Float,
    Int,
    Vec3,
    Bool,
}

// --- 2. OPERÁTOROK ---
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalOp {
    // Basic
    AddF, SubF, MulF, DivF, SinF, CosF, ExpF, SqrF,
    // Linalg
    AddV3, DotV3, ScaleV3,
    // Logic
    IfElseF,
}

// --- 3. BYTECODE UTASÍTÁSOK ---
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalInstruction {
    LoadVarF(u8), LoadConstF(u16),
    LoadVarV3(u8), LoadConstV3(u16),
    LoadVarB(u8), LoadConstB(u16),
    LoadVarI(u8), LoadConstI(u16),
    AddF, SubF, MulF, DivF, SinF, CosF, ExpF, SqrF,
    AddV3, DotV3, ScaleV3,
    IfElseF,
}

// --- 4. UNIVERZÁLIS KONSTANSOK ---
// Erre azért van szükség, mert a Domain::ScalarValue nem lehet már sima f32!
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UniversalScalar {
    Float(f32),
    Int(i32),
    Vec3([f32; 3]),
    Bool(bool),
}

impl fmt::Display for UniversalScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UniversalScalar::Float(val) => write!(f, "{:.4}", val),
            UniversalScalar::Int(val) => write!(f, "{}", val),
            UniversalScalar::Vec3(arr) => write!(f, "[{:.2}, {:.2}, {:.2}]", arr[0], arr[1], arr[2]),
            UniversalScalar::Bool(val) => write!(f, "{}", val),
        }
    }
}

// --- 5. A DOMAIN IMPLEMENTÁCIÓJA ---
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct UniversalDomain;

impl Domain for UniversalDomain {
    type Operator = UniversalOp;
    type Instruction = UniversalInstruction;
    type SimdValue = f32x4; // Az evolúció legvégén (MSE számítás) mindig Float-ot várunk
    type ScalarValue = UniversalScalar;
    type TypeId = UniversalType;

    #[inline(always)]
    fn operator_arity(op: &Self::Operator) -> usize {
        match op {
            UniversalOp::SinF | UniversalOp::CosF | UniversalOp::ExpF | UniversalOp::SqrF => 1,
            UniversalOp::AddF | UniversalOp::SubF | UniversalOp::MulF | UniversalOp::DivF | 
            UniversalOp::AddV3 | UniversalOp::DotV3 | UniversalOp::ScaleV3 => 2,
            UniversalOp::IfElseF => 3,
        }
    }

    #[inline(always)]
    fn operator_weight(op: &Self::Operator) -> usize {
        match op {
            UniversalOp::AddF | UniversalOp::SubF | UniversalOp::MulF | UniversalOp::AddV3 => 1,
            UniversalOp::DivF | UniversalOp::DotV3 | UniversalOp::ScaleV3 | UniversalOp::SqrF => 2,
            UniversalOp::SinF | UniversalOp::CosF => 3,
            UniversalOp::IfElseF | UniversalOp::ExpF => 4,
        }
    }

    fn format_operator(op: &Self::Operator, args: &[String]) -> String {
        match op {
            UniversalOp::AddF => format!("({} + {})", args[0], args[1]),
            UniversalOp::SubF => format!("({} - {})", args[0], args[1]),
            UniversalOp::SinF => format!("sin({})", args[0]),
            UniversalOp::DotV3 => format!("({} • {})", args[0], args[1]),
            UniversalOp::ScaleV3 | UniversalOp::MulF => format!("({} * {})", args[0], args[1]),
            UniversalOp::IfElseF => format!("(if {} then {} else {})", args[0], args[1], args[2]),
            UniversalOp::ExpF => format!("exp({})", args[0]),
            UniversalOp::SqrF => format!("({})^2", args[0]),
            UniversalOp::DivF => format!("({} / {})", args[0], args[1]),
            _ => format!("{:?}({})", op, args.join(", ")),
        }
    }

    // --- STGP: Típus szignatúrák (Ez fogja vezérelni a generátort!) ---
    #[inline(always)]
    fn return_type(op: &Self::Operator) -> Self::TypeId {
        match op {
            UniversalOp::AddF | UniversalOp::SubF | UniversalOp::MulF | UniversalOp::DivF | 
            UniversalOp::SinF | UniversalOp::CosF | UniversalOp::DotV3 | UniversalOp::IfElseF |
            UniversalOp::ExpF | UniversalOp::SqrF => UniversalType::Float,
            UniversalOp::AddV3 | UniversalOp::ScaleV3 => UniversalType::Vec3,
        }
    }

    #[inline(always)]
    fn expected_types(op: &Self::Operator) -> Vec<Self::TypeId> {
        match op {
            UniversalOp::AddF | UniversalOp::SubF | UniversalOp::MulF | UniversalOp::DivF | 
            UniversalOp::ExpF | UniversalOp::SqrF => vec![UniversalType::Float, UniversalType::Float],
            UniversalOp::SinF | UniversalOp::CosF => vec![UniversalType::Float],
            UniversalOp::AddV3 => vec![UniversalType::Vec3, UniversalType::Vec3],
            UniversalOp::DotV3 => vec![UniversalType::Vec3, UniversalType::Vec3],
            UniversalOp::ScaleV3 => vec![UniversalType::Float, UniversalType::Vec3],
            UniversalOp::IfElseF => vec![UniversalType::Bool, UniversalType::Float, UniversalType::Float],
        }
    }

    // --- ALAPÉRTELMEZETT VÁLTOZÓ ÉS KONSTANS TÍPUSOK ---
    // FIGYELEM: Ahhoz, hogy a generátor tudja, milyen típusú változót/konstanst húzzon,
    // a Domain trait-et majd finomítani kell. Jelenleg Float-ot adunk vissza alapként.
    #[inline(always)] fn variable_type() -> Self::TypeId { UniversalType::Float }
    #[inline(always)] fn constant_type() -> Self::TypeId { UniversalType::Float }

    fn random_operator(target_type: Self::TypeId, allowed_ops: &[Self::Operator], rng: &mut impl RngExt) -> Option<Self::Operator> {
        let valid_ops: Vec<Self::Operator> = allowed_ops.iter()
            .copied()
            .filter(|op| Self::return_type(op) == target_type)
            .collect();

        if valid_ops.is_empty() { 
            return None; 
        }
        
        Some(valid_ops[rng.random_range(0..valid_ops.len())])
    }

    fn random_constant(target_type: Self::TypeId, rng: &mut impl RngExt) -> Option<Self::ScalarValue> {
        match target_type {
            UniversalType::Float => Some(UniversalScalar::Float(rng.random_range(-5.0..5.0))),
            UniversalType::Vec3 => Some(UniversalScalar::Vec3([
                rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0), rng.random_range(-5.0..5.0)
            ])),
            UniversalType::Bool => Some(UniversalScalar::Bool(rng.random::<bool>())),
            UniversalType::Int => Some(UniversalScalar::Int(rng.random_range(-10..10))),
        }
    }

    fn perturb_constant(val: &mut Self::ScalarValue, rng: &mut impl RngExt) {
        match val {
            UniversalScalar::Float(f) => *f += rng.random_range(-0.5..0.5),
            UniversalScalar::Vec3(v) => {
                v[0] += rng.random_range(-0.5..0.5);
                v[1] += rng.random_range(-0.5..0.5);
                v[2] += rng.random_range(-0.5..0.5);
            },
            _ => {} // Int és Bool nem finomhangolható Nelder-Mead szerűen
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
            
            UniversalOp::AddV3 => UniversalInstruction::AddV3,
            UniversalOp::DotV3 => UniversalInstruction::DotV3,
            UniversalOp::ScaleV3 => UniversalInstruction::ScaleV3,
            UniversalOp::IfElseF => UniversalInstruction::IfElseF,
        }
    }

    // A LoadVar és LoadConst jelenleg a Domain traitben nem kap típus infót, 
    // ezt majd orvosolnunk kell! Egyelőre feltételezzük, hogy Float:
    #[inline(always)]
    fn load_var_instruction(idx: u8, target_type: Self::TypeId) -> Self::Instruction {
        match target_type {
            UniversalType::Float => UniversalInstruction::LoadVarF(idx),
            UniversalType::Vec3 => UniversalInstruction::LoadVarV3(idx),
            UniversalType::Bool => UniversalInstruction::LoadVarB(idx),
            UniversalType::Int => UniversalInstruction::LoadVarI(idx),
        }
    }
    #[inline(always)]
    fn load_const_instruction(idx: u16, target_type: Self::TypeId) -> Self::Instruction {
        match target_type {
            UniversalType::Float => UniversalInstruction::LoadConstF(idx),
            UniversalType::Vec3 => UniversalInstruction::LoadConstV3(idx),
            UniversalType::Bool => UniversalInstruction::LoadConstB(idx),
            UniversalType::Int => UniversalInstruction::LoadConstI(idx),
        }
    }
    // (A Nelder-Mead konverziók)
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
        
        let mut stack: Vec<Vec<Node<Self>>> = Vec::with_capacity(32);

        for node in nodes {
            match node {
                Node::Variable(_, _) | Node::Constant(_, _) => {
                    stack.push(vec![*node]);
                },
                Node::Operator(op) => {
                    let arity = Self::operator_arity(op);
                    if stack.len() < arity {
                        stack.push(vec![*node]);
                        continue;
                    }

                    // Gyerekek levétele a veremről
                    let mut children = Vec::with_capacity(arity);
                    for _ in 0..arity {
                        children.push(stack.pop().unwrap());
                    }
                    children.reverse(); // Mert LIFO a verem

                    // 1. CONSTANT FOLDING (Konstansok előre kiszámítása)
                    // Megnézzük, hogy minden gyerek egyetlen Float konstans-e
                    let all_float_const = children.iter().all(|c| {
                        c.len() == 1 && matches!(c[0], Node::Constant(UniversalScalar::Float(_), _))
                    });

                    if all_float_const {
                        let mut vals = Vec::new();
                        for c in &children {
                            if let Node::Constant(UniversalScalar::Float(v), _) = c[0] {
                                vals.push(v);
                            }
                        }

                        let folded = match op {
                            UniversalOp::AddF => Some(vals[0] + vals[1]),
                            UniversalOp::SubF => Some(vals[0] - vals[1]),
                            UniversalOp::MulF => Some(vals[0] * vals[1]),
                            UniversalOp::DivF => {
                                if vals[1].abs() > 1e-6 { Some(vals[0] / vals[1]) } else { None }
                            },
                            UniversalOp::SinF => Some(vals[0].sin()),
                            UniversalOp::CosF => Some(vals[0].cos()),
                            UniversalOp::ExpF => Some(vals[0].exp()),
                            UniversalOp::SqrF => Some(vals[0] * vals[0]),
                            _ => None,
                        };

                        if let Some(f) = folded {
                            if f.is_finite() {
                                stack.push(vec![Node::Constant(UniversalScalar::Float(f), UniversalType::Float)]);
                                continue;
                            }
                        }
                    }

                    // 2. ALGEBRAI EGYSZERŰSÍTÉSEK (x + 0, x * 1, stb.)
                    if arity == 2 {
                        let left_is_const = children[0].len() == 1 && matches!(children[0][0], Node::Constant(UniversalScalar::Float(_), _));
                        let right_is_const = children[1].len() == 1 && matches!(children[1][0], Node::Constant(UniversalScalar::Float(_), _));
                        
                        let left_val = if left_is_const { 
                            if let Node::Constant(UniversalScalar::Float(v), _) = children[0][0] { v } else { 0.0 } 
                        } else { 0.0 };
                        
                        let right_val = if right_is_const { 
                            if let Node::Constant(UniversalScalar::Float(v), _) = children[1][0] { v } else { 0.0 } 
                        } else { 0.0 };

                        match op {
                            UniversalOp::AddF | UniversalOp::SubF => {
                                if right_is_const && right_val.abs() < 1e-6 {
                                    stack.push(children[0].clone()); // x +/- 0 -> x
                                    continue;
                                }
                                if op == &UniversalOp::AddF && left_is_const && left_val.abs() < 1e-6 {
                                    stack.push(children[1].clone()); // 0 + x -> x
                                    continue;
                                }
                            },
                            UniversalOp::MulF => {
                                if right_is_const {
                                    if (right_val - 1.0).abs() < 1e-6 { stack.push(children[0].clone()); continue; } // x * 1 -> x
                                    if right_val.abs() < 1e-6 { stack.push(vec![Node::Constant(UniversalScalar::Float(0.0), UniversalType::Float)]); continue; } // x * 0 -> 0
                                }
                                if left_is_const {
                                    if (left_val - 1.0).abs() < 1e-6 { stack.push(children[1].clone()); continue; } // 1 * x -> x
                                    if left_val.abs() < 1e-6 { stack.push(vec![Node::Constant(UniversalScalar::Float(0.0), UniversalType::Float)]); continue; } // 0 * x -> 0
                                }
                            },
                            UniversalOp::DivF => {
                                if right_is_const && (right_val - 1.0).abs() < 1e-6 {
                                    stack.push(children[0].clone()); // x / 1 -> x
                                    continue;
                                }
                            }
                            _ => {}
                        }
                    }

                    // 3. HA NEM TUDTUK EGYSZERŰSÍTENI, ÉPÍTSÜK VISSZA A RÉSZFÁT
                    let mut subtree = Vec::new();
                    for mut child in children {
                        subtree.append(&mut child);
                    }
                    subtree.push(*node);
                    stack.push(subtree);
                }
            }
        }

        stack.pop().unwrap_or_else(|| nodes.to_vec())
    }

    #[inline(always)]
    fn eval_simd(
        code: &[Self::Instruction], 
        constants: &[Self::ScalarValue], 
        features: &[Self::SimdValue]
    ) -> Self::SimdValue {
        
        let mut stack_f: [f32x4; 32] = [f32x4::splat(0.0); 32];
        let mut sp_f: usize = 0;
        
        let mut stack_v3: [[f32x4; 3]; 32] = [[f32x4::splat(0.0); 3]; 32];
        let mut sp_v3: usize = 0;

        let mut stack_b: [f32x4; 32] = [f32x4::splat(0.0); 32];
        let mut sp_b: usize = 0;

        for op in code {
            match op {
                UniversalInstruction::LoadVarF(idx) => unsafe {
                    *stack_f.get_unchecked_mut(sp_f) = *features.get_unchecked(*idx as usize);
                    sp_f += 1;
                },
                UniversalInstruction::LoadConstF(idx) => unsafe {
                    if let UniversalScalar::Float(val) = constants.get_unchecked(*idx as usize) {
                        *stack_f.get_unchecked_mut(sp_f) = f32x4::splat(*val);
                    }
                    sp_f += 1;
                },
                UniversalInstruction::AddF => unsafe {
                    sp_f -= 2;
                    let a = *stack_f.get_unchecked(sp_f);
                    let b = *stack_f.get_unchecked(sp_f + 1);
                    *stack_f.get_unchecked_mut(sp_f) = a + b;
                    sp_f += 1;
                },
                UniversalInstruction::SubF => unsafe {
                    sp_f -= 2;
                    let a = *stack_f.get_unchecked(sp_f);
                    let b = *stack_f.get_unchecked(sp_f + 1);
                    *stack_f.get_unchecked_mut(sp_f) = a - b;
                    sp_f += 1;
                },
                UniversalInstruction::MulF => unsafe {
                    sp_f -= 2;
                    let a = *stack_f.get_unchecked(sp_f);
                    let b = *stack_f.get_unchecked(sp_f + 1);
                    *stack_f.get_unchecked_mut(sp_f) = a * b;
                    sp_f += 1;
                },
                UniversalInstruction::DivF => unsafe {
                    sp_f -= 2;
                    let a = *stack_f.get_unchecked(sp_f);
                    let b = *stack_f.get_unchecked(sp_f + 1);
                    
                    let epsilon = f32x4::splat(1e-9);
                    let ones = f32x4::splat(1.0);
                    let is_zero_mask = b.abs().simd_lt(epsilon);                    
                    let safe_b = is_zero_mask.blend(ones, b);
                    
                    *stack_f.get_unchecked_mut(sp_f) = a / safe_b;
                    sp_f += 1;
                },
                UniversalInstruction::SinF => unsafe {
                    let idx = sp_f - 1;
                    let a = *stack_f.get_unchecked(idx);
                    *stack_f.get_unchecked_mut(idx) = a.sin();
                },
                UniversalInstruction::CosF => unsafe {
                    let idx = sp_f - 1;
                    let a = *stack_f.get_unchecked(idx);
                    *stack_f.get_unchecked_mut(idx) = a.cos();
                },
                UniversalInstruction::ExpF => unsafe {
                    let idx = sp_f - 1;
                    let a = *stack_f.get_unchecked(idx);
                    *stack_f.get_unchecked_mut(idx) = a.exp();
                },
                UniversalInstruction::SqrF => unsafe {
                    let idx = sp_f - 1;
                    let a = *stack_f.get_unchecked(idx);
                    *stack_f.get_unchecked_mut(idx) = a * a;
                },
                _ => {} // A Linalg és Logic egyelőre maradhat így
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
            
            // Itt dől el a varázslat: az eval_simd végrehajtja a többtípusos programot, de Float-ot ad vissza
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