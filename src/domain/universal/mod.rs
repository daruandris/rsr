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
    AddF, SubF, MulF, DivF, SinF, CosF, ExpF, SqrF, LnF, SqrtF,
    // Linalg
    AddV3, DotV3, ScaleV3,
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
    LoadVarV3(u8), LoadConstV3(u16),
    LoadVarB(u8), LoadConstB(u16),
    LoadVarI(u8), LoadConstI(u16),
    AddF, SubF, MulF, DivF, SinF, CosF, ExpF, SqrF, LnF, SqrtF,
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
            UniversalOp::SinF | UniversalOp::CosF | UniversalOp::ExpF | UniversalOp::SqrF |
            UniversalOp::LnF | UniversalOp::SqrtF => 1,
            UniversalOp::AddF | UniversalOp::SubF | UniversalOp::MulF | UniversalOp::DivF | 
            UniversalOp::AddV3 | UniversalOp::DotV3 | UniversalOp::ScaleV3 => 2,
            UniversalOp::IfElseF => 3,
        }
    }

    #[inline(always)]
    fn operator_weight(op: &Self::Operator) -> usize {
        match op {
            UniversalOp::AddF | UniversalOp::SubF | UniversalOp::MulF | UniversalOp::AddV3 => 1,
            UniversalOp::DivF | UniversalOp::DotV3 | UniversalOp::ScaleV3 | UniversalOp::SqrF |
            UniversalOp::SqrtF => 2,
            UniversalOp::SinF | UniversalOp::CosF => 3,
            UniversalOp::IfElseF | UniversalOp::ExpF | UniversalOp::LnF => 4,
        }
    }

    fn format_operator(op: &Self::Operator, args: &[String]) -> String {
        match op {
            UniversalOp::AddF => format!("({} + {})", args[0], args[1]),
            UniversalOp::SubF => format!("({} - {})", args[0], args[1]),
            UniversalOp::SinF => format!("sin({})", args[0]),
            UniversalOp::CosF => format!("cos({})", args[0]),
            UniversalOp::DotV3 => format!("({} • {})", args[0], args[1]),
            UniversalOp::ScaleV3 | UniversalOp::MulF => format!("({} * {})", args[0], args[1]),
            UniversalOp::IfElseF => format!("(if {} then {} else {})", args[0], args[1], args[2]),
            UniversalOp::ExpF => format!("exp({})", args[0]),
            UniversalOp::SqrF => format!("({})^2", args[0]),
            UniversalOp::DivF => format!("({} / {})", args[0], args[1]),
            UniversalOp::SqrtF => format!("sqrt(|{}|)", args[0]),
            UniversalOp::LnF => format!("ln(|{}|)", args[0]),
            _ => format!("{:?}({})", op, args.join(", ")),
        }
    }

    // --- STGP: Típus szignatúrák (Ez fogja vezérelni a generátort!) ---
    #[inline(always)]
    fn return_type(op: &Self::Operator) -> Self::TypeId {
        match op {
            UniversalOp::AddV3 | UniversalOp::ScaleV3 => UniversalType::Vec3,
            _ => UniversalType::Float,
        }
    }

    #[inline(always)]
    fn expected_types(op: &Self::Operator) -> Vec<Self::TypeId> {
       match op {
            UniversalOp::AddF | UniversalOp::SubF | UniversalOp::MulF | UniversalOp::DivF => vec![UniversalType::Float, UniversalType::Float],
            UniversalOp::SinF | UniversalOp::CosF | UniversalOp::ExpF | 
            UniversalOp::SqrF | UniversalOp::LnF | UniversalOp::SqrtF => vec![UniversalType::Float],
            UniversalOp::AddV3 | UniversalOp::DotV3 => vec![UniversalType::Vec3, UniversalType::Vec3],
            UniversalOp::ScaleV3 => vec![UniversalType::Float, UniversalType::Vec3],
            UniversalOp::IfElseF => vec![UniversalType::Bool, UniversalType::Float, UniversalType::Float],
        }
    }

    // --- ALAPÉRTELMEZETT VÁLTOZÓ ÉS KONSTANS TÍPUSOK ---
    // FIGYELEM: Ahhoz, hogy a generátor tudja, milyen típusú változót/konstanst húzzon,
    // a Domain trait-et majd finomítani kell. Jelenleg Float-ot adunk vissza alapként.
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
            UniversalOp::LnF => UniversalInstruction::LnF,
            UniversalOp::SqrtF => UniversalInstruction::SqrtF,
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

                    let mut children = Vec::with_capacity(arity);
                    for _ in 0..arity {
                        children.push(stack.pop().unwrap());
                    }
                    children.reverse(); // LIFO miatt visszafordítjuk

                    // === 1. KONSTANS ÖSSZEVONÁS (Constant Folding) ===
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
                            UniversalOp::DivF => if vals[1].abs() > 1e-9 { Some(vals[0] / vals[1]) } else { None },
                            UniversalOp::SinF => Some(vals[0].sin()),
                            UniversalOp::CosF => Some(vals[0].cos()),
                            UniversalOp::ExpF => Some(vals[0].exp()),
                            UniversalOp::SqrF => Some(vals[0] * vals[0]),
                            UniversalOp::SqrtF => Some(vals[0].abs().sqrt()),
                            UniversalOp::LnF => Some((vals[0].abs() + 1e-9).ln()),
                            _ => None,
                        };

                        if let Some(f) = folded {
                            if f.is_finite() {
                                stack.push(vec![Node::Constant(UniversalScalar::Float(f), UniversalType::Float)]);
                                continue; // Kész, kiváltottuk a részfát egy konstanssal!
                            }
                        }
                    }

                    // === 2. ALGEBRAI EGYSZERŰSÍTÉSEK ===
                    let mut simplified = false;

                    // Egyváltozós inverz szabályok (pl. ln(exp(x)) -> x)
                    if arity == 1 {
                        let child_expr = &children[0];
                        if let Some(Node::Operator(child_op)) = child_expr.last() {
                            match (op, child_op) {
                                (UniversalOp::LnF, UniversalOp::ExpF) |
                                (UniversalOp::ExpF, UniversalOp::LnF) |
                                (UniversalOp::SqrtF, UniversalOp::SqrF) |
                                (UniversalOp::SqrF, UniversalOp::SqrtF) => {
                                    // Mivel postfix, a gyermek operátor az utolsó elem. 
                                    // Ezt levágjuk, a maradék maga az 'x'.
                                    let mut inner_x = child_expr.clone();
                                    inner_x.pop(); 
                                    stack.push(inner_x);
                                    simplified = true;
                                },
                                _ => {}
                            }
                        }
                    } 
                    // Kétváltozós szabályok (x*0, x+0, x/x, x-x)
                    else if arity == 2 {
                        let left_is_const = children[0].len() == 1 && matches!(children[0][0], Node::Constant(UniversalScalar::Float(_), _));
                        let right_is_const = children[1].len() == 1 && matches!(children[1][0], Node::Constant(UniversalScalar::Float(_), _));
                        
                        let left_val = if left_is_const { if let Node::Constant(UniversalScalar::Float(v), _) = children[0][0] { v } else { 0.0 } } else { 0.0 };
                        let right_val = if right_is_const { if let Node::Constant(UniversalScalar::Float(v), _) = children[1][0] { v } else { 0.0 } } else { 0.0 };

                        match op {
                            UniversalOp::AddF => {
                                if right_is_const && right_val.abs() < 1e-6 { stack.push(children[0].clone()); simplified = true; } // x + 0
                                else if left_is_const && left_val.abs() < 1e-6 { stack.push(children[1].clone()); simplified = true; } // 0 + x
                            },
                            UniversalOp::SubF => {
                                if right_is_const && right_val.abs() < 1e-6 { stack.push(children[0].clone()); simplified = true; } // x - 0
                                else if children[0] == children[1] { 
                                    stack.push(vec![Node::Constant(UniversalScalar::Float(0.0), UniversalType::Float)]); simplified = true; // x - x = 0
                                }
                            },
                            UniversalOp::MulF => {
                                if right_is_const {
                                    if (right_val - 1.0).abs() < 1e-6 { stack.push(children[0].clone()); simplified = true; } // x * 1
                                    else if right_val.abs() < 1e-6 { stack.push(vec![Node::Constant(UniversalScalar::Float(0.0), UniversalType::Float)]); simplified = true; } // x * 0
                                } else if left_is_const {
                                    if (left_val - 1.0).abs() < 1e-6 { stack.push(children[1].clone()); simplified = true; } // 1 * x
                                    else if left_val.abs() < 1e-6 { stack.push(vec![Node::Constant(UniversalScalar::Float(0.0), UniversalType::Float)]); simplified = true; } // 0 * x
                                }
                            },
                            UniversalOp::DivF => {
                                if right_is_const && (right_val - 1.0).abs() < 1e-6 { stack.push(children[0].clone()); simplified = true; } // x / 1
                                else if children[0] == children[1] { 
                                    stack.push(vec![Node::Constant(UniversalScalar::Float(1.0), UniversalType::Float)]); simplified = true; // x / x = 1
                                }
                                else if left_is_const && left_val.abs() < 1e-6 {
                                    stack.push(vec![Node::Constant(UniversalScalar::Float(0.0), UniversalType::Float)]); simplified = true; // 0 / x = 0
                                }
                            },
                            _ => {}
                        }
                    }

                    // === 3. HA NINCS EGYSZERŰSÍTÉS, RAKJUK ÖSSZE ===
                    if !simplified {
                        let mut subtree = Vec::new();
                        for mut child in children {
                            subtree.append(&mut child);
                        }
                        subtree.push(*node);
                        stack.push(subtree);
                    }
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
                UniversalInstruction::SqrtF => unsafe {
                    let idx = sp_f - 1;
                    let a = *stack_f.get_unchecked(idx);
                    *stack_f.get_unchecked_mut(idx) = a.abs().sqrt(); // Ne legyen NaN negatívokból!
                },
                UniversalInstruction::LnF => unsafe {
                    let idx = sp_f - 1;
                    let a = *stack_f.get_unchecked(idx);
                    let safe_a = a.abs() + f32x4::splat(1e-9); // Védelem a log(0) ellen
                    *stack_f.get_unchecked_mut(idx) = safe_a.ln();
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