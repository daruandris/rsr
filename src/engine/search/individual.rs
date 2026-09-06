use crate::Instruction;
use crate::domains::basic::BasicOpCode;
use crate::domains::linalg::LinalgOpCode;
use crate::engine::data::dataset::Dataset;
use crate::engine::eval::evaluator;
use crate::engine::eval::scalar::Scalar;
use crate::engine::expr::format::format_ast;
use crate::engine::expr::node::Node;
use crate::engine::expr::program::Program;
use crate::engine::expr::simplify::simplify_ast;
use crate::engine::optimize::optimize_individual_constants;
use crate::engine::eval::types::ValueType;
use crate::engine::search::config::LossFunctionType;
use std::fmt;

#[derive(Clone)]
pub struct Individual {
    pub nodes: Vec<Node>,
    pub constants: Vec<Scalar>,
    pub disabled_constants: Vec<ValueType>,
    pub fitness: f32,
    pub age: usize,
    pub program: Option<Program>,
    pub rank: u32,
    pub crowding_distance: f32,
}

impl Individual {
    pub fn new(nodes: Vec<Node>, constants: Vec<Scalar>, disabled_constants: Vec<ValueType>) -> Self {
        Self {
            nodes,
            constants,
            disabled_constants,
            fitness: f32::MAX,
            age: 0,
            program: None,
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    pub fn compile(&mut self) {
        if self.program.is_none() {
            self.program = Some(Program::from_nodes(&self.nodes, &self.constants, &self.disabled_constants));
        }
    }

    pub fn calculate_loss(&mut self, dataset: &Dataset, loss_type: LossFunctionType) -> f32 {
        if self.program.is_none() {
            self.compile();
        }

        if let Some(prog) = &self.program {
            evaluator::compute_loss(prog, dataset, loss_type)
        } else {
            f32::MAX
        }
    }

    pub fn get_constants(&self) -> Vec<Scalar> {
        self.constants.clone()
    }

    pub fn set_constants(&mut self, new_constants: &[Scalar]) {
        self.constants.copy_from_slice(new_constants);
        self.invalidate();
    }

    pub fn optimize_constants(&mut self, dataset: &Dataset, iterations: usize, loss_type: LossFunctionType) {
        optimize_individual_constants(self, dataset, iterations, loss_type);
        let threshold = 1e-5;
        let mut constants = self.get_constants();
        for c in constants.iter_mut() {
            c.apply_threshold(threshold);
        }
        self.set_constants(&constants);
        self.simplify();
        self.compile();
    }

    pub fn simplify(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
        let (new_nodes, new_consts) = simplify_ast(&self.nodes, &self.constants, &self.disabled_constants);
        self.nodes = new_nodes;
        self.constants = new_consts;
        self.invalidate();
    }

    pub fn invalidate(&mut self) {
        self.program = None;
        self.fitness = f32::MAX;
    }

    pub fn get_subtree_bounds(&self, root_idx: usize) -> (usize, usize) {
        let mut needed = 1;
        let mut current_idx = root_idx;
        loop {
            needed += self.nodes[current_idx].arity() as isize - 1;
            if needed == 0 {
                return (current_idx, root_idx);
            }
            if current_idx == 0 {
                break;
            }
            current_idx -= 1;
        }
        (0, root_idx)
    }

    pub fn complexity(&self) -> usize {
        self.nodes.iter().map(|node| node.weight()).sum()
    }

    pub fn has_forbidden_patterns(&self) -> bool {
        const FLAG_TRIG: u16 = 1 << 0;
        const FLAG_EXP: u16 = 1 << 1;
        const FLAG_LN: u16 = 1 << 2;
        const FLAG_POWER: u16 = 1 << 3;
        const FLAG_TRANSPOSE: u16 = 1 << 4;
        const FLAG_INVERSE: u16 = 1 << 5;
        const FLAG_DET: u16 = 1 << 6;
        
        // ÚJ: Szétválasztott Solid flagek a fizikai hierarchia alapján
        const FLAG_SOLID_KINEMATIC: u16 = 1 << 7; // C, B, (és ide értendő a Cofactor is, ha van neki külön Solid op-ja)
        const FLAG_SOLID_INVARIANT: u16 = 1 << 8; // I1, I2, Tr, J

        let mut stack: Vec<u16> = Vec::with_capacity(32);

        for node in &self.nodes {
            match node {
                Node::Variable(_, _) | Node::Constant(_, _) => {
                    stack.push(0); // Alapváltozó, tiszta
                }
                Node::Operator(op) => {
                    let arity = op.arity();
                    if stack.len() < arity {
                        return true;
                    }

                    let mut child_flags = 0;
                    for _ in 0..arity {
                        // Az összeadás/szorzás operátorok itt szépen egyesítik (OR) a gyerekeik flagjeit!
                        child_flags |= stack.pop().unwrap();
                    }

                    match op {
                        Instruction::Solid(solid_op) => {
                            use crate::domains::solid::SolidOpCode::*;
                            match solid_op {
                                // 1. KINEMATIKAI TENZOROK (C, B)
                                // Ezeket csak nyers F-ből (vagy max transzponáltjából) szabad képezni.
                                RightCauchyGreenM3 | LeftCauchyGreenM3 => {
                                    // TILTÁS: Ne csináljunk C-t/B-t másik C-ből/B-ből, Invariánsból, vagy INVERZBŐL!
                                    if (child_flags & (FLAG_SOLID_KINEMATIC | FLAG_SOLID_INVARIANT | FLAG_INVERSE)) != 0 {
                                        return true;
                                    }
                                    stack.push(child_flags | FLAG_SOLID_KINEMATIC);
                                }
                                
                                // 2. INVARIÁNSOK (I1, I2, Trace)
                                // Ezek skalárok. Tilos őket egymásba ágyazni!
                                Invariant2M3 | TraceSqrM3 => {
                                    // TILTÁS: Invariáns belsejében ne legyen másik Invariáns vagy Determináns
                                    if (child_flags & (FLAG_SOLID_INVARIANT | FLAG_DET)) != 0 {
                                        return true;
                                    }
                                    stack.push(child_flags | FLAG_SOLID_INVARIANT);
                                }
                                
                                // 3. DEVIATORIKUS RÉSZ
                                DeviatoricM3 => {
                                    // Tilos kétszer deviátorosítani, vagy invariánst deviátorosítani (mivel az skalár)
                                    if (child_flags & FLAG_SOLID_INVARIANT) != 0 {
                                        return true;
                                    }
                                    stack.push(child_flags | FLAG_SOLID_KINEMATIC); // Ez továbbra is tenzor marad
                                }
                                
                                // Ha van nálad külön Cofactor operátor a Solid-ban:
                                CofactorM3 => {
                                //     // TILTÁS: Kofaktort inverzből, invariánsból, másik kinematikai tenzorból nem csinálunk!
                                     if (child_flags & (FLAG_SOLID_KINEMATIC | FLAG_SOLID_INVARIANT | FLAG_INVERSE)) != 0 {
                                         return true;
                                     }
                                     stack.push(child_flags | FLAG_SOLID_KINEMATIC);
                                }

                                _ => {
                                    stack.push(child_flags);
                                }
                            }
                        }
                        Instruction::Basic(basic_op) => match basic_op {
                            BasicOpCode::SinF | BasicOpCode::CosF => {
                                if (child_flags & (FLAG_TRIG | FLAG_EXP | FLAG_LN)) != 0 { return true; }
                                stack.push(child_flags | FLAG_TRIG);
                            }
                            BasicOpCode::ExpF => {
                                if (child_flags & (FLAG_TRIG | FLAG_EXP | FLAG_LN | FLAG_POWER)) != 0 { return true; }
                                stack.push(child_flags | FLAG_EXP);
                            }
                            BasicOpCode::LnF => {
                                if (child_flags & (FLAG_TRIG | FLAG_EXP | FLAG_LN)) != 0 { return true; }
                                stack.push(child_flags | FLAG_LN);
                            }
                            BasicOpCode::SqrtF | BasicOpCode::SqrF => {
                                if (child_flags & (FLAG_POWER | FLAG_TRIG | FLAG_LN | FLAG_EXP)) != 0 { return true; }
                                stack.push(child_flags | FLAG_POWER);
                            }
                            // HA VAN ADD / MUL / SUB, azok ide jönnek (gondolom, csak passzolják a child_flags-et)
                            _ => stack.push(child_flags),
                        },
                        Instruction::Linalg(linalg_op) => match linalg_op {
                            LinalgOpCode::TransposeM2 | LinalgOpCode::TransposeM3 => {
                                if (child_flags & FLAG_TRANSPOSE) != 0 { return true; }
                                stack.push(child_flags | FLAG_TRANSPOSE);
                            }
                            LinalgOpCode::InverseM2 | LinalgOpCode::InverseM3 => {
                                // Ne invertáljunk már meglevő kinematikai tenzort (C, B), vagy másik inverzt!
                                if (child_flags & (FLAG_INVERSE | FLAG_SOLID_KINEMATIC)) != 0 { return true; }
                                stack.push(child_flags | FLAG_INVERSE);
                            }
                            LinalgOpCode::DetM2 | LinalgOpCode::DetM3 => {
                                // Det(Det) tilos, Det(Trace) tilos.
                                if (child_flags & (FLAG_DET | FLAG_SOLID_INVARIANT)) != 0 { return true; }
                                stack.push(child_flags | FLAG_DET); // Ez skalárként viselkedik, úgyhogy Invariant kategória felé hajlik
                            }
                            _ => stack.push(child_flags),
                        },
                        _ => stack.push(child_flags),
                    }
                }
            }
        }
        false
    }
}

impl fmt::Display for Individual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format_ast(&self.nodes, &self.constants))
    }
}
