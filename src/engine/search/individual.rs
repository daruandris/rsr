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

    pub fn has_solid_physics_error(&self) -> bool {
        #[derive(Clone, Copy, PartialEq)]
        enum Space { Material, Spatial, Mixed }

        #[derive(Clone, Copy, PartialEq)]
        enum TensorType {
            Scalar,
            Mat3(Space, Space), // (Left Space, Right Space)
            Other,
        }

        struct PhysNode {
            ttype: TensorType,
            history: u32,
        }

        // Történeti (history) flagek az egymásba ágyazás megakadályozásához
        const HIST_KINEMATIC: u32 = 1 << 0; // C, B, E
        const HIST_DEVIATORIC: u32 = 1 << 1; // dev
        const HIST_COFACTOR: u32 = 1 << 2; // Cof
        const HIST_INVERSE: u32 = 1 << 3; // Inv
        const HIST_TRANSPOSE: u32 = 1 << 4; // Transpose
        const HIST_INVARIANT: u32 = 1 << 5; // I1, I2, Tr, J2, J3, Det, I4, I5, I6, I7

        let mut stack: Vec<PhysNode> = Vec::with_capacity(32);

        // Két tér akkor illeszthető össze, ha azonosak, vagy valamelyik vegyes (pl. konstans)
        let spaces_match = |a: Space, b: Space| -> bool {
            a == b || a == Space::Mixed || b == Space::Mixed
        };

        for node in &self.nodes {
            match node {
                Node::Variable(_, type_id) => {
                    let ttype = match type_id {
                        ValueType::Float => TensorType::Scalar,
                        // Bemeneti F tenzor (Deformation Gradient) tere: (Spatial, Material)
                        ValueType::Mat3 => TensorType::Mat3(Space::Spatial, Space::Material),
                        _ => TensorType::Other,
                    };
                    stack.push(PhysNode { ttype, history: 0 });
                }
                Node::Constant(_, type_id) => {
                    let ttype = match type_id {
                        ValueType::Float => TensorType::Scalar,
                        // Generált konstans tenzor bármely térbe beilleszkedhet
                        ValueType::Mat3 => TensorType::Mat3(Space::Mixed, Space::Mixed),
                        _ => TensorType::Other,
                    };
                    stack.push(PhysNode { ttype, history: 0 });
                }
                Node::Operator(op) => {
                    let arity = op.arity();
                    if stack.len() < arity { return true; } // Helytelen AST
                    
                    let mut children = Vec::with_capacity(arity);
                    for _ in 0..arity {
                        children.push(stack.pop().unwrap());
                    }
                    children.reverse();

                    let mut combined_history = 0;
                    for child in &children {
                        combined_history |= child.history;
                    }

                    let mut new_history = combined_history;
                    let mut res_type = TensorType::Other;

                    match op {
                        Instruction::Solid(solid_op) => {
                            use crate::domains::solid::SolidOpCode::*;
                            match solid_op {
                                RightCauchyGreenM3 | LeftCauchyGreenM3 | GreenLagrangeStrainM3 => {
                                    // Tilos másik kinematikai tenzorba, invariánsba, vagy inverzbe tenni
                                    if (combined_history & (HIST_KINEMATIC | HIST_INVARIANT | HIST_INVERSE)) != 0 {
                                        return true;
                                    }
                                    
                                    if let TensorType::Mat3(l, r) = children[0].ttype {
                                        // SZIGORÍTÁS: Ezek az operátorok kizárólag [Spatial, Material] (F-jellegű) tenzoron értelmezettek!
                                        if !spaces_match(l, Space::Spatial) || !spaces_match(r, Space::Material) {
                                            return true;
                                        }

                                        if *solid_op == RightCauchyGreenM3 || *solid_op == GreenLagrangeStrainM3 {
                                            res_type = TensorType::Mat3(Space::Material, Space::Material); // C és E anyagi tenzorok
                                        } else {
                                            res_type = TensorType::Mat3(Space::Spatial, Space::Spatial); // B térbeli tenzor
                                        }
                                    } else { 
                                        return true; 
                                    }
                                    new_history |= HIST_KINEMATIC;
                                }
                                CofactorM3 => {
                                    if (combined_history & (HIST_KINEMATIC | HIST_INVARIANT | HIST_INVERSE | HIST_COFACTOR)) != 0 {
                                        return true;
                                    }
                                    if let TensorType::Mat3(l, r) = children[0].ttype {
                                        res_type = TensorType::Mat3(r, l); // Cofactor felcseréli az indexeket (mint inverz)
                                    } else { return true; }
                                    new_history |= HIST_COFACTOR;
                                }
                                DeviatoricM3 => {
                                    // Nem deviátorosítjuk kétszer a tenzort
                                    if (combined_history & HIST_DEVIATORIC) != 0 { return true; }
                                    if let TensorType::Mat3(l, r) = children[0].ttype {
                                        res_type = TensorType::Mat3(l, r); // A dev() a teret békén hagyja
                                    } else { return true; }
                                    new_history |= HIST_DEVIATORIC;
                                }
                                Invariant2M3 | InvariantJ2M3 | InvariantJ3M3 | TraceSqrM3 => {
                                    if (combined_history & HIST_INVARIANT) != 0 { return true; }
                                    if let TensorType::Mat3(l, r) = children[0].ttype {
                                        // Invariáns képzés csak transzponált szimmetrikus jellegű tenzorokon értelmes (pl M,M)
                                        if !spaces_match(l, r) { return true; } // F (S,M) invariánsa helytelen
                                    } else { return true; }
                                    res_type = TensorType::Scalar;
                                    new_history |= HIST_INVARIANT;
                                }
                                IsochoricInvariant1 | IsochoricInvariant2 => {
                                    if (combined_history & HIST_INVARIANT) != 0 { return true; }
                                    if let TensorType::Mat3(l, r) = children[0].ttype {
                                        // Ezek a függvények F-re (S,M) vannak optimalizálva
                                        if !spaces_match(l, Space::Spatial) || !spaces_match(r, Space::Material) {
                                            return true;
                                        }
                                    } else { return true; }
                                    res_type = TensorType::Scalar;
                                    new_history |= HIST_INVARIANT;
                                }
                                // A match solid_op blokkban bővítsd ki a következőkkel:
                               InvariantI4 | InvariantI5 | InvariantI6 | InvariantI7 => {
                                    if (combined_history & HIST_INVARIANT) != 0 { return true; }
                                    
                                    // A children[0] a Mat3, a children[1] a Vec3
                                    if let TensorType::Mat3(l, r) = children[0].ttype {
                                        // Ezek a pszeudo-invariánsok csak referenciatérbeli (Anyagi) tenzorokon értelmezettek
                                        if !spaces_match(l, Space::Material) || !spaces_match(r, Space::Material) {
                                            return true;
                                        }
                                    } else { 
                                        return true; 
                                    }
                                    
                                    res_type = TensorType::Scalar;
                                    new_history |= HIST_INVARIANT;
                                }
                                IdentityM3 => {
                                    // Mivel nincs bemenete, egyből adhatjuk neki a teret
                                    res_type = TensorType::Mat3(Space::Mixed, Space::Mixed);
                                }
                            }
                        }
                        Instruction::Linalg(linalg_op) => {
                            use crate::domains::linalg::LinalgOpCode::*;
                            match linalg_op {
                                AddM3 | SubM3 => {
                                    if let (TensorType::Mat3(l1, r1), TensorType::Mat3(l2, r2)) = (children[0].ttype, children[1].ttype) {
                                        // Pl: F + F^T elvérzik itt, mert (S,M) != (M,S)
                                        if !spaces_match(l1, l2) || !spaces_match(r1, r2) {
                                            return true; 
                                        }
                                        let res_l = if l1 != Space::Mixed { l1 } else { l2 };
                                        let res_r = if r1 != Space::Mixed { r1 } else { r2 };
                                        res_type = TensorType::Mat3(res_l, res_r);
                                    } else { res_type = TensorType::Mat3(Space::Mixed, Space::Mixed); }
                                }
                                MulM3 => {
                                    if let (TensorType::Mat3(l1, r1), TensorType::Mat3(l2, r2)) = (children[0].ttype, children[1].ttype) {
                                        // Tenzorszorzásnál a belső indexeknek (r1 és l2) stimmelniük kell
                                        // Pl: F * F elvérzik, mert (S,M) * (S,M) -> M!=S
                                        if !spaces_match(r1, l2) {
                                            return true; 
                                        }
                                        let res_l = if l1 != Space::Mixed { l1 } else { Space::Mixed }; 
                                        let res_r = if r2 != Space::Mixed { r2 } else { Space::Mixed }; 
                                        res_type = TensorType::Mat3(res_l, res_r);
                                    } else { res_type = TensorType::Mat3(Space::Mixed, Space::Mixed); }
                                }
                                ScaleM3 => { res_type = children[1].ttype; }
                                TransposeM3 => {
                                    if (combined_history & HIST_TRANSPOSE) != 0 { return true; }
                                    if let TensorType::Mat3(l, r) = children[0].ttype {
                                        res_type = TensorType::Mat3(r, l);
                                    } else { res_type = TensorType::Mat3(Space::Mixed, Space::Mixed); }
                                    new_history |= HIST_TRANSPOSE;
                                }
                                InverseM3 => {
                                    if (combined_history & HIST_INVERSE) != 0 { return true; }
                                    if let TensorType::Mat3(l, r) = children[0].ttype {
                                        res_type = TensorType::Mat3(r, l);
                                    } else { res_type = TensorType::Mat3(Space::Mixed, Space::Mixed); }
                                    new_history |= HIST_INVERSE;
                                }
                                TraceM3 => {
                                    if (combined_history & HIST_INVARIANT) != 0 { return true; }
                                    if let TensorType::Mat3(l, r) = children[0].ttype {
                                        if !spaces_match(l, r) { return true; } // Pl: tr(F) letiltva
                                    }
                                    res_type = TensorType::Scalar;
                                    new_history |= HIST_INVARIANT;
                                }
                                DetM3 => {
                                    if (combined_history & HIST_INVARIANT) != 0 { return true; }
                                    res_type = TensorType::Scalar;
                                    new_history |= HIST_INVARIANT;
                                }
                                _ => {
                                    res_type = match op.return_type() {
                                        ValueType::Float => TensorType::Scalar,
                                        ValueType::Mat3 => TensorType::Mat3(Space::Mixed, Space::Mixed),
                                        _ => TensorType::Other,
                                    };
                                }
                            }
                        }
                        _ => {
                            res_type = match op.return_type() {
                                ValueType::Float => TensorType::Scalar,
                                ValueType::Mat3 => TensorType::Mat3(Space::Mixed, Space::Mixed),
                                _ => TensorType::Other,
                            };
                        }
                    }
                    stack.push(PhysNode { ttype: res_type, history: new_history });
                }
            }
        }
        false
    }

    pub fn has_forbidden_patterns(&self) -> bool {
        // Hívjuk meg a fizikai engine-t. Ha nem Solid domaint futtatsz (pl Navier-Stokes),
        // ezt az if ágat elég kikommentezni/konfigurációhoz kötni.
        if self.has_solid_physics_error() {
            return true;
        }

        const FLAG_TRIG: u16 = 1 << 0;
        const FLAG_EXP: u16 = 1 << 1;
        const FLAG_LN: u16 = 1 << 2;
        const FLAG_POWER: u16 = 1 << 3;
        const FLAG_TRANSPOSE: u16 = 1 << 4;
        const FLAG_INVERSE: u16 = 1 << 5;
        const FLAG_DET: u16 = 1 << 6;

        let mut stack: Vec<u16> = Vec::with_capacity(32);

        for node in &self.nodes {
            match node {
                Node::Variable(_, _) | Node::Constant(_, _) => {
                    stack.push(0);
                }
                Node::Operator(op) => {
                    let arity = op.arity();
                    if stack.len() < arity {
                        return true;
                    }

                    let mut child_flags = 0;
                    for _ in 0..arity {
                        child_flags |= stack.pop().unwrap();
                    }

                    match op {
                        Instruction::Basic(basic_op) => match basic_op {
                            BasicOpCode::SinF | BasicOpCode::CosF => {
                                if (child_flags & (FLAG_TRIG | FLAG_EXP | FLAG_LN)) != 0 { return true; }
                                stack.push(child_flags | FLAG_TRIG);
                            }
                            BasicOpCode::ExpF => {
                                if (child_flags & (FLAG_TRIG | FLAG_EXP | FLAG_LN )) != 0 { return true; }
                                stack.push(child_flags | FLAG_EXP);
                            }
                            BasicOpCode::LnF => {
                                if (child_flags & (FLAG_TRIG | FLAG_EXP | FLAG_LN)) != 0 { return true; }
                                stack.push(child_flags | FLAG_LN);
                            }
                            BasicOpCode::SqrtF | BasicOpCode::SqrF => {
                                if (child_flags & (FLAG_POWER | FLAG_TRIG | FLAG_LN )) != 0 { return true; }
                                stack.push(child_flags | FLAG_POWER);
                            }
                            _ => stack.push(child_flags),
                        },
                        Instruction::Linalg(linalg_op) => match linalg_op {
                            LinalgOpCode::TransposeM2 | LinalgOpCode::TransposeM3 => {
                                if (child_flags & FLAG_TRANSPOSE) != 0 { return true; }
                                stack.push(child_flags | FLAG_TRANSPOSE);
                            }
                            LinalgOpCode::InverseM2 | LinalgOpCode::InverseM3 => {
                                if (child_flags & FLAG_INVERSE) != 0 { return true; }
                                stack.push(child_flags | FLAG_INVERSE);
                            }
                            LinalgOpCode::DetM2 | LinalgOpCode::DetM3 => {
                                if (child_flags & FLAG_DET) != 0 { return true; }
                                stack.push(child_flags | FLAG_DET);
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
