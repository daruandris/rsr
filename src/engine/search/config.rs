use crate::Instruction;
use crate::domains::basic::BasicOpCode;
use crate::domains::linalg::LinalgOpCode;
use crate::domains::solid::SolidOpCode;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpModule {
    Basic,
    Linalg,
    Logic,
    Solid,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub num_islands: usize,
    pub island_size: usize,
    pub max_generations: usize,
    pub crossover_rate: f32,
    pub tournament_size: usize,
    pub migration_interval: usize,
    pub base_parsimony_penalty: f32,
    pub opt_prob: f32,
    pub opt_iterations: usize,
    pub final_opt_iterations: usize,
    pub stagnation_threshold: usize,
    pub base_target_mse: f32,
    pub min_improvement: f32,
    pub random_injection_rate: f32,
    pub min_random_injection: usize,
    pub max_tree_size: usize,
    pub mutation_max_depth: usize,
    pub mutation_cycles: usize,
    pub verbose: bool,

    pub allowed_modules: Vec<OpModule>,
    pub custom_ops: Vec<Instruction>,
    pub excluded_ops: Vec<Instruction>,
    pub subset_size: Option<usize>,
    pub mini_batch_size: usize,
}

impl Config {
    pub fn default(allowed_modules: Vec<OpModule>) -> Self {
        Config {
            num_islands: 24,
            island_size: 25,
            max_generations: 3000,
            crossover_rate: 0.10,
            tournament_size: 2,
            migration_interval: 25,
            base_parsimony_penalty: 0.00005,
            opt_prob: 0.01,
            opt_iterations: 100,
            final_opt_iterations: 4000,
            stagnation_threshold: 1000,
            base_target_mse: 1e-7,
            min_improvement: 1e-6,
            random_injection_rate: 0.10,
            min_random_injection: 2,
            max_tree_size: 32,
            mutation_max_depth: 4,
            mutation_cycles: 2,
            verbose: true,
            allowed_modules,
            custom_ops: vec![],
            excluded_ops: vec![],
            subset_size: Some(400),
            mini_batch_size: 64,
        }
    }

    pub fn with_module(mut self, module: OpModule) -> Self {
        if !self.allowed_modules.contains(&module) {
            self.allowed_modules.push(module);
        }
        self
    }

    pub fn with_op(mut self, op: Instruction) -> Self {
        if !self.custom_ops.contains(&op) {
            self.custom_ops.push(op);
        }
        self
    }

    pub fn without_op(mut self, op: Instruction) -> Self {
        if !self.excluded_ops.contains(&op) {
            self.excluded_ops.push(op);
        }
        self
    }

    pub fn with_modules(mut self, modules: Vec<OpModule>) -> Self {
        for module in modules {
            if !self.allowed_modules.contains(&module) {
                self.allowed_modules.push(module);
            }
        }
        self
    }

    pub fn with_ops(mut self, ops: Vec<Instruction>) -> Self {
        for op in ops {
            if !self.custom_ops.contains(&op) {
                self.custom_ops.push(op);
            }
        }
        self
    }

    pub fn without_ops(mut self, ops: Vec<Instruction>) -> Self {
        for op in ops {
            if !self.excluded_ops.contains(&op) {
                self.excluded_ops.push(op);
            }
        }
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    pub fn base_parsimony_penalty(mut self, penalty: f32) -> Self {
        self.base_parsimony_penalty = penalty;
        self
    }

    pub fn mini_batch_size(mut self, size: usize) -> Self {
        self.mini_batch_size = size;
        self
    }

    // ==========================================
    // ANYAGCSALÁDOK SZERINTI PROFILOK
    // ==========================================

    /// Gumik, polimerek, elasztomerek, lágy szövetek (Nagy alakváltozás)
    pub fn hyperelastic_isotropic(mut self) -> Self {
        self.allowed_modules = vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid];
        //should disable tensor constants
        let mut exclusions = vec![
            // Kis alakváltozási tenzor tiltása a modell objektivitása miatt
            Instruction::Solid(SolidOpCode::GreenLagrangeStrainM3),
            Instruction::Solid(SolidOpCode::TraceSqrM3),
            Instruction::Solid(SolidOpCode::DeviatoricM3),
            
            // Szögfüggvények tiltása
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
        ];

        // Összes térbeli vektor operátor tiltása (objektivitás garantálása)
        let vector_ops = [
            LinalgOpCode::MakeVec2, LinalgOpCode::MakeVec3,
            LinalgOpCode::GetXV2, LinalgOpCode::GetYV2,
            LinalgOpCode::GetXV3, LinalgOpCode::GetYV3, LinalgOpCode::GetZV3,
            LinalgOpCode::AddV2, LinalgOpCode::SubV2, LinalgOpCode::ScaleV2, 
            LinalgOpCode::DotV2, LinalgOpCode::NormV2,
            LinalgOpCode::AddV3, LinalgOpCode::SubV3, LinalgOpCode::ScaleV3, 
            LinalgOpCode::DotV3, LinalgOpCode::NormV3, LinalgOpCode::CrossV3,
            LinalgOpCode::MulM2V2, LinalgOpCode::MulM3V3,
        ];
        
        for op in vector_ops {
            exclusions.push(Instruction::Linalg(op));
        }

        self.excluded_ops.extend(exclusions);
        self
    }

    /// Fémek, merev műanyagok, kerámiák (Kis alakváltozás)
    pub fn linear_stiff_metals(mut self) -> Self {
        self.allowed_modules = vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid];
         //should disable tensor constants
        self.excluded_ops.extend(vec![
            // C és B tenzorok, illetve isochor invariánsok tiltása
            Instruction::Solid(SolidOpCode::RightCauchyGreenM3),
            Instruction::Solid(SolidOpCode::LeftCauchyGreenM3),
            Instruction::Solid(SolidOpCode::IsochoricInvariant1),
            Instruction::Solid(SolidOpCode::IsochoricInvariant2),
            Instruction::Solid(SolidOpCode::CofactorM3),
            
            // Numerikus instabilitást okozó függvények tiltása fémeknél
            Instruction::Basic(BasicOpCode::ExpF),
            Instruction::Basic(BasicOpCode::LnF),
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
        ]);
        self
    }

    /// Szálerősített kompozitok, 3D nyomtatott alkatrészek, fa
    pub fn anisotropic_composites(mut self) -> Self {
        // Alapvetően a hiperelasztikus modellt vesszük alapul, de a vektoros
        // operátorokat (MakeVec3, DotV3, CrossV3) ENGEDÉLYEZZÜK az irányvektorok miatt.
        self.allowed_modules = vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid];
         //should disable tensor constants
        let mut exclusions = vec![
            Instruction::Solid(SolidOpCode::GreenLagrangeStrainM3),
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
        ];

        // Csak a nem releváns 2D és transzformációs vektor operátorokat tiltjuk
        let vector_ops = [
            LinalgOpCode::MakeVec2, LinalgOpCode::GetXV2, LinalgOpCode::GetYV2,
            LinalgOpCode::AddV2, LinalgOpCode::SubV2, LinalgOpCode::ScaleV2, 
            LinalgOpCode::DotV2, LinalgOpCode::NormV2, LinalgOpCode::MulM2V2,
        ];
        
        for op in vector_ops {
            exclusions.push(Instruction::Linalg(op));
        }

        self.excluded_ops.extend(exclusions);
        self
    }

    // ==========================================
    // FELADATTÍPUSOK SZERINTI PROFILOK
    // ==========================================

    /// Energiasűrűség-függvény keresése (Target: Scalar)
    pub fn strain_energy_discovery(mut self) -> Self {
        // Klasszikus 2-4 paraméteres modellek előnyben részesítése (magas parsimony penalty)
        self.base_parsimony_penalty = 0.05; // Jelentősen magasabb büntetés a komplexitásra[cite: 1]
        
        // Magas L-BFGS iterációszám a konstansok precíz belövéséhez
        self.opt_iterations = 300; 
        self.final_opt_iterations = 10_000;
        self.opt_prob = 0.05; // Gyakoribb lokális optimalizáció
        
        self
    }

    /// Feszültségtenzor közvetlen keresése (Target: Mat3)
    pub fn constitutive_stress_law(mut self) -> Self {
        // Ennél a feladatnál elengedhetetlenek a mátrix aritmetikai operátorok
        // Ez a profil inkább biztosítja, hogy a konfiguráció támogatja a tenzoriális kimenetet.
        self.base_parsimony_penalty = 0.005; // Standard büntetés[cite: 1]
        self.opt_iterations = 100;
        
        // Mátrix inverz és szorzás kötelező meglétének biztosítása (ha nem lennének benne)
        let required_ops = vec![
            Instruction::Linalg(LinalgOpCode::AddM3),
            Instruction::Linalg(LinalgOpCode::SubM3),
            Instruction::Linalg(LinalgOpCode::ScaleM3),
            Instruction::Linalg(LinalgOpCode::InverseM3),
            Instruction::Linalg(LinalgOpCode::TransposeM3),
        ];
        
        for op in required_ops {
            self.excluded_ops.retain(|x| x != &op);
        }
        self
    }

    /// Folyási és tönkremeneteli felületek (Target: Scalar)
    pub fn yield_surface_discovery(mut self) -> Self {
        // Szorosan támaszkodik a TraceM3-ra és a deviatorikus tenzorokra
        self.base_parsimony_penalty = 0.01;
        
        // A bemenet itt jellemzően a feszültségtenzor (sigma),
        // ezért a kinematikai tenzorokat (C, B, E) felesleges és tiltott generálni.
        self.excluded_ops.extend(vec![
            Instruction::Solid(SolidOpCode::RightCauchyGreenM3),
            Instruction::Solid(SolidOpCode::LeftCauchyGreenM3),
            Instruction::Solid(SolidOpCode::GreenLagrangeStrainM3),
            Instruction::Solid(SolidOpCode::IsochoricInvariant1),
            Instruction::Solid(SolidOpCode::IsochoricInvariant2),
        ]);
        
        self
    }
}
