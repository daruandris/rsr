use crate::{Instruction, ValueType};
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
    pub disabled_constant_types: Vec<ValueType>,
}

impl Config {
    pub fn default(allowed_modules: Vec<OpModule>) -> Self {
        Config {
            num_islands: 32,
            island_size: 500,
            max_generations: 5000,
            crossover_rate: 0.10,
            tournament_size: 2,
            migration_interval: 25,
            base_parsimony_penalty: 0.0005,
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
            disabled_constant_types: vec![],
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

    pub fn without_constants(mut self, types: Vec<ValueType>) -> Self {
        for t in types {
            if !self.disabled_constant_types.contains(&t) {
                self.disabled_constant_types.push(t);
            }
        }
        self
    }

    // ==========================================
    // IPARI FELADAT-SPECIFIKUS PROFILOK (TASK-BASED CONFIGS)
    // ==========================================

    /// 2. Folyási Felület és Tönkremenetel Felfedezése (Yield Surface Discovery)
    /// 
    /// Cél: f (skalár folyási feltétel) előállítása Sigma (Feszültség) tenzorból.
    /// Ipar: Fémfeldolgozás, talajmechanika, törésmechanika.
    /// Szabályok: Feszültség-invariánsokon (J2, von Mises, hidrosztatikus nyomás) alapul.
    /// Tilos: Kinematikai tenzorok (C, B), izochor invariánsok.
    pub fn yield_surface() -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
        
        config.disabled_constant_types = vec![
            ValueType::Vec2, ValueType::Vec3, ValueType::Mat2, ValueType::Mat3
        ];

        let mut exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
            Instruction::Basic(BasicOpCode::LnF),
            Instruction::Basic(BasicOpCode::ExpF),
            Instruction::Basic(BasicOpCode::DivF),
            
            // Folyási felületeknél a bemenet feszültség, így a deformációs 
            // operátorok (C, B, I1_bar) fizikailag értelmezhetetlenek itt.
            Instruction::Solid(SolidOpCode::RightCauchyGreenM3),
            Instruction::Solid(SolidOpCode::LeftCauchyGreenM3),
            Instruction::Solid(SolidOpCode::IsochoricInvariant1),
            Instruction::Solid(SolidOpCode::IsochoricInvariant2),
            Instruction::Solid(SolidOpCode::GreenLagrangeStrainM3),
        ];

        // Vektoros operátorok tiltása
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

        config.excluded_ops = exclusions;
        config
    }

    /// 3. Közvetlen Feszültségtenzor Modellezés (Constitutive Tensor Law)
    /// 
    /// Cél: Sigma (Mátrix) előállítása F vagy E (Mátrix) bemenetből.
    /// Ipar: Anizotróp anyagok, viszkoelaszticitás komplex leírása.
    /// Szabályok: Szabad mátrix-aritmetika (szorzás, inverz, transzponált).
    pub fn constitutive_tensor_law() -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);

        // Itt engedélyezhetjük a mátrix konstansokat az anizotrópiához (pl. szálirányok)
        config.disabled_constant_types = vec![
            ValueType::Vec2, ValueType::Vec3
        ];

        let mut exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
            Instruction::Basic(BasicOpCode::LnF),
        ];

        // Vektoros operátorok tiltása
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

        config.excluded_ops = exclusions;
        config
    }
}
