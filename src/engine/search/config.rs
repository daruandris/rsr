use crate::domains::solid::SolidOpCode;
use crate::{Instruction, ValueType};
use crate::domains::basic::BasicOpCode;
use crate::domains::linalg::LinalgOpCode;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LossFunctionType {
    /// Default Mean Squared Error
    DirectMse,
    TensorMseMat2,
    TensorMseMat3,
    PlanarBiaxialMse,
}

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
    pub loss_type: LossFunctionType,
    pub target_type: ValueType,
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
            loss_type: LossFunctionType::DirectMse,
            target_type: ValueType::Float,
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

    /// 3. Feszültség-alakváltozás kapcsolat keresése
    /// 
    /// data_x: F vagy E (Mat3), data_y: sigma vagy P (Mat3)
    /// target: pl. f(F)=sigma

    // gumi, szilikon
    pub fn solid_incompressible_isotropic() -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
        config.loss_type = LossFunctionType::TensorMseMat3;// TODO
        config.target_type = ValueType::Mat3;
        config.disabled_constant_types = vec![
            ValueType::Vec2, ValueType::Vec3, ValueType::Mat2, ValueType::Mat3, 
        ];
        config.base_parsimony_penalty = 0.00001;
        config.max_tree_size = 64;
        config.mutation_max_depth = 6;
        config.island_size = 1000;
        config.subset_size = Some(800);
        config.mini_batch_size = 128;
        config.stagnation_threshold = 200;
        config.tournament_size = 3;
        config.opt_iterations = 250;
        config.opt_prob = 0.05;
        config.num_islands = 16;
        config.base_target_mse = 0.00005;

       let exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF), Instruction::Basic(BasicOpCode::CosF),
            Instruction::Linalg(LinalgOpCode::DetM3), Instruction::Linalg(LinalgOpCode::DetM2),
            Instruction::Solid(SolidOpCode::InvariantJ3M3),
            //Instruction::Solid(SolidOpCode::InvariantI4_M3),
            //Instruction::Solid(SolidOpCode::InvariantI6_M3),
            //Instruction::Solid(SolidOpCode::InvariantI5_M3),
            //Instruction::Solid(SolidOpCode::InvariantI7_M3),
        ];
        config.excluded_ops = Self::add_vector_exclusions(exclusions);
        config
    }

    // bilógiai szövetek
    pub fn solid_incompressible_anisotropic() -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
        config.loss_type = LossFunctionType::PlanarBiaxialMse;
        config.target_type = ValueType::Mat3;
        config.disabled_constant_types = vec![
            ValueType::Vec2, ValueType::Vec3, ValueType::Mat2, ValueType::Mat3, 
        ];
        config.base_parsimony_penalty = 0.00001;
        config.max_tree_size = 64;
        config.mutation_max_depth = 6;
        config.island_size = 1000;
        config.subset_size = Some(800);
        config.mini_batch_size = 128;
        config.stagnation_threshold = 200;
        config.tournament_size = 3;
        config.opt_iterations = 250;
        config.opt_prob = 0.05;
        config.num_islands = 16;
        config.base_target_mse = 0.00005;

        let exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),

            Instruction::Linalg(LinalgOpCode::DetM3),
            Instruction::Linalg(LinalgOpCode::DetM2),
            Instruction::Solid(SolidOpCode::InvariantJ3M3),
        ];
        
        config.excluded_ops = Self::add_vector_exclusions(exclusions);
        config
    }

    // fémek
    pub fn solid_elastoplastic_metals() -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
        config.loss_type = LossFunctionType::TensorMseMat3;//todo
        config.target_type = ValueType::Mat3;
        config.disabled_constant_types = vec![ValueType::Vec2, ValueType::Vec3, ValueType::Mat2, ValueType::Mat3];
        
        config.base_parsimony_penalty = 0.0001; 
        config.max_tree_size = 48;
        config.island_size = 800;
        config.stagnation_threshold = 800;
        config.opt_iterations = 200;

        let exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
            Instruction::Basic(BasicOpCode::ExpF),
            Instruction::Basic(BasicOpCode::LnF),
            Instruction::Solid(SolidOpCode::IsochoricInvariant1),
            Instruction::Solid(SolidOpCode::IsochoricInvariant2),
            //Instruction::Solid(SolidOpCode::InvariantI4_M3),
            //Instruction::Solid(SolidOpCode::InvariantI6_M3),
            Instruction::Solid(SolidOpCode::CofactorM3),
        ];

        config.excluded_ops = Self::add_vector_exclusions(exclusions);
        config
    }

    // habok, szivacsok
    pub fn solid_compressible_isotropic() -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
        config.loss_type = LossFunctionType::TensorMseMat3;//todo
        config.target_type = ValueType::Mat3;
        config.disabled_constant_types = vec![ValueType::Vec2, ValueType::Vec3, ValueType::Mat2, ValueType::Mat3];
        
        config.base_parsimony_penalty = 0.00005;
        config.max_tree_size = 56;
        config.mutation_max_depth = 5;
        config.island_size = 1000;
        config.stagnation_threshold = 400;
        config.opt_iterations = 250;
        config.opt_prob = 0.05;
        
        let exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
            //Instruction::Solid(SolidOpCode::InvariantI4_M3),
            //Instruction::Solid(SolidOpCode::InvariantI6_M3),
            //Instruction::Solid(SolidOpCode::InvariantI5_M3),
            //Instruction::Solid(SolidOpCode::InvariantI7_M3),
        ];

        config.excluded_ops = Self::add_vector_exclusions(exclusions);
        config
    }

    // 3d nyomtatott rácsok, metamateriálok
    pub fn solid_compressible_anisotropic() -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
        config.loss_type = LossFunctionType::TensorMseMat3;// TODO
        config.target_type = ValueType::Mat3;
        config.disabled_constant_types = vec![ValueType::Vec2, ValueType::Vec3, ValueType::Mat2, ValueType::Mat3];
        
        config.base_parsimony_penalty = 0.00005;
        config.max_tree_size = 64;
        config.mutation_max_depth = 6;
        config.island_size = 1000;
        config.stagnation_threshold = 400;
        config.opt_iterations = 250;
        config.opt_prob = 0.05;
        
        let exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),            
            //Instruction::Solid(SolidOpCode::InvariantI5_M3),
            //Instruction::Solid(SolidOpCode::InvariantI7_M3),
        ];

        config.excluded_ops = Self::add_vector_exclusions(exclusions);
        config
    }
    
    fn add_vector_exclusions(mut exclusions: Vec<Instruction>) -> Vec<Instruction> {
            let vector_ops = [
                LinalgOpCode::MakeVec2, LinalgOpCode::MakeVec3, LinalgOpCode::GetXV2, LinalgOpCode::GetYV2,
                LinalgOpCode::GetXV3, LinalgOpCode::GetYV3, LinalgOpCode::GetZV3, LinalgOpCode::AddV2, 
                LinalgOpCode::SubV2, LinalgOpCode::ScaleV2, LinalgOpCode::DotV2, LinalgOpCode::NormV2,
                LinalgOpCode::AddV3, LinalgOpCode::SubV3, LinalgOpCode::ScaleV3, LinalgOpCode::DotV3, 
                LinalgOpCode::NormV3, LinalgOpCode::CrossV3, LinalgOpCode::MulM2V2, LinalgOpCode::MulM3V3,
            ];
            for op in vector_ops { exclusions.push(Instruction::Linalg(op)); }
            exclusions
        }
}

