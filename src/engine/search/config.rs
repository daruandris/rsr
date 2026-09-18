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
    pub extract_scalars: bool
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
            extract_scalars: true,
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

    /// **Material Properties:**
    /// Hyperelastic, incompressible (volume remains constant, $J = \det(F) = 1$), and isotropic (mechanical properties are identical in all directions).
    /// 
    /// **Typical Examples:**
    /// Rubber, silicone, elastomers.
    /// 
    /// **Mapping Type:**
    /// Typically $F \rightarrow P$ (Deformation Gradient to 1st Piola-Kirchhoff stress), though $F \rightarrow \sigma$ can also be used. If the data_y is Cauchy-stress ($\sigma$), then set the target_is_cauchy parameter to true.
    /// 
    /// **Required Test Data:**
    /// Uniaxial tension/compression, equibiaxial tension, or pure shear tests. 
    /// **CRITICAL:** The experimental dataset MUST represent a strictly **homogeneous stress state**. This means data cannot be extracted from a random, complex geometry (like an engine block); it must come from standardized specimens where the measured zone undergoes uniform deformation.
    /// 
    /// **Excluded Operators:**
    /// * `DetM3`, `DetM2`, `InvariantJ3M3`: Excluded because the material is incompressible, meaning the determinant of the deformation gradient is always 1, making these operations redundant[cite: 1, 2].
    /// * `InvariantI4`, `InvariantI5`, `InvariantI6`, `InvariantI7`: Excluded because these are pseudo-invariants used to describe fiber directions in anisotropic materials. Since this material is isotropic, these are not needed.
    /// *(Note: Vector operations and basic trigonometric functions like Sin/Cos are excluded globally to reduce the search space).*
    /// 
    /// **Evaluation:**
    /// // TODO: `LossFunctionType::TensorMseMat3` is currently a placeholder here. 
    /// // A specific loss function tailored to isotropic incompressible datasets (e.g., `UniaxialMse` or `EquibiaxialMse`) must be implemented and set here instead.
    pub fn solid_incompressible_isotropic(target_is_cauchy: bool) -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
        config.loss_type = LossFunctionType::TensorMseMat3;// TODO
        config.target_type = ValueType::Mat3;
        config.disabled_constant_types = vec![
            ValueType::Vec2, ValueType::Vec3, ValueType::Mat2, ValueType::Mat3, 
        ];
        config.base_parsimony_penalty = 0.000005;
        config.max_tree_size = 64; //TODO to make it bigger than 32
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
        config.extract_scalars = false;

       let mut exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF), Instruction::Basic(BasicOpCode::CosF), Instruction::Basic(BasicOpCode::DivF),
            Instruction::Linalg(LinalgOpCode::DetM3), Instruction::Linalg(LinalgOpCode::DetM2),
            Instruction::Solid(SolidOpCode::InvariantJ3M3),
            Instruction::Solid(SolidOpCode::InvariantI4),
            Instruction::Solid(SolidOpCode::InvariantI6),
            Instruction::Solid(SolidOpCode::InvariantI5),
            Instruction::Solid(SolidOpCode::InvariantI7),
            Instruction::Solid(SolidOpCode::DispersedI4),
            Instruction::Solid(SolidOpCode::DispersedI5),
        ];

        if !target_is_cauchy {
            exclusions.push(Instruction::Solid(SolidOpCode::DeviatoricM3));
        }
        config.excluded_ops = Self::add_vector_exclusions(exclusions);
        config
    }

    /// Configuration for Incompressible Anisotropic Materials
    /// 
    /// **Material Properties:**
    /// Hyperelastic, practically incompressible (isochoric, $J = \det(F) = 1$), and mechanically anisotropic (highly direction-dependent due to internal fiber structures).
    /// 
    /// **Typical Examples:**
    /// Biological soft tissues, such as the aorta, skin, or muscle tissue.
    /// 
    /// **Mapping Type:**
    /// $F \rightarrow P$ (Deformation Gradient to 1st Piola-Kirchhoff stress).
    /// 
    /// **Required Test Data:**
    /// Planar biaxial experimental datasets (e.g., stretching a cross-shaped tissue sample in two directions while measuring forces and stretches). 
    /// **CRITICAL:** As with all solid configs, the dataset must capture a pure **homogeneous stress state** from a precisely controlled lab environment. Search terms like "Biaxial tensile test raw data soft tissue" are recommended.
    /// 
    /// **Excluded Operators:**
    /// * `DetM3`, `DetM2`, `InvariantJ3M3`: Excluded because the material is volume-preserving (incompressible), making volumetric variables constant[cite: 1, 2].
    /// *(Note: Unlike the isotropic config, invariant operators like I4 and I6 are KEPT here because they are essential for describing anisotropic fiber directions).*
    /// 
    /// **Evaluation:**
    /// Evaluated using `LossFunctionType::PlanarBiaxialMse`, which is specifically implemented to handle multi-axial force and displacement data correctly.
    pub fn solid_incompressible_anisotropic() -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
        config.loss_type = LossFunctionType::PlanarBiaxialMse;
        config.target_type = ValueType::Mat3;
        config.disabled_constant_types = vec![
             ValueType::Mat2, ValueType::Mat3,  
        ];
        config.base_parsimony_penalty = 0.00005;
        config.max_tree_size = 128; //TODO to make it bigger than 32
        config.mutation_max_depth = 6;
        config.island_size = 1000;
        config.subset_size = Some(800);
        config.mini_batch_size = 128;
        config.stagnation_threshold = 200;
        config.tournament_size = 3;
        config.opt_iterations = 250;
        config.opt_prob = 0.05;
        config.num_islands = 16;
        config.base_target_mse = 1e-7;
        config.extract_scalars = false;

        let exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
            Instruction::Basic(BasicOpCode::DivF),

            Instruction::Linalg(LinalgOpCode::DetM3),
            Instruction::Linalg(LinalgOpCode::DetM2),
            Instruction::Solid(SolidOpCode::InvariantJ3M3),
            Instruction::Solid(SolidOpCode::DeviatoricM3),
            Instruction::Solid(SolidOpCode::DispersedI4),
            Instruction::Solid(SolidOpCode::DispersedI5),

        ];
        
        config.excluded_ops = Self::add_vector_exclusions(exclusions);
        config
    }

    /// Configuration for Incompressible Anisotropic Dispersive Materials
    /// 
    /// **Material Properties:**
    /// Biological soft tissues with strong fiber dispersion (e.g., skin, heart myocardium).
    /// Fibers are not perfectly aligned, but spread spherically/elliptically around a mean direction.
    pub fn solid_incompressible_anisotropic_dispersive() -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
        config.loss_type = LossFunctionType::TensorMseMat3;
        config.target_type = ValueType::Mat3;
        config.disabled_constant_types = vec![
             ValueType::Mat2, ValueType::Mat3,  
        ];
        config.base_parsimony_penalty = 0.00005;
        config.max_tree_size = 128;
        config.mutation_max_depth = 6;
        config.island_size = 1000;
        config.subset_size = Some(800);
        config.mini_batch_size = 128;
        config.stagnation_threshold = 200;
        config.tournament_size = 3;
        config.opt_iterations = 250;
        config.opt_prob = 0.05;
        config.num_islands = 16;
        config.base_target_mse = 1e-7;
        config.extract_scalars = false;

        let exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
            Instruction::Basic(BasicOpCode::DivF),

            Instruction::Linalg(LinalgOpCode::DetM3),
            Instruction::Linalg(LinalgOpCode::DetM2),
            Instruction::Solid(SolidOpCode::InvariantJ3M3),
            Instruction::Solid(SolidOpCode::DeviatoricM3),
            Instruction::Solid(SolidOpCode::InvariantI4),
            Instruction::Solid(SolidOpCode::InvariantI5),
            Instruction::Solid(SolidOpCode::InvariantI6),
            Instruction::Solid(SolidOpCode::InvariantI7),
        ];
        
        config.excluded_ops = Self::add_vector_exclusions(exclusions);
        config
    }

    /// Configuration for Compressible Isotropic Materials
    /// 
    /// **Material Properties:**
    /// Hyperelastic/Elastic, strongly compressible (volume changes drastically under load, $J \neq 1$), and isotropic (uniform properties in all directions).
    /// 
    /// **Typical Examples:**
    /// Foams, sponges, and porous polymers.
    /// 
    /// **Mapping Type:**
    /// $F \rightarrow P$ or $F \rightarrow \sigma$ (Deformation Gradient to Stress).
    /// 
    /// **Required Test Data:**
    /// Volumetric/hydrostatic pressure tests, or uniaxial compression tests. 
    /// **CRITICAL:** Standardized testing creating a **homogeneous stress state** is mandatory. In these tests, both vertical compression and lateral expansion MUST be measured simultaneously, because the material's volume changes.
    /// 
    /// **Excluded Operators:**
    /// * `InvariantI4`, `InvariantI5`, `InvariantI6`, `InvariantI7`: Excluded because these are directional pseudo-invariants used exclusively for fiber-reinforced or anisotropic materials. They have no physical meaning in isotropic foams.
    /// 
    /// **Evaluation:**
    /// // TODO: `LossFunctionType::TensorMseMat3` is currently a placeholder. 
    /// // A specific loss function designed for compressible data (e.g., `HydrostaticPressureMse` or `CompressibleUniaxialMse`) must be implemented and set here instead.
    pub fn solid_compressible_isotropic() -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
        config.loss_type = LossFunctionType::TensorMseMat3;//todo
        config.target_type = ValueType::Mat3;
        config.disabled_constant_types = vec![ValueType::Vec2, ValueType::Vec3, ValueType::Mat2, ValueType::Mat3];
        
        config.base_parsimony_penalty = 0.00005;
         //config.max_tree_size = 64; TODO to make it bigger than 32
        config.mutation_max_depth = 5;
        config.island_size = 1000;
        config.stagnation_threshold = 400;
        config.opt_iterations = 250;
        config.opt_prob = 0.05;
        config.extract_scalars = false;
        
        let exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
            Instruction::Basic(BasicOpCode::DivF),
            Instruction::Solid(SolidOpCode::InvariantI4),
            Instruction::Solid(SolidOpCode::InvariantI6),
            Instruction::Solid(SolidOpCode::InvariantI5),
            Instruction::Solid(SolidOpCode::InvariantI7),
            Instruction::Solid(SolidOpCode::DispersedI4),
            Instruction::Solid(SolidOpCode::DispersedI5),
        ];

        config.excluded_ops = Self::add_vector_exclusions(exclusions);
        config
    }

    // Configuration for Compressible Anisotropic Materials
    /// 
    /// **Material Properties:**
    /// Elastic/Hyperelastic, compressible (volume is not constant due to internal voids/air), and anisotropic (geometry heavily dictates direction-dependent stiffness).
    /// 
    /// **Typical Examples:**
    /// Mechanical metamaterials, 3D printed lattices, cellular solids.
    /// 
    /// **Mapping Type:**
    /// $F \rightarrow P$ (Deformation Gradient to macroscopic "homogenized" 1st Piola-Kirchhoff stress).
    /// 
    /// **Required Test Data:**
    /// Uniaxial compression tests on cellular solids (e.g., pressing a precise metamaterial cube). 
    /// **CRITICAL:** The test must guarantee a macroscopic **homogeneous stress state** across the lattice structure. Since the lattice is compressible, lateral expansion/buckling cannot be mathematically derived from vertical compression; it MUST be optically measured alongside the load.
    /// 
    /// **Excluded Operators:**
    /// * `InvariantI5`, `InvariantI7`: Excluded to constrain the search space. While the material is anisotropic, keeping a reduced set of directional invariants (like I4 and I6) is usually sufficient to model lattice symmetries without overwhelming the genetic algorithm.
    /// 
    /// **Evaluation:**
    /// // TODO: `LossFunctionType::TensorMseMat3` is currently a placeholder. 
    /// // A specific loss function handling homogenized macroscopic stresses for cellular solids (e.g., `LatticeCompressionMse`) must be implemented and set here instead.
    pub fn solid_compressible_anisotropic() -> Self {
        let mut config = Self::default(vec![OpModule::Basic, OpModule::Linalg, OpModule::Solid]);
        config.loss_type = LossFunctionType::TensorMseMat3;// TODO
        config.target_type = ValueType::Mat3;
        config.disabled_constant_types = vec![ ValueType::Mat2, ValueType::Mat3];
        
        config.base_parsimony_penalty = 0.00005;
        //config.max_tree_size = 64; TODO to make it bigger than 32
        config.mutation_max_depth = 6;
        config.island_size = 1000;
        config.stagnation_threshold = 400;
        config.opt_iterations = 250;
        config.opt_prob = 0.05;
        config.extract_scalars = false;
        
        let exclusions = vec![
            Instruction::Basic(BasicOpCode::SinF),
            Instruction::Basic(BasicOpCode::CosF),
            Instruction::Basic(BasicOpCode::DivF),          
            Instruction::Solid(SolidOpCode::InvariantI5),
            Instruction::Solid(SolidOpCode::InvariantI7),
            Instruction::Solid(SolidOpCode::DeviatoricM3),
            Instruction::Solid(SolidOpCode::DispersedI4),
            Instruction::Solid(SolidOpCode::DispersedI5),
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
                LinalgOpCode::MakeMat2, LinalgOpCode::MakeMat3,
            ];
            for op in vector_ops { exclusions.push(Instruction::Linalg(op)); }
            exclusions        
    }
}

