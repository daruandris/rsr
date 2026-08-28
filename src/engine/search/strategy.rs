use super::config::{Config, OpModule};
use crate::Instruction;
use crate::domains::basic::BasicOpCode;
use crate::domains::linalg::LinalgOpCode;
use crate::domains::solid::SolidOpCode;

pub trait Strategy: Clone + Send + Sync {
    fn num_islands(&self) -> usize;
    fn island_size(&self) -> usize;
    fn max_generations(&self) -> usize;
    fn target_mse(&self) -> f32;
    fn crossover_rate(&self) -> f32;
    fn tournament_size(&self) -> usize;
    fn migration_interval(&self) -> usize;
    fn parsimony_penalty(&self) -> f32;
    fn random_injection_rate(&self) -> f32;
    fn min_random_injection(&self) -> usize;
    fn max_tree_size(&self) -> usize;
    fn mutation_max_depth(&self) -> usize;
    fn mutation_cycles(&self) -> usize;
    fn opt_prob(&self) -> f32;
    fn opt_iterations(&self) -> usize;
    fn final_opt_iterations(&self) -> usize;
    fn stagnation_threshold(&self) -> usize;
    fn min_improvement(&self) -> f32;
    fn verbose(&self) -> bool;
    fn mini_batch_size(&self) -> usize;

    fn get_allowed_operators(&self) -> Vec<Instruction>;
    fn on_generation_end(&mut self, _best_mse: f32, _stagnation_counter: usize) {}
    fn on_nuke(&mut self) {}
}

#[derive(Clone)]
pub struct StaticStrategy {
    pub config: Config,
}

impl StaticStrategy {
    pub fn new(config: Config) -> Self {
        Self { config }
    }
}

impl Strategy for StaticStrategy {
    #[inline(always)]
    fn num_islands(&self) -> usize {
        self.config.num_islands
    }
    #[inline(always)]
    fn island_size(&self) -> usize {
        self.config.island_size
    }
    #[inline(always)]
    fn max_generations(&self) -> usize {
        self.config.max_generations
    }
    #[inline(always)]
    fn target_mse(&self) -> f32 {
        self.config.target_mse
    }
    #[inline(always)]
    fn crossover_rate(&self) -> f32 {
        self.config.crossover_rate
    }
    #[inline(always)]
    fn tournament_size(&self) -> usize {
        self.config.tournament_size
    }
    #[inline(always)]
    fn migration_interval(&self) -> usize {
        self.config.migration_interval
    }
    #[inline(always)]
    fn parsimony_penalty(&self) -> f32 {
        self.config.parsimony_penalty
    }
    #[inline(always)]
    fn random_injection_rate(&self) -> f32 {
        self.config.random_injection_rate
    }
    #[inline(always)]
    fn min_random_injection(&self) -> usize {
        self.config.min_random_injection
    }
    #[inline(always)]
    fn max_tree_size(&self) -> usize {
        self.config.max_tree_size
    }
    #[inline(always)]
    fn mutation_max_depth(&self) -> usize {
        self.config.mutation_max_depth
    }
    #[inline(always)]
    fn mutation_cycles(&self) -> usize {
        self.config.mutation_cycles
    }
    #[inline(always)]
    fn opt_prob(&self) -> f32 {
        self.config.opt_prob
    }
    #[inline(always)]
    fn opt_iterations(&self) -> usize {
        self.config.opt_iterations
    }
    #[inline(always)]
    fn final_opt_iterations(&self) -> usize {
        self.config.final_opt_iterations
    }
    #[inline(always)]
    fn stagnation_threshold(&self) -> usize {
        self.config.stagnation_threshold
    }
    #[inline(always)]
    fn min_improvement(&self) -> f32 {
        self.config.min_improvement
    }
    #[inline(always)]
    fn verbose(&self) -> bool {
        self.config.verbose
    }

    #[inline(always)]
    fn mini_batch_size(&self) -> usize {
        self.config.mini_batch_size
    }

    fn get_allowed_operators(&self) -> Vec<Instruction> {
        let mut ops = Vec::new();
        for module in &self.config.allowed_modules {
            match module {
                OpModule::Basic => ops.extend_from_slice(&[
                    Instruction::Basic(BasicOpCode::AddF),
                    Instruction::Basic(BasicOpCode::SubF),
                    Instruction::Basic(BasicOpCode::MulF),
                    Instruction::Basic(BasicOpCode::DivF),
                    Instruction::Basic(BasicOpCode::SinF),
                    Instruction::Basic(BasicOpCode::CosF),
                    Instruction::Basic(BasicOpCode::ExpF),
                    Instruction::Basic(BasicOpCode::SqrF),
                    Instruction::Basic(BasicOpCode::LnF),
                    Instruction::Basic(BasicOpCode::SqrtF),
                ]),
                OpModule::Linalg => ops.extend_from_slice(&[
                    Instruction::Linalg(LinalgOpCode::MakeVec2),
                    Instruction::Linalg(LinalgOpCode::MakeVec3),
                    Instruction::Linalg(LinalgOpCode::GetXV2),
                    Instruction::Linalg(LinalgOpCode::GetYV2),
                    Instruction::Linalg(LinalgOpCode::GetXV3),
                    Instruction::Linalg(LinalgOpCode::GetYV3),
                    Instruction::Linalg(LinalgOpCode::GetZV3),
                    Instruction::Linalg(LinalgOpCode::AddV2),
                    Instruction::Linalg(LinalgOpCode::SubV2),
                    Instruction::Linalg(LinalgOpCode::ScaleV2),
                    Instruction::Linalg(LinalgOpCode::DotV2),
                    Instruction::Linalg(LinalgOpCode::NormV2),
                    Instruction::Linalg(LinalgOpCode::AddV3),
                    Instruction::Linalg(LinalgOpCode::SubV3),
                    Instruction::Linalg(LinalgOpCode::ScaleV3),
                    Instruction::Linalg(LinalgOpCode::DotV3),
                    Instruction::Linalg(LinalgOpCode::NormV3),
                    Instruction::Linalg(LinalgOpCode::CrossV3),
                    Instruction::Linalg(LinalgOpCode::MakeMat2),
                    Instruction::Linalg(LinalgOpCode::AddM2),
                    Instruction::Linalg(LinalgOpCode::SubM2),
                    Instruction::Linalg(LinalgOpCode::ScaleM2),
                    Instruction::Linalg(LinalgOpCode::MulM2),
                    Instruction::Linalg(LinalgOpCode::MulM2V2),
                    Instruction::Linalg(LinalgOpCode::DetM2),
                    Instruction::Linalg(LinalgOpCode::TraceM2),
                    Instruction::Linalg(LinalgOpCode::TransposeM2),
                    Instruction::Linalg(LinalgOpCode::InverseM2),
                    Instruction::Linalg(LinalgOpCode::MakeMat3),
                    Instruction::Linalg(LinalgOpCode::AddM3),
                    Instruction::Linalg(LinalgOpCode::SubM3),
                    Instruction::Linalg(LinalgOpCode::ScaleM3),
                    Instruction::Linalg(LinalgOpCode::MulM3),
                    Instruction::Linalg(LinalgOpCode::MulM3V3),
                    Instruction::Linalg(LinalgOpCode::DetM3),
                    Instruction::Linalg(LinalgOpCode::TraceM3),
                    Instruction::Linalg(LinalgOpCode::TransposeM3),
                    Instruction::Linalg(LinalgOpCode::InverseM3),
                ]),
                OpModule::Logic => { /* Boolean domain will go here later */ }
                OpModule::Solid => ops.extend_from_slice(&[
                    Instruction::Solid(SolidOpCode::RightCauchyGreenM3),
                    Instruction::Solid(SolidOpCode::LeftCauchyGreenM3),
                    Instruction::Solid(SolidOpCode::Invariant2M3),
                    Instruction::Solid(SolidOpCode::CofactorM3),
                    Instruction::Solid(SolidOpCode::GreenLagrangeStrainM3),
                    Instruction::Solid(SolidOpCode::IsochoricInvariant1),
                    Instruction::Solid(SolidOpCode::IsochoricInvariant2),
                ]),
            }
        }

        ops.extend(&self.config.custom_ops);
        let mut final_ops = Vec::new();
        for op in ops {
            if !self.config.excluded_ops.contains(&op) && !final_ops.contains(&op) {
                final_ops.push(op);
            }
        }
        final_ops
    }
}
