use crate::eval::op::Op;
use super::config::{Config, OpModule};

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
    
    fn get_allowed_operators(&self) -> Vec<Op>;
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
    #[inline(always)] fn num_islands(&self) -> usize { self.config.num_islands }
    #[inline(always)] fn island_size(&self) -> usize { self.config.island_size }
    #[inline(always)] fn max_generations(&self) -> usize { self.config.max_generations }
    #[inline(always)] fn target_mse(&self) -> f32 { self.config.target_mse }
    #[inline(always)] fn crossover_rate(&self) -> f32 { self.config.crossover_rate }
    #[inline(always)] fn tournament_size(&self) -> usize { self.config.tournament_size }
    #[inline(always)] fn migration_interval(&self) -> usize { self.config.migration_interval }
    #[inline(always)] fn parsimony_penalty(&self) -> f32 { self.config.parsimony_penalty }
    #[inline(always)] fn random_injection_rate(&self) -> f32 { self.config.random_injection_rate }
    #[inline(always)] fn min_random_injection(&self) -> usize { self.config.min_random_injection }
    #[inline(always)] fn max_tree_size(&self) -> usize { self.config.max_tree_size }
    #[inline(always)] fn mutation_max_depth(&self) -> usize { self.config.mutation_max_depth }
    #[inline(always)] fn mutation_cycles(&self) -> usize { self.config.mutation_cycles }
    #[inline(always)] fn opt_prob(&self) -> f32 { self.config.opt_prob }
    #[inline(always)] fn opt_iterations(&self) -> usize { self.config.opt_iterations }
    #[inline(always)] fn final_opt_iterations(&self) -> usize { self.config.final_opt_iterations }
    #[inline(always)] fn stagnation_threshold(&self) -> usize { self.config.stagnation_threshold }
    #[inline(always)] fn min_improvement(&self) -> f32 { self.config.min_improvement }
    #[inline(always)] fn verbose(&self) -> bool { self.config.verbose }

    fn get_allowed_operators(&self) -> Vec<Op> {
        let mut ops = Vec::new();
        for module in &self.config.allowed_modules {
            match module {
                OpModule::Basic => ops.extend_from_slice(&[
                    Op::AddF, Op::SubF, Op::MulF, Op::DivF, 
                    Op::SinF, Op::CosF, Op::ExpF, Op::SqrF, Op::LnF, Op::SqrtF
                ]),
                OpModule::Linalg => ops.extend_from_slice(&[
                    Op::MakeVec2, Op::MakeVec3, Op::GetXV2, Op::GetYV2,
                    Op::GetXV3, Op::GetYV3, Op::GetZV3,
                    Op::AddV2, Op::SubV2, Op::ScaleV2, Op::DotV2, Op::NormV2,
                    Op::AddV3, Op::SubV3, Op::ScaleV3, Op::DotV3, Op::NormV3, Op::CrossV3,
                    Op::MakeMat2, Op::AddM2, Op::SubM2, Op::ScaleM2, Op::MulM2, Op::MulM2V2, 
                    Op::DetM2, Op::TraceM2, Op::TransposeM2, Op::InverseM2,
                    Op::MakeMat3, Op::AddM3, Op::SubM3, Op::ScaleM3, Op::MulM3, Op::MulM3V3, 
                    Op::DetM3, Op::TraceM3, Op::TransposeM3, Op::InverseM3,
                ]),
                OpModule::Logic => ops.extend_from_slice(&[Op::IfElseF]),
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