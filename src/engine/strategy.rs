use crate::engine::config::{EvolutionConfig, OpModule};
use crate::domain::universal::{UniversalOp};

pub trait Strategy: Clone + Send + Sync {
    // Architektúrális paraméterek
    fn num_islands(&self) -> usize;
    fn island_size(&self) -> usize;
    
    // Evolúciós paraméterek
    fn max_generations(&self) -> usize;
    fn target_mse(&self) -> f32;
    fn crossover_rate(&self) -> f32;
    fn tournament_size(&self) -> usize;
    fn migration_interval(&self) -> usize;
    fn parsimony_penalty(&self) -> f32;
    
    // Mutációs és generálási paraméterek
    fn random_injection_rate(&self) -> f32;
    fn min_random_injection(&self) -> usize;
    fn max_tree_size(&self) -> usize;
    fn mutation_max_depth(&self) -> usize;
    fn mutation_cycles(&self) -> usize;
    
    // Optimalizációs paraméterek
    fn opt_prob(&self) -> f32;
    fn opt_iterations(&self) -> usize;
    fn final_opt_iterations(&self) -> usize;
    
    // Stagnálás paraméterek
    fn stagnation_threshold(&self) -> usize;
    fn min_improvement(&self) -> f32;
    
    // Egyéb
    fn verbose(&self) -> bool;
    fn get_allowed_operators(&self) -> Vec<UniversalOp>;

    // --- ESEMÉNYEK (Hooks) a későbbi dinamikus tanuláshoz ---
    fn on_generation_end(&mut self, _best_mse: f32, _stagnation_counter: usize) {}
    fn on_nuke(&mut self) {}
}

#[derive(Clone)]
pub struct StaticStrategy {
    pub config: EvolutionConfig,
}

impl StaticStrategy {
    pub fn new(config: EvolutionConfig) -> Self {
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

    fn get_allowed_operators(&self) -> Vec<UniversalOp> {
        let mut ops = Vec::new();
        for module in &self.config.allowed_modules {
            match module {
                OpModule::Basic => ops.extend_from_slice(&[
                    UniversalOp::AddF, UniversalOp::SubF, UniversalOp::MulF, UniversalOp::DivF, 
                    UniversalOp::SinF, UniversalOp::CosF, UniversalOp::ExpF, UniversalOp::SqrF,
                    UniversalOp::LnF, UniversalOp::SqrtF
                ]),
                OpModule::Linalg => ops.extend_from_slice(&[
                    UniversalOp::AddV3, UniversalOp::DotV3, UniversalOp::ScaleV3
                ]),
                OpModule::Logic => ops.extend_from_slice(&[
                    UniversalOp::IfElseF
                ]),
            }
        }
        ops
    }
}