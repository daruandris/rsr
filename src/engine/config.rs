use crate::domain::universal::UniversalOp;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpModule { Basic, Linalg, Logic }

#[derive(Clone, Debug)]
pub struct EvolutionConfig {
    pub num_islands: usize,
    pub island_size: usize,
    pub max_generations: usize,
    pub crossover_rate: f32,
    pub tournament_size: usize,
    pub migration_interval: usize,
    pub parsimony_penalty: f32,
    pub opt_prob: f32,
    pub opt_iterations: usize,
    pub final_opt_iterations: usize,
    pub stagnation_threshold: usize,
    pub target_mse: f32,
    pub min_improvement: f32,
    pub random_injection_rate: f32,
    pub min_random_injection: usize,
    pub max_tree_size : usize,
    pub mutation_max_depth: usize,
    pub mutation_cycles: usize,
    pub verbose: bool,

    pub allowed_modules: Vec<OpModule>,
    pub custom_ops: Vec<UniversalOp>,
    pub excluded_ops: Vec<UniversalOp>,
}

impl EvolutionConfig {
    pub fn with_module(mut self, module: OpModule) -> Self {
        if !self.allowed_modules.contains(&module) { self.allowed_modules.push(module); }
        self
    }

    pub fn with_op(mut self, op: UniversalOp) -> Self {
        if !self.custom_ops.contains(&op) { self.custom_ops.push(op); }
        self
    }

    pub fn without_op(mut self, op: UniversalOp) -> Self {
        if !self.excluded_ops.contains(&op) { self.excluded_ops.push(op); }
        self
    }
}