use crate::Instruction;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpModule {
    Basic,
    Linalg,
    Logic,
}

#[derive(Clone, Debug)]
pub struct Config {
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
    pub max_tree_size: usize,
    pub mutation_max_depth: usize,
    pub mutation_cycles: usize,
    pub verbose: bool,

    pub allowed_modules: Vec<OpModule>,
    pub custom_ops: Vec<Instruction>,
    pub excluded_ops: Vec<Instruction>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_islands: 4,
            island_size: 100,
            max_generations: 1000,
            crossover_rate: 0.7,
            tournament_size: 3,
            migration_interval: 50,
            parsimony_penalty: 0.001,
            opt_prob: 0.2,
            opt_iterations: 15,
            final_opt_iterations: 100,
            stagnation_threshold: 20,
            target_mse: 1e-5,
            min_improvement: 1e-6,
            random_injection_rate: 0.1,
            min_random_injection: 5,
            max_tree_size: 50,
            mutation_max_depth: 4,
            mutation_cycles: 1,
            verbose: true,
            allowed_modules: vec![OpModule::Basic],
            custom_ops: vec![],
            excluded_ops: vec![],
        }
    }
}

impl Config {
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
}