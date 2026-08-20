use crate::engine::data::dataset::Dataset;
use crate::engine::search::config::{Config, OpModule};
use crate::engine::search::engine::Engine;
use crate::engine::search::strategy::{StaticStrategy, Strategy};

pub struct SymbolicRegressor {
    pub config: Config,
}

pub struct FitResult {
    pub equation: String,
    pub mse: f32,
    pub complexity: usize,
}

impl SymbolicRegressor {
    pub fn new(allowed_modules: Vec<OpModule>) -> Self {
        let config = Config::default(allowed_modules);
        Self { config }
    }

    pub fn target_mse(mut self, target: f32) -> Self {
        self.config.target_mse = target;
        self
    }

    pub fn generations(mut self, gens: usize) -> Self {
        self.config.max_generations = gens;
        self
    }

    pub fn with_module(mut self, module: OpModule) -> Self {
        if !self.config.allowed_modules.contains(&module) {
            self.config.allowed_modules.push(module);
        }
        self
    }

    pub fn fit(&self, dataset: &Dataset) -> FitResult {
        let strategy = StaticStrategy::new(self.config.clone());
        let allowed_ops = strategy.get_allowed_operators();

        let mut engine = Engine::new(strategy, dataset.get_variable_registry(), allowed_ops);
        engine.run(dataset);

        let best = engine.get_global_best();
        let clean_eq = crate::engine::ffi::symengine::simplify_symengine(&best.to_string());

        FitResult {
            equation: clean_eq,
            mse: best.fitness,
            complexity: best.complexity(),
        }
    }
}
