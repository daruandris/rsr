//! High-level interface for symbolic regression.
//!
//! This module provides the user-facing [`SymbolicRegressor`] struct, which wraps
//! the underlying genetic engine and configuration into an easy-to-use API.

use crate::engine::data::dataset::Dataset;
use crate::engine::search::config::{Config, OpModule};
use crate::engine::search::engine::Engine;
use crate::engine::search::strategy::{StaticStrategy, Strategy};

/// The primary interface for performing symbolic regression.
///
/// `SymbolicRegressor` uses a builder pattern for configuration. You can chain
/// methods like [`target_mse`](Self::target_mse) and [`generations`](Self::generations)
/// to customize the search parameters before calling [`fit`](Self::fit).
///
/// # Examples
///
/// ```
/// use rsr::prelude::*;
/// use rsr::api::SymbolicRegressor;
///
/// let regressor = SymbolicRegressor::default_with_modules(vec![OpModule::Basic])
///     .target_mse(1e-6)
///     .generations(500)
///     .with_module(OpModule::Linalg);
/// ```
pub struct SymbolicRegressor {
    /// The underlying configuration for the genetic algorithm and numerical optimizers.
    pub config: Config,
}

/// The outcome of a successful symbolic regression fit.
pub struct FitResult {
    /// The final simplified mathematical equation as a string (e.g., `"2.5 * x + sin(y)"`).
    pub equation: String,
    /// The Mean Squared Error of the best equation found.
    pub mse: f32,
    /// The complexity score of the equation (sum of the weights of its nodes).
    pub complexity: usize,
}

impl SymbolicRegressor {
    /// Creates a regressor using a fully specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - A fully prepared [`Config`] instance.
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Creates a regressor using the default configuration, initialized
    /// with the specified operation modules.
    ///
    /// This is the recommended entry point if you want to use the builder pattern
    /// to chain configuration updates.
    ///
    /// # Arguments
    ///
    /// * `allowed_modules` - A vector of [`OpModule`]s defining the allowed mathematical operations.
    pub fn default_with_modules(allowed_modules: Vec<OpModule>) -> Self {
        Self {
            config: Config::default(allowed_modules),
        }
    }

    /// Sets the target Mean Squared Error.
    ///
    /// If the engine finds an equation with an MSE less than or equal to this target,
    /// it will stop the evolutionary search early.
    pub fn target_mse(mut self, target: f32) -> Self {
        self.config.target_mse = target;
        self
    }

    /// Sets the maximum number of generations the genetic engine will run.
    pub fn generations(mut self, gens: usize) -> Self {
        self.config.max_generations = gens;
        self
    }

    /// Adds an additional operation module to the regressor's configuration.
    ///
    /// If the module is already present, this method does nothing.
    pub fn with_module(mut self, module: OpModule) -> Self {
        if !self.config.allowed_modules.contains(&module) {
            self.config.allowed_modules.push(module);
        }
        self
    }

    pub fn train_subset_size(mut self, size: usize) -> Self {
        self.config.subset_size = Some(size);
        self
    }

    /// Executes the symbolic regression process on the provided dataset.
    ///
    /// This method initializes the genetic [`Engine`], runs the evolutionary loop 
    /// (including periodic continuous optimization), and returns the best equation found.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The [`Dataset`] containing the normalized input features and target values.
    ///
    /// # Returns
    ///
    /// A [`FitResult`] containing the simplified equation string, its final MSE, and complexity.
    pub fn fit(&self, full_dataset: &Dataset) -> FitResult {
        let strategy = StaticStrategy::new(self.config.clone());
        let allowed_ops = strategy.get_allowed_operators();

        let train_data = if let Some(size) = self.config.subset_size {
            full_dataset.subset(size)
        } else {
            full_dataset.clone() 
        };

        let mut engine = Engine::new(strategy, train_data.get_variable_registry(), allowed_ops);
        engine.run(&train_data);

        let mut best = engine.get_global_best().clone();
        let final_mse = best.calculate_mse(full_dataset);
        let clean_eq = crate::engine::ffi::symengine::simplify_symengine(&best.to_string());

        FitResult {
            equation: clean_eq,
            mse: final_mse,
            complexity: best.complexity(),
        }
    }
}
