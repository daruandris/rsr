//! High-level interface for symbolic regression.
//!
//! This module provides the user-facing [`SymbolicRegressor`] struct, which wraps
//! the underlying genetic engine and configuration into an easy-to-use API.

use crate::engine::data::dataset::Dataset;
use crate::engine::expr::program::Program;
use crate::engine::search::config::{Config, OpModule};
use crate::engine::search::engine::Engine;
use crate::engine::search::strategy::{StaticStrategy, Strategy};

use log::warn;

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
    /// The Mean Squared Error of the best equation found (scaled to the normalized data).
    pub mse: f32,
    /// The unnormalized, raw Mean Squared Error of the best equation found.
    pub clear_mse: f32,
    /// The complexity score of the equation (sum of the weights of its nodes).
    pub complexity: usize,
    /// The Pareto front containing the best equations found at each complexity level.
    pub pareto_front: Vec<ParetoEntry>,
    /// The compiled program of the best equation, used for post-evaluations (e.g., plotting).
    pub program: Option<Program>,
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
        self.config.base_target_mse = target;
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
        if !full_dataset.is_normalized && full_dataset.num_features > 1 {
            let mut min_var = f32::MAX;
            let mut max_var = f32::MIN;
            for &std in &full_dataset.feature_std_devs {
                let var = std * std;
                if var < min_var { min_var = var; }
                if var > max_var { max_var = var; }
            }
            if min_var > 0.0 && (max_var / min_var) > 100.0 {
                warn!(
                    "Large scale differences detected between input features (Max Var / Min Var > 100). \
                    Continuous optimizers (L-BFGS, CMA-ES) may become numerically unstable. \
                    Consider using `Schema::with_normalization(true)`."
                );
            }
        }
        if !full_dataset.num_samples.is_multiple_of(8) {
            warn!(
                "The full dataset size ({}) is not a multiple of 8. Padding applied. (Slight performance hit)",
                full_dataset.num_samples
            );
        }
        if let Some(size) = self.config.subset_size
            && size % 8 != 0 {
                warn!(
                    "The config.subset_size ({}) is not a multiple of 8. Padding applied.",
                    size
                );
            }
        if !self.config.mini_batch_size.is_multiple_of(8) {
            warn!(
                "The config.mini_batch_size ({}) is not a multiple of 8. Padding applied.",
                self.config.mini_batch_size
            );
        }
        
        let strategy = StaticStrategy::new(self.config.clone(), full_dataset.target_variance);
        let allowed_ops = strategy.get_allowed_operators();

        let train_data = if let Some(size) = self.config.subset_size {
            full_dataset.subset(size)
        } else {
            full_dataset.clone()
        };

        let mut engine = Engine::new(strategy, train_data.get_variable_registry(), allowed_ops);
        engine.run(&train_data);

        let mut best = engine.get_global_best().clone();
        let final_mse = best.calculate_loss(full_dataset, self.config.loss_type);
        let clear_mse = final_mse * full_dataset.target_variance;
        let clean_eq = crate::engine::ffi::symengine::simplify_symengine(&best.to_string());

        let mut pareto_front = Vec::new();
        for (comp, mse, ind) in engine.get_pareto_front() {
            let eq_str = crate::engine::ffi::symengine::simplify_symengine(&ind.to_string());
            pareto_front.push(ParetoEntry {
                equation: eq_str,
                mse,
                clear_mse: mse * full_dataset.target_variance,
                complexity: comp,
            });
        }

        FitResult {
            equation: clean_eq,
            mse: final_mse,
            clear_mse,
            complexity: best.complexity(),
            pareto_front,
            program: best.program.clone(),
        }
    }
}

/// A Pareto-front egyetlen eleme
#[derive(Clone, Debug)]
pub struct ParetoEntry {
    pub equation: String,
    pub mse: f32,
    pub clear_mse: f32,
    pub complexity: usize,
}