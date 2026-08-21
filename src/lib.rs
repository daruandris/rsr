//!RSR is a high-performance, SIMD-accelerated genetic programming engine written in Rust, specialized for symbolic regression. It is designed to automatically discover mathematical equations and physical invariants directly from dataset observations.
//!Unlike standard symbolic regression tools, RSR natively supports Tensor and Linear Algebra operations(2 and 3 dimensional vectors and matrices), combined with Forward-Mode Automatic Differentiation using custom SIMD-optimized dual numbers.
//!
//!## Key features
//!
//! * **Fast Evaluation**: A custom virtual machine leveraging the `wide` crate for SIMD (`f32x4`) execution, evaluating multiple data points in parallel.
//! * **Zero-Cost Abstractions**: The hot paths of the engine—including evaluation, crossover, mutation, and AST simplification—are carefully designed to operate entirely on the stack, avoiding costly heap allocations.
//! * **Automatic Differentiation**: Built-in forward-mode AD using dual numbers provides exact gradients, powering the L-BFGS optimizer for rapid constant fine-tuning.
//! * **Hybrid Search Strategy**:
//!   * **Island-model GP** handles the structural search across isolated populations.
//!   * **Continuous Optimization** (L-BFGS for differentiable trees, CMA-ES / Nelder-Mead for non-differentiable or complex algebraic structures) optimizes the numeric constants within the formulas.
//! * **Multi-Domain Support**:
//!   * `Basic`: Scalar arithmetic, trigonometry, exponentials, and logarithms.
//!   * `Linalg`: Comprehensive vector and matrix operations (`Vec2`, `Vec3`, `Mat2`, `Mat3`) with specialized dual-number evaluations.
//! * **Algebraic Simplification**: Integrates seamlessly with SymEngine (via C++ FFI) to prune
//!
//! ## Quick Start
//!
//! ```rust
//! use rsr::prelude::*;
//! use rsr::api::SymbolicRegressor;
//! use rsr::engine::data::schema::Schema;
//!
//! // 1. Prepare your data (e.g., y = 2.0 * x^2)
//! let data_x = vec![
//!     vec![1.0], vec![2.0], vec![3.0], vec![4.0]
//! ];
//! let data_y = vec![2.0, 8.0, 18.0, 32.0];
//!
//! // 2. Define the schema and create a dataset
//! let schema = Schema::new(vec![ValueType::Float])
//!     .with_normalization(false);
//! let dataset = Dataset::from_arrays(&data_x, &data_y, &schema);
//!
//! // 3. Configure and run the regressor
//! let config = Config::default(vec![OpModule::Basic]);
//!
//! let regressor = SymbolicRegressor { config };
//! let result = regressor.fit(&dataset);
//!
//! println!("Found equation: {}", result.equation);
//! println!("MSE: {}", result.mse);
//! ```
//!
//! ## Architecture Overview
//!
//! - [`engine`]: Contains the core logic including the expressions (`expr`), evaluation (`eval`), search algorithms (`search`), and numerical optimizers (`optimize`).
//! - [`domains`]: Defines the available instruction sets and mathematical operations (e.g., Basic math, Linear Algebra).
//! - [`api`]: Provides the high-level, user-facing `SymbolicRegressor` interface.

pub mod domains;
pub mod engine;

pub mod api;
pub mod prelude;

crate::compose_engine!(
    SymbolicEngine,
    Basic => crate::domains::basic::BasicDomain,
    Linalg => crate::domains::linalg::LinalgDomain
);

pub use api::*;
pub use prelude::*;
