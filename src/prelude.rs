//! The RSR prelude.
//!
//! This module re-exports the most commonly used types required for configuring
//! and running the symbolic regressor.

pub use crate::api::{FitResult, SymbolicRegressor};
pub use crate::engine::data::dataset::Dataset;
pub use crate::engine::data::schema::Schema;
pub use crate::engine::eval::types::ValueType;
pub use crate::engine::search::config::{Config, OpModule};
pub use crate::plot::{PlotConfig, SolidPlotMode};

// Ezt meghagyjuk, mert a konfigurációnál (custom_ops, excluded_ops) szükség van rá
pub use crate::Instruction;
