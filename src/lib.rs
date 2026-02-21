pub mod ast;
pub mod domain;
pub mod engine;
pub mod ffi;
pub mod metrics;
pub mod operators;
pub mod optimization;

pub use engine::Engine;
pub use engine::strategy::{Strategy, StaticStrategy};
pub use domain::universal::{UniversalDomain, UniversalOp, UniversalType};
pub use metrics::dataset::SimdDataset;
pub use engine::config::EvolutionConfig;