pub mod ast;
pub mod evolution;
pub mod ffi;
pub mod metrics;
pub mod operators;
pub mod optimization;

pub use evolution::config::EvolutionConfig;
pub use evolution::engine::Engine;
pub use metrics::dataset::SimdDataset;