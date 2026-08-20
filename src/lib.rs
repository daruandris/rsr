pub mod engine;
pub mod domains;

pub mod api;
pub mod prelude;

crate::compose_engine!(
    SymbolicEngine,
    Basic => crate::domains::basic::BasicDomain,
    Linalg => crate::domains::linalg::LinalgDomain
);

pub use prelude::*;
pub use api::*;