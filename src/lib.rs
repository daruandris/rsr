pub mod domain;
pub mod data;
pub mod eval;
pub mod expr;
pub mod ffi;
pub mod optimize;
pub mod prelude;
pub mod search;

crate::compose_engine!(
    SymbolicEngine,
    Basic => crate::eval::basic_domain::BasicDomain,
    Linalg => crate::eval::linalg_domain::LinalgDomain
);

pub use prelude::*;
