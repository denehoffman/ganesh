//! Feature-gated Python integration helpers for downstream Rust crates using `pyo3`.
//!
//! The items in this module are intended to reduce wrapper boilerplate in projects that expose
//! Python bindings around `ganesh`.

/// Wrapper-facing Python config conversion scaffolding.
pub mod config;

/// Wrapper-facing Python error integration scaffolding.
pub mod errors;

/// Wrapper-facing Python summary export scaffolding.
pub mod summary;

pub use config::{FromPyConfig, PyLBFGSBConfig};
pub use errors::{
    register_exceptions, GaneshConfigError, GaneshNumericalError, GaneshPyError,
};
pub use summary::{IntoPySummary, PyMinimizationSummary};
