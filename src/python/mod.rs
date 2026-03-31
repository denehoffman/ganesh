//! Feature-gated Python integration helpers for downstream Rust crates using `pyo3`.
//!
//! The items in this module are intended to reduce wrapper boilerplate in projects that expose
//! Python bindings around `ganesh`.

/// Wrapper-facing Python config conversion scaffolding.
pub mod config;

/// Wrapper-facing Python error integration scaffolding.
pub mod errors;

/// Shared numeric conversion helpers for Python-facing wrappers.
pub mod numeric;

/// Machine-readable schema helpers for typed Python config wrappers.
pub mod schema;

/// Wrapper-facing Python summary export scaffolding.
pub mod summary;

pub use config::{
    FromPyConfig, PyAIESConfig, PyCMAESConfig, PyDifferentialEvolutionConfig, PyESSConfig,
    PyLBFGSBConfig, PyNelderMeadConfig, PyPSOConfig, PySimulatedAnnealingConfig,
};
pub use errors::{
    register_exceptions, GaneshConfigError, GaneshNumericalError, GaneshPyError,
};
pub use schema::{ConfigFieldKind, ConfigFieldSchema, ConfigSchema, HasPyConfigSchema};
pub use summary::{
    IntoPySummary, PyMCMCSummary, PyMinimizationSummary, PySimulatedAnnealingSummary,
};
