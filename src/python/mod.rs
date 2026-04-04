//! Feature-gated Python integration helpers for downstream `pyo3` crates and the optional mixed
//! Python package build.
//!
//! Configs and run options are meant to be shared as pure Python objects and parsed here via
//! duck-typed extraction. Summaries remain native Rust `#[pyclass]` wrappers so downstream
//! bindings can return rich result objects with minimal glue.

/// Wrapper-facing Python config extraction scaffolding.
pub mod config;

pub(crate) mod extract;

/// Wrapper-facing Python error integration scaffolding.
pub mod errors;

/// Shared numeric conversion helpers for Python-facing wrappers.
pub mod numeric;

#[cfg(feature = "python-module")]
mod module;

/// Python-facing run-options extraction support for built-in callbacks.
pub mod run_options;

/// Machine-readable schema helpers for the pure Python config contract.
pub mod schema;

/// Wrapper-facing Python status export scaffolding.
pub mod status;

/// Wrapper-facing Python summary export scaffolding.
pub mod summary;

pub use errors::{register_exceptions, GaneshConfigError, GaneshNumericalError, GaneshPyError};
pub use run_options::{
    PyAIESOptions, PyAdamEMATerminator, PyAdamOptions, PyAutocorrelationTerminator,
    PyCMAESConditionCovTerminator, PyCMAESEqualFunValuesTerminator, PyCMAESNoEffectAxisTerminator,
    PyCMAESNoEffectCoordTerminator, PyCMAESOptions, PyCMAESSigmaTerminator,
    PyCMAESStagnationTerminator, PyCMAESTolFunTerminator, PyCMAESTolXTerminator,
    PyCMAESTolXUpTerminator, PyConjugateGradientGTerminator, PyConjugateGradientOptions,
    PyDifferentialEvolutionOptions, PyESSOptions, PyLBFGSBFTerminator, PyLBFGSBGTerminator,
    PyLBFGSBInfNormGTerminator, PyLBFGSBOptions, PyNelderMeadFTerminator, PyNelderMeadOptions,
    PyNelderMeadXTerminator, PyPSOOptions, PySimulatedAnnealingOptions,
    PySimulatedAnnealingTemperatureTerminator, PyTrustRegionGTerminator, PyTrustRegionOptions,
};
pub use schema::{ConfigFieldKind, ConfigFieldSchema, ConfigSchema, HasPyConfigSchema};
pub use status::{
    register_status_types, PyEnsembleStatus, PyGradientFreeStatus, PyGradientStatus,
    PySimulatedAnnealingStatus, PyStatusMessage, PySwarmStatus,
};
pub use summary::{
    register_summary_types, IntoPySummary, PyMCMCSummary, PyMinimizationSummary,
    PyMultiStartSummary, PySimulatedAnnealingSummary,
};

#[cfg(test)]
pub(crate) fn attach_for_tests<F, R>(f: F) -> R
where
    F: for<'py> FnOnce(pyo3::Python<'py>) -> R,
{
    use std::sync::Mutex;

    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    pyo3::Python::attach(f)
}
