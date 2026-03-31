//! Python error scaffolding for downstream wrapper crates.

use pyo3::{exceptions::PyRuntimeError, PyErr};

use crate::error::GaneshError;

/// Convert a [`GaneshError`] into a generic Python runtime error.
///
/// More specific Python exception mapping can be layered on top of this in later wrapper work.
impl From<GaneshError> for PyErr {
    fn from(err: GaneshError) -> Self {
        PyRuntimeError::new_err(err.to_string())
    }
}
