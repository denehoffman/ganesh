//! Python-facing config conversion traits for downstream wrapper crates.

use pyo3::{Bound, PyAny, PyResult};

/// Convert a Python-facing wrapper object into a native `ganesh` config.
///
/// This trait is intended for typed `#[pyclass]` wrapper objects used by downstream crates. The
/// goal is to keep config parsing typed rather than dictionary-based.
pub trait FromPyConfig<'py>: Sized {
    /// Construct `Self` from a Python-facing wrapper object.
    fn from_py_config(obj: &Bound<'py, PyAny>) -> PyResult<Self>;
}
