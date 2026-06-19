//! Python error integration for downstream wrapper crates.
#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]

use pyo3::{
    create_exception,
    exceptions::PyException,
    types::{PyModule, PyModuleMethods},
    Bound, PyErr, PyResult,
};

use crate::error::GaneshError;

// Base Python exception for `ganesh` wrapper-facing errors.
create_exception!(ganesh, GaneshPyError, PyException);

// Python exception for wrapper-facing configuration errors.
create_exception!(ganesh, GaneshConfigError, GaneshPyError);

// Python exception for wrapper-facing numerical errors.
create_exception!(ganesh, GaneshNumericalError, GaneshPyError);

/// Register the `ganesh` Python exception hierarchy in a downstream `pyo3` module.
pub fn register_exceptions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("GaneshError", module.py().get_type::<GaneshPyError>())?;
    module.add(
        "GaneshConfigError",
        module.py().get_type::<GaneshConfigError>(),
    )?;
    module.add(
        "GaneshNumericalError",
        module.py().get_type::<GaneshNumericalError>(),
    )?;
    Ok(())
}

/// Convert a [`GaneshError`] into a typed Python exception.
impl From<GaneshError> for PyErr {
    fn from(err: GaneshError) -> Self {
        match err {
            GaneshError::ConfigError(message) => GaneshConfigError::new_err(message),
            GaneshError::NumericalError(message) => GaneshNumericalError::new_err(message),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::GaneshError;

    #[test]
    fn config_error_maps_to_python_config_exception() {
        crate::python::attach_for_tests(|py| {
            let err: PyErr = GaneshError::ConfigError("bad config".into()).into();
            assert!(err.is_instance_of::<GaneshConfigError>(py));
            assert!(err.is_instance_of::<GaneshPyError>(py));
        });
    }

    #[test]
    fn numerical_error_maps_to_python_numerical_exception() {
        crate::python::attach_for_tests(|py| {
            let err: PyErr = GaneshError::NumericalError("bad conditioning".into()).into();
            assert!(err.is_instance_of::<GaneshNumericalError>(py));
            assert!(err.is_instance_of::<GaneshPyError>(py));
        });
    }
}
