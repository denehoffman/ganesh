//! Shared numeric conversion helpers for Python-facing wrapper types.

use pyo3::{
    types::PyAnyMethods,
    Bound, PyAny, PyResult, Python,
};

use crate::Float;

#[cfg(feature = "python-numpy")]
use numpy::{
    ndarray::{Array2, Array3},
    PyArray1, PyArray2, PyArray3, PyArrayMethods,
};
#[cfg(feature = "python-numpy")]
use pyo3::exceptions::PyTypeError;
#[cfg(not(feature = "python-numpy"))]
use pyo3::types::PyList;
#[cfg(all(test, feature = "python-numpy"))]
use std::sync::Once;

/// Extract a numeric vector from a Python value.
pub fn extract_vector(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Float>> {
    #[cfg(feature = "python-numpy")]
    {
        if let Ok(array) = obj.downcast::<PyArray1<Float>>() {
            return Ok(array.readonly().as_slice()?.to_vec());
        }
    }
    obj.extract::<Vec<Float>>()
}

/// Extract a numeric matrix from a Python value.
pub fn extract_matrix(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Float>>> {
    #[cfg(feature = "python-numpy")]
    {
        if let Ok(array) = obj.downcast::<PyArray2<Float>>() {
            let readonly = array.readonly();
            let view = readonly.as_array();
            return Ok(view.outer_iter().map(|row| row.to_vec()).collect());
        }
    }
    obj.extract::<Vec<Vec<Float>>>()
}

/// Extract a rank-3 numeric tensor from a Python value.
pub fn extract_tensor3(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Vec<Float>>>> {
    #[cfg(feature = "python-numpy")]
    {
        if let Ok(array) = obj.downcast::<PyArray3<Float>>() {
            let readonly = array.readonly();
            let view = readonly.as_array();
            return Ok(view
                .outer_iter()
                .map(|matrix| matrix.outer_iter().map(|row| row.to_vec()).collect())
                .collect());
        }
    }
    obj.extract::<Vec<Vec<Vec<Float>>>>()
}

/// Convert a numeric vector into the active Python array backend.
pub fn vector_to_python<'py>(py: Python<'py>, values: &[Float]) -> PyResult<Bound<'py, PyAny>> {
    #[cfg(feature = "python-numpy")]
    {
        Ok(PyArray1::from_slice(py, values).into_any())
    }
    #[cfg(not(feature = "python-numpy"))]
    {
        Ok(PyList::new(py, values)?.into_any())
    }
}

/// Convert a numeric matrix into the active Python array backend.
pub fn matrix_to_python<'py>(
    py: Python<'py>,
    values: &[Vec<Float>],
) -> PyResult<Bound<'py, PyAny>> {
    #[cfg(feature = "python-numpy")]
    {
        let rows = values.len();
        let cols = values.first().map_or(0, Vec::len);
        if values.iter().any(|row| row.len() != cols) {
            return Err(PyTypeError::new_err(
                "expected a rectangular matrix for NumPy conversion",
            ));
        }
        let flat = values
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        let array = Array2::from_shape_vec((rows, cols), flat)
            .map_err(|err| PyTypeError::new_err(err.to_string()))?;
        Ok(PyArray2::from_owned_array(py, array).into_any())
    }
    #[cfg(not(feature = "python-numpy"))]
    {
        let rows = values
            .iter()
            .map(|row| PyList::new(py, row))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, rows)?.into_any())
    }
}

/// Convert a three-dimensional numeric tensor into the active Python array backend.
pub fn tensor3_to_python<'py>(
    py: Python<'py>,
    values: &[Vec<Vec<Float>>],
) -> PyResult<Bound<'py, PyAny>> {
    #[cfg(feature = "python-numpy")]
    {
        let dim0 = values.len();
        let dim1 = values.first().map_or(0, Vec::len);
        let dim2 = values
            .first()
            .and_then(|rows| rows.first())
            .map_or(0, Vec::len);
        if values.iter().any(|rows| rows.len() != dim1)
            || values
                .iter()
                .flat_map(|rows| rows.iter())
                .any(|row| row.len() != dim2)
        {
            return Err(PyTypeError::new_err(
                "expected a rectangular rank-3 tensor for NumPy conversion",
            ));
        }
        let flat = values
            .iter()
            .flat_map(|rows| rows.iter().flat_map(|row| row.iter().copied()))
            .collect::<Vec<_>>();
        let array = Array3::from_shape_vec((dim0, dim1, dim2), flat)
            .map_err(|err| PyTypeError::new_err(err.to_string()))?;
        Ok(PyArray3::from_owned_array(py, array).into_any())
    }
    #[cfg(not(feature = "python-numpy"))]
    {
        let outer = values
            .iter()
            .map(|rows| {
                let middle = rows
                    .iter()
                    .map(|row| PyList::new(py, row))
                    .collect::<PyResult<Vec<_>>>()?;
                PyList::new(py, middle)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, outer)?.into_any())
    }
}

#[cfg(all(test, feature = "python-numpy"))]
pub(crate) fn ensure_numpy_test_runtime() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            py.import("datetime").unwrap();
            py.import("numpy").unwrap();
        });
    });
}
