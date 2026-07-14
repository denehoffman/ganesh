use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyAnyMethods};

use crate::{
    algorithms::mcmc::{AIESInit, ESSInit},
    error::GaneshResult,
    NalgebraProvider, Vector,
};

fn extract_vector(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(array) = value.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(array.as_array().iter().copied().collect());
    }
    value
        .extract::<Vec<f64>>()
        .map_err(|_| PyValueError::new_err("expected a one-dimensional float64 array or sequence"))
}

fn extract_matrix(value: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(array) = value.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(array
            .as_array()
            .rows()
            .into_iter()
            .map(|row| row.iter().copied().collect())
            .collect());
    }
    let rows = value.extract::<Vec<Vec<f64>>>().map_err(|_| {
        PyValueError::new_err("expected a two-dimensional float64 array or nested sequence")
    })?;
    if rows.is_empty() || rows[0].is_empty() || rows.iter().any(|row| row.len() != rows[0].len()) {
        return Err(PyValueError::new_err(
            "matrix inputs must be non-empty and rectangular",
        ));
    }
    Ok(rows)
}

pub(super) fn vector_to_py(py: Python<'_>, values: Vec<f64>) -> Py<PyArray1<f64>> {
    PyArray1::from_vec(py, values).unbind()
}

pub(super) fn matrix_to_py(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    values: Vec<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let array = numpy::ndarray::Array2::from_shape_vec((rows, cols), values)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyArray2::from_owned_array(py, array).unbind())
}

/// A validated starting vector shared by minimizers and particle methods.
#[pyclass(name = "VectorInit", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyVectorInit {
    values: Vec<f64>,
}

#[pymethods]
impl PyVectorInit {
    #[new]
    fn new(values: &Bound<'_, PyAny>) -> PyResult<Self> {
        let values = extract_vector(values)?;
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "initial values must be non-empty and finite",
            ));
        }
        Ok(Self { values })
    }

    #[getter]
    fn values(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        vector_to_py(py, self.values.clone())
    }
}

impl PyVectorInit {
    /// Convert to Ganesh's default vector type.
    #[must_use]
    pub fn to_rust(&self) -> Vector<f64, NalgebraProvider> {
        Vector::from_vec(self.values.clone())
    }
}

macro_rules! ensemble_init {
    ($name:ident, $python:literal, $rust:ty) => {
        #[doc = concat!("Python-facing validated `", $python, "` initialization.")]
        #[pyclass(name = $python, frozen, from_py_object)]
        #[derive(Clone, Debug)]
        pub struct $name {
            walkers: Vec<Vec<f64>>,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new(walkers: &Bound<'_, PyAny>) -> PyResult<Self> {
                let walkers = extract_matrix(walkers)?;
                let vectors = walkers
                    .iter()
                    .cloned()
                    .map(Vector::from_vec)
                    .collect::<Vec<Vector<f64, NalgebraProvider>>>();
                <$rust>::new(vectors).map_err(super::ganesh_error)?;
                Ok(Self { walkers })
            }

            #[getter]
            fn walkers(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
                matrix_to_py(
                    py,
                    self.walkers.len(),
                    self.walkers[0].len(),
                    self.walkers.iter().flatten().copied().collect(),
                )
            }
        }

        impl $name {
            #[doc = concat!("Convert to Ganesh's `", $python, "`.")]
            pub fn to_rust(&self) -> GaneshResult<$rust> {
                <$rust>::new(self.walkers.iter().cloned().map(Vector::from_vec).collect())
            }
        }
    };
}

ensemble_init!(PyAIESInit, "AIESInit", AIESInit<f64, NalgebraProvider>);
ensemble_init!(PyESSInit, "ESSInit", ESSInit<f64, NalgebraProvider>);
