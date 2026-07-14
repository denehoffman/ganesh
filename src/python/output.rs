use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;

use crate::{
    core::{EvalCounts, MCMCSummary, MinimizationSummary, MultiStartSummary},
    traits::StatusMessage,
    NalgebraProvider,
};

use super::numeric::{matrix_to_py, vector_to_py};

/// Python-facing evaluation counts.
#[pyclass(name = "EvalCounts", frozen, from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyEvalCounts {
    f: usize,
    g: usize,
    h: usize,
}

#[pymethods]
impl PyEvalCounts {
    #[getter]
    const fn f(&self) -> usize {
        self.f
    }
    #[getter]
    const fn g(&self) -> usize {
        self.g
    }
    #[getter]
    const fn h(&self) -> usize {
        self.h
    }
}

impl From<EvalCounts> for PyEvalCounts {
    fn from(value: EvalCounts) -> Self {
        Self {
            f: value.f(),
            g: value.g(),
            h: value.h(),
        }
    }
}

/// Python-facing status message.
#[pyclass(name = "StatusMessage", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStatusMessage {
    success: bool,
    text: Option<String>,
    display: String,
}

#[pymethods]
impl PyStatusMessage {
    #[getter]
    const fn success(&self) -> bool {
        self.success
    }
    #[getter]
    fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }
    fn __str__(&self) -> &str {
        &self.display
    }
}

impl From<StatusMessage> for PyStatusMessage {
    fn from(value: StatusMessage) -> Self {
        Self {
            success: value.success(),
            text: value.text().map(ToOwned::to_owned),
            display: value.to_string(),
        }
    }
}

/// Python-facing minimization result.
#[pyclass(name = "MinimizationSummary", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyMinimizationSummary {
    inner: MinimizationSummary<f64, NalgebraProvider>,
}

#[pymethods]
impl PyMinimizationSummary {
    #[getter]
    fn x0(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        vector_to_py(py, self.inner.x0.to_vec())
    }
    #[getter]
    fn x(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        vector_to_py(py, self.inner.x.to_vec())
    }
    #[getter]
    fn std(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        vector_to_py(py, self.inner.std.to_vec())
    }
    #[getter]
    const fn fx(&self) -> f64 {
        self.inner.fx
    }
    #[getter]
    fn evals(&self) -> PyEvalCounts {
        self.inner.evals.into()
    }
    #[getter]
    fn message(&self) -> PyStatusMessage {
        self.inner.message.clone().into()
    }
    #[getter]
    fn parameter_names(&self) -> Option<Vec<String>> {
        self.inner.parameter_names.clone()
    }
    #[getter]
    fn covariance(&self, py: Python<'_>) -> PyResult<Py<PyArray2<f64>>> {
        let rows = self.inner.covariance.rows();
        let cols = self.inner.covariance.cols();
        let values = (0..rows)
            .flat_map(|row| (0..cols).map(move |col| self.inner.covariance.get(row, col)))
            .collect();
        matrix_to_py(py, rows, cols, values)
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

impl From<MinimizationSummary<f64, NalgebraProvider>> for PyMinimizationSummary {
    fn from(inner: MinimizationSummary<f64, NalgebraProvider>) -> Self {
        Self { inner }
    }
}

/// Python-facing multistart minimization result.
#[pyclass(name = "MultiStartSummary", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyMultiStartSummary {
    inner: MultiStartSummary<f64, NalgebraProvider>,
}

#[pymethods]
impl PyMultiStartSummary {
    #[getter]
    fn runs(&self) -> Vec<PyMinimizationSummary> {
        self.inner.runs.iter().cloned().map(Into::into).collect()
    }
    #[getter]
    const fn best_run_index(&self) -> Option<usize> {
        self.inner.best_run_index
    }
    #[getter]
    const fn restart_count(&self) -> usize {
        self.inner.restart_count
    }
    #[getter]
    fn best(&self) -> Option<PyMinimizationSummary> {
        self.inner.best().cloned().map(Into::into)
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

impl From<MultiStartSummary<f64, NalgebraProvider>> for PyMultiStartSummary {
    fn from(inner: MultiStartSummary<f64, NalgebraProvider>) -> Self {
        Self { inner }
    }
}

/// Python-facing MCMC result.
#[pyclass(name = "MCMCSummary", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyMCMCSummary {
    inner: MCMCSummary<f64, NalgebraProvider>,
}

#[pymethods]
impl PyMCMCSummary {
    #[getter]
    const fn dimension(&self) -> (usize, usize, usize) {
        self.inner.dimension
    }
    #[getter]
    fn evals(&self) -> PyEvalCounts {
        self.inner.evals.into()
    }
    #[getter]
    fn message(&self) -> PyStatusMessage {
        self.inner.message.clone().into()
    }
    #[getter]
    fn parameter_names(&self) -> Option<Vec<String>> {
        self.inner.parameter_names.clone()
    }
    #[getter]
    fn chain(&self, py: Python<'_>) -> PyResult<Py<PyArray3<f64>>> {
        let (walkers, steps, variables) = self.inner.dimension;
        let values = self
            .inner
            .chain
            .iter()
            .flatten()
            .flat_map(|point| point.to_vec())
            .collect();
        let array = numpy::ndarray::Array3::from_shape_vec((walkers, steps, variables), values)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(PyArray3::from_owned_array(py, array).unbind())
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
}

impl From<MCMCSummary<f64, NalgebraProvider>> for PyMCMCSummary {
    fn from(inner: MCMCSummary<f64, NalgebraProvider>) -> Self {
        Self { inner }
    }
}
