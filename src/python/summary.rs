//! Python-facing summary export traits for downstream wrapper crates.

use pyo3::{
    pyclass, pymethods,
    types::{PyDict, PyDictMethods, PyModule, PyModuleMethods},
    Bound, IntoPyObject, Py, PyAny, PyResult, Python,
};

use crate::{
    algorithms::mcmc::ChainStorageMode,
    core::{
        MCMCDiagnostics, MCMCSummary, MinimizationSummary, MultiStartSummary,
        SimulatedAnnealingSummary,
    },
    python::numeric::{matrix_to_python, tensor3_to_python, vector_to_python},
};

/// Export a native `ganesh` summary into Python-facing forms.
///
/// Downstream wrapper crates can implement this trait for selected summary types and expose either
/// dictionary-style summaries, typed `#[pyclass]` wrappers, or both.
pub trait IntoPySummary {
    /// Convert the summary into a Python dictionary.
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>>;

    /// Convert the summary into a typed Python wrapper object.
    fn to_py_class<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;
}

/// Register the built-in Python summary wrapper classes in a native module.
pub fn register_summary_types(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyMinimizationSummary>()?;
    module.add_class::<PyMCMCSummary>()?;
    module.add_class::<PyMultiStartSummary>()?;
    module.add_class::<PySimulatedAnnealingSummary>()?;
    Ok(())
}

fn bounds_to_python(
    bounds: &crate::core::transforms::Bounds,
) -> Vec<(Option<crate::Float>, Option<crate::Float>)> {
    bounds.iter().map(|(bound, _)| bound.as_options()).collect()
}

fn message_to_python<'py>(
    py: Python<'py>,
    message: &crate::traits::StatusMessage,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("status_type", message.status_type.to_string())?;
    dict.set_item("text", message.text.clone())?;
    dict.set_item("success", message.success())?;
    Ok(dict)
}

fn chain_storage_to_python<'py>(
    py: Python<'py>,
    chain_storage: ChainStorageMode,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match chain_storage {
        ChainStorageMode::Full => {
            dict.set_item("mode", "Full")?;
        }
        ChainStorageMode::Rolling { window } => {
            dict.set_item("mode", "Rolling")?;
            dict.set_item("window", window)?;
        }
        ChainStorageMode::Sampled {
            keep_every,
            max_samples,
        } => {
            dict.set_item("mode", "Sampled")?;
            dict.set_item("keep_every", keep_every)?;
            dict.set_item("max_samples", max_samples)?;
        }
    }
    Ok(dict)
}

fn diagnostics_to_python<'py>(
    py: Python<'py>,
    diagnostics: &MCMCDiagnostics,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("r_hat", vector_to_python(py, diagnostics.r_hat.as_slice())?)?;
    dict.set_item("ess", vector_to_python(py, diagnostics.ess.as_slice())?)?;
    dict.set_item(
        "acceptance_rates",
        vector_to_python(py, diagnostics.acceptance_rates.as_slice())?,
    )?;
    dict.set_item("mean_acceptance_rate", diagnostics.mean_acceptance_rate)?;
    Ok(dict)
}

fn chain_to_python(chain: &[Vec<crate::DVector<crate::Float>>]) -> Vec<Vec<Vec<crate::Float>>> {
    chain
        .iter()
        .map(|walker| {
            walker
                .iter()
                .map(|position| position.as_slice().to_vec())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn flat_chain_to_python(chain: &[crate::DVector<crate::Float>]) -> Vec<Vec<crate::Float>> {
    chain
        .iter()
        .map(|position| position.as_slice().to_vec())
        .collect()
}

/// Python-facing typed wrapper for [`MinimizationSummary`].
///
/// Notes
/// -----
/// This wrapper is returned by Python-facing optimization APIs. Numeric vector
/// and matrix fields are exposed as NumPy arrays.
#[pyclass(skip_from_py_object, module = "ganesh", name = "MinimizationSummary")]
#[derive(Clone)]
pub struct PyMinimizationSummary {
    summary: MinimizationSummary,
}

#[pymethods]
impl PyMinimizationSummary {
    /// Optional parameter bounds.
    ///
    /// Returns
    /// -------
    /// list[tuple[float | None, float | None]] | None
    ///     Bounds represented as ``(lower, upper)`` pairs, where ``None``
    ///     means unbounded on that side.
    #[getter]
    pub fn bounds(&self) -> Option<Vec<(Option<crate::Float>, Option<crate::Float>)>> {
        self.summary.bounds.as_ref().map(bounds_to_python)
    }

    /// Optional parameter names.
    ///
    /// Returns
    /// -------
    /// list[str] | None
    #[getter]
    pub fn parameter_names(&self) -> Option<Vec<String>> {
        self.summary.parameter_names.clone()
    }

    /// Summary status type.
    ///
    /// Returns
    /// -------
    /// str
    #[getter]
    pub fn status_type(&self) -> String {
        self.summary.message.status_type.to_string()
    }

    /// Human-readable status message.
    ///
    /// Returns
    /// -------
    /// str
    #[getter]
    pub fn message_text(&self) -> String {
        self.summary.message.text.clone()
    }

    /// Whether the run reported success.
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    pub fn success(&self) -> bool {
        self.summary.message.success()
    }

    /// Initial parameter vector.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    #[getter]
    pub fn x0<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.summary.x0.as_slice())
    }

    /// Final parameter vector.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    #[getter]
    pub fn x<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.summary.x.as_slice())
    }

    /// Parameter standard deviations.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    #[getter]
    pub fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.summary.std.as_slice())
    }

    /// Final objective value.
    ///
    /// Returns
    /// -------
    /// float
    #[getter]
    pub fn fx(&self) -> crate::Float {
        self.summary.fx
    }

    /// Number of cost-function evaluations.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub const fn cost_evals(&self) -> usize {
        self.summary.cost_evals
    }

    /// Number of gradient evaluations.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub const fn gradient_evals(&self) -> usize {
        self.summary.gradient_evals
    }

    /// Estimated covariance matrix.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    #[getter]
    pub fn covariance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let covariance = self
            .summary
            .covariance
            .row_iter()
            .map(|row| row.iter().copied().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        matrix_to_python(py, &covariance)
    }

    /// Export the wrapped summary as a plain Python dictionary.
    ///
    /// Returns
    /// -------
    /// dict
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.summary.to_py_dict(py)
    }
}

impl From<MinimizationSummary> for PyMinimizationSummary {
    fn from(summary: MinimizationSummary) -> Self {
        Self { summary }
    }
}

/// Python-facing typed wrapper for [`MCMCSummary`].
///
/// Notes
/// -----
/// Numeric arrays are exposed as NumPy arrays. Chain post-processing methods
/// use keyword-only ``burn`` and ``thin`` arguments so the retained-chain view
/// is explicit at each call site.
#[pyclass(skip_from_py_object, module = "ganesh", name = "MCMCSummary")]
#[derive(Clone)]
pub struct PyMCMCSummary {
    summary: MCMCSummary,
}

#[pymethods]
impl PyMCMCSummary {
    /// Optional parameter bounds.
    ///
    /// Returns
    /// -------
    /// list[tuple[float | None, float | None]] | None
    #[getter]
    pub fn bounds(&self) -> Option<Vec<(Option<crate::Float>, Option<crate::Float>)>> {
        self.summary.bounds.as_ref().map(bounds_to_python)
    }

    /// Optional parameter names.
    ///
    /// Returns
    /// -------
    /// list[str] | None
    #[getter]
    pub fn parameter_names(&self) -> Option<Vec<String>> {
        self.summary.parameter_names.clone()
    }

    /// Summary status type.
    ///
    /// Returns
    /// -------
    /// str
    #[getter]
    pub fn status_type(&self) -> String {
        self.summary.message.status_type.to_string()
    }

    /// Human-readable status message.
    ///
    /// Returns
    /// -------
    /// str
    #[getter]
    pub fn message_text(&self) -> String {
        self.summary.message.text.clone()
    }

    /// Whether the run reported success.
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    pub fn success(&self) -> bool {
        self.summary.message.success()
    }

    /// Chain-storage description.
    ///
    /// Returns
    /// -------
    /// dict
    pub fn chain_storage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        chain_storage_to_python(py, self.summary.chain_storage)
    }

    /// Number of cost-function evaluations.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub const fn cost_evals(&self) -> usize {
        self.summary.cost_evals
    }

    /// Number of gradient evaluations.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub const fn gradient_evals(&self) -> usize {
        self.summary.gradient_evals
    }

    /// Retained-chain dimensions.
    ///
    /// Returns
    /// -------
    /// tuple[int, int, int]
    ///     Dimensions in ``(n_walkers, n_steps, n_variables)`` order.
    #[getter]
    pub const fn dimension(&self) -> (usize, usize, usize) {
        self.summary.dimension
    }

    /// Compute convergence and efficiency diagnostics.
    ///
    /// Parameters
    /// ----------
    /// burn : int | None, optional
    ///     Number of retained steps discarded from the front of each walker
    ///     chain.
    /// thin : int | None, optional
    ///     Retain every ``thin``-th sample after burn-in.
    ///
    /// Returns
    /// -------
    /// dict
    ///     Dictionary with ``r_hat``, ``ess``, ``acceptance_rates``, and
    ///     ``mean_acceptance_rate`` entries.
    #[pyo3(signature = (*, burn=None, thin=None))]
    pub fn diagnostics<'py>(
        &self,
        py: Python<'py>,
        burn: Option<usize>,
        thin: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let diagnostics = self.summary.diagnostics(burn, thin);
        diagnostics_to_python(py, &diagnostics)
    }

    /// Get the retained chain after optional burn-in and thinning.
    ///
    /// Parameters
    /// ----------
    /// burn : int | None, optional
    ///     Number of retained steps discarded from the front of each walker
    ///     chain.
    /// thin : int | None, optional
    ///     Retain every ``thin``-th sample after burn-in.
    /// flat : bool, default=False
    ///     If ``False``, return the chain with shape
    ///     ``(n_walkers, n_steps, n_dim)``. If ``True``, flatten the walker and
    ///     step dimensions and return shape ``(n_samples, n_dim)``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    #[pyo3(signature = (*, burn=None, thin=None, flat=false))]
    pub fn chain<'py>(
        &self,
        py: Python<'py>,
        burn: Option<usize>,
        thin: Option<usize>,
        flat: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if flat {
            return matrix_to_python(
                py,
                &flat_chain_to_python(&self.summary.get_flat_chain(burn, thin)),
            );
        }
        tensor3_to_python(py, &chain_to_python(&self.summary.get_chain(burn, thin)))
    }

    /// Export the wrapped summary as a plain Python dictionary.
    ///
    /// Returns
    /// -------
    /// dict
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.summary.to_py_dict(py)
    }
}

impl From<MCMCSummary> for PyMCMCSummary {
    fn from(summary: MCMCSummary) -> Self {
        Self { summary }
    }
}

/// Python-facing typed wrapper for [`MultiStartSummary`].
///
/// Notes
/// -----
/// Each completed run is exposed as a [`MinimizationSummary`] wrapper.
#[pyclass(skip_from_py_object, module = "ganesh", name = "MultiStartSummary")]
#[derive(Clone)]
pub struct PyMultiStartSummary {
    summary: MultiStartSummary,
}

#[pymethods]
impl PyMultiStartSummary {
    /// Completed run summaries.
    ///
    /// Returns
    /// -------
    /// list[MinimizationSummary]
    #[getter]
    pub fn runs<'py>(&self, py: Python<'py>) -> PyResult<Vec<Py<PyMinimizationSummary>>> {
        self.summary
            .runs
            .iter()
            .cloned()
            .map(PyMinimizationSummary::from)
            .map(|summary| Py::new(py, summary))
            .collect()
    }

    /// Index of the best completed run.
    ///
    /// Returns
    /// -------
    /// int | None
    #[getter]
    pub const fn best_run_index(&self) -> Option<usize> {
        self.summary.best_run_index
    }

    /// Best completed run summary.
    ///
    /// Returns
    /// -------
    /// MinimizationSummary | None
    #[getter]
    pub fn best_run<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Py<PyMinimizationSummary>>> {
        self.summary
            .best()
            .cloned()
            .map(PyMinimizationSummary::from)
            .map(|summary| Py::new(py, summary))
            .transpose()
    }

    /// Number of completed restarts, excluding the first run.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub const fn restart_count(&self) -> usize {
        self.summary.restart_count
    }

    /// Number of completed runs.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn completed_runs(&self) -> usize {
        self.summary.completed_runs()
    }

    /// Export the wrapped summary as a plain Python dictionary.
    ///
    /// Returns
    /// -------
    /// dict
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.summary.to_py_dict(py)
    }
}

impl From<MultiStartSummary> for PyMultiStartSummary {
    fn from(summary: MultiStartSummary) -> Self {
        Self { summary }
    }
}

/// Python-facing typed wrapper for [`SimulatedAnnealingSummary<crate::DVector<crate::Float>>`].
///
/// Notes
/// -----
/// Numeric vector fields are exposed as NumPy arrays.
#[pyclass(
    skip_from_py_object,
    module = "ganesh",
    name = "SimulatedAnnealingSummary"
)]
#[derive(Clone)]
pub struct PySimulatedAnnealingSummary {
    summary: SimulatedAnnealingSummary<crate::DVector<crate::Float>>,
}

#[pymethods]
impl PySimulatedAnnealingSummary {
    /// Optional parameter bounds.
    ///
    /// Returns
    /// -------
    /// list[tuple[float | None, float | None]] | None
    #[getter]
    pub fn bounds(&self) -> Option<Vec<(Option<crate::Float>, Option<crate::Float>)>> {
        self.summary.bounds.as_ref().map(bounds_to_python)
    }

    /// Summary status type.
    ///
    /// Returns
    /// -------
    /// str
    #[getter]
    pub fn status_type(&self) -> String {
        self.summary.message.status_type.to_string()
    }

    /// Human-readable status message.
    ///
    /// Returns
    /// -------
    /// str
    #[getter]
    pub fn message_text(&self) -> String {
        self.summary.message.text.clone()
    }

    /// Whether the run reported success.
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    pub fn success(&self) -> bool {
        self.summary.message.success()
    }

    /// Initial state vector.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    #[getter]
    pub fn x0<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.summary.x0.as_slice())
    }

    /// Best state found during the run.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    #[getter]
    pub fn x<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.summary.x.as_slice())
    }

    /// Best objective value.
    ///
    /// Returns
    /// -------
    /// float
    #[getter]
    pub fn fx(&self) -> crate::Float {
        self.summary.fx
    }

    /// Number of cost-function evaluations.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub const fn cost_evals(&self) -> usize {
        self.summary.cost_evals
    }

    /// Export the wrapped summary as a plain Python dictionary.
    ///
    /// Returns
    /// -------
    /// dict
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.summary.to_py_dict(py)
    }
}

impl From<SimulatedAnnealingSummary<crate::DVector<crate::Float>>> for PySimulatedAnnealingSummary {
    fn from(summary: SimulatedAnnealingSummary<crate::DVector<crate::Float>>) -> Self {
        Self { summary }
    }
}

impl IntoPySummary for MinimizationSummary {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("bounds", self.bounds.as_ref().map(bounds_to_python))?;
        dict.set_item("parameter_names", self.parameter_names.clone())?;
        dict.set_item("message", message_to_python(py, &self.message)?)?;

        dict.set_item("x0", vector_to_python(py, self.x0.as_slice())?)?;
        dict.set_item("x", vector_to_python(py, self.x.as_slice())?)?;
        dict.set_item("std", vector_to_python(py, self.std.as_slice())?)?;
        dict.set_item("fx", self.fx)?;
        dict.set_item("cost_evals", self.cost_evals)?;
        dict.set_item("gradient_evals", self.gradient_evals)?;
        let covariance = self
            .covariance
            .row_iter()
            .map(|row| row.iter().copied().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        dict.set_item("covariance", matrix_to_python(py, &covariance)?)?;
        Ok(dict)
    }

    fn to_py_class<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let wrapper = Py::new(py, PyMinimizationSummary::from(self.clone()))?;
        Ok(wrapper.into_bound(py).into_any())
    }
}

impl IntoPySummary for MCMCSummary {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("bounds", self.bounds.as_ref().map(bounds_to_python))?;
        dict.set_item("parameter_names", self.parameter_names.clone())?;
        dict.set_item("message", message_to_python(py, &self.message)?)?;
        dict.set_item(
            "chain",
            tensor3_to_python(py, &chain_to_python(&self.chain))?,
        )?;
        dict.set_item(
            "chain_storage",
            chain_storage_to_python(py, self.chain_storage)?,
        )?;
        dict.set_item("cost_evals", self.cost_evals)?;
        dict.set_item("gradient_evals", self.gradient_evals)?;
        dict.set_item("dimension", self.dimension)?;
        Ok(dict)
    }

    fn to_py_class<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let wrapper = Py::new(py, PyMCMCSummary::from(self.clone()))?;
        Ok(wrapper.into_bound(py).into_any())
    }
}

impl IntoPySummary for MultiStartSummary {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let runs = self
            .runs
            .iter()
            .map(|run| run.to_py_dict(py).map(|bound| bound.unbind()))
            .collect::<PyResult<Vec<_>>>()?;
        let best_run = self
            .best()
            .map(|run| run.to_py_dict(py).map(|bound| bound.unbind()))
            .transpose()?;
        dict.set_item("runs", runs)?;
        dict.set_item("best_run_index", self.best_run_index)?;
        dict.set_item("best_run", best_run)?;
        dict.set_item("restart_count", self.restart_count)?;
        dict.set_item("completed_runs", self.completed_runs())?;
        Ok(dict)
    }

    fn to_py_class<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let wrapper = Py::new(py, PyMultiStartSummary::from(self.clone()))?;
        Ok(wrapper.into_bound(py).into_any())
    }
}

impl IntoPySummary for SimulatedAnnealingSummary<crate::DVector<crate::Float>> {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("bounds", self.bounds.as_ref().map(bounds_to_python))?;
        dict.set_item("message", message_to_python(py, &self.message)?)?;
        dict.set_item("x0", vector_to_python(py, self.x0.as_slice())?)?;
        dict.set_item("x", vector_to_python(py, self.x.as_slice())?)?;
        dict.set_item("fx", self.fx)?;
        dict.set_item("cost_evals", self.cost_evals)?;
        Ok(dict)
    }

    fn to_py_class<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let wrapper = Py::new(py, PySimulatedAnnealingSummary::from(self.clone()))?;
        Ok(wrapper.into_bound(py).into_any())
    }
}

impl<'py> IntoPyObject<'py> for MinimizationSummary {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(py, PyMinimizationSummary::from(self))?
            .into_bound(py)
            .into_any())
    }
}

impl<'py> IntoPyObject<'py> for MCMCSummary {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(py, PyMCMCSummary::from(self))?
            .into_bound(py)
            .into_any())
    }
}

impl<'py> IntoPyObject<'py> for MultiStartSummary {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(py, PyMultiStartSummary::from(self))?
            .into_bound(py)
            .into_any())
    }
}

impl<'py> IntoPyObject<'py> for SimulatedAnnealingSummary<crate::DVector<crate::Float>> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(py, PySimulatedAnnealingSummary::from(self))?
            .into_bound(py)
            .into_any())
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{types::PyAnyMethods, Py};

    use super::*;
    use crate::{core::transforms::Bounds, traits::StatusMessage, DMatrix, DVector};

    fn sample_summary() -> MinimizationSummary {
        MinimizationSummary {
            bounds: Some(Bounds::new_default([
                (Some(-1.0), Some(1.0)),
                (None, Some(2.0)),
            ])),
            parameter_names: Some(vec!["alpha".into(), "beta".into()]),
            message: StatusMessage::default().set_success_with_message("ok"),
            x0: DVector::from_vec(vec![1.0, 2.0]),
            x: DVector::from_vec(vec![0.5, 1.5]),
            std: DVector::from_vec(vec![0.1, 0.2]),
            fx: 1.25,
            cost_evals: 10,
            gradient_evals: 4,
            covariance: DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
        }
    }

    #[test]
    fn native_summary_into_pyobject_returns_typed_wrapper() {
        crate::python::attach_for_tests(|py| {
            let wrapper = sample_summary().into_pyobject(py).unwrap();
            let wrapper = wrapper.extract::<Py<PyMinimizationSummary>>().unwrap();
            let wrapper = wrapper.bind(py).borrow();
            assert_eq!(wrapper.fx(), 1.25);
            assert_eq!(wrapper.status_type(), "Success");
        });
    }
}
