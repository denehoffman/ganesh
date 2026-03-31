//! Python-facing summary export traits for downstream wrapper crates.

use pyo3::{
    pyclass, pymethods, Py,
    types::{PyDict, PyDictMethods},
    Bound, PyAny, PyResult, Python,
};

use crate::{
    algorithms::mcmc::ChainStorageMode,
    core::{MCMCDiagnostics, MCMCSummary, MinimizationSummary, SimulatedAnnealingSummary},
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

fn bounds_to_python(bounds: &crate::core::transforms::Bounds) -> Vec<(Option<crate::Float>, Option<crate::Float>)> {
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
    dict.set_item("acceptance_rates", vector_to_python(py, diagnostics.acceptance_rates.as_slice())?)?;
    dict.set_item("mean_acceptance_rate", diagnostics.mean_acceptance_rate)?;
    Ok(dict)
}

fn chain_to_python(chain: &[Vec<crate::DVector<crate::Float>>]) -> Vec<Vec<Vec<crate::Float>>> {
    chain.iter()
        .map(|walker| {
            walker
                .iter()
                .map(|position| position.as_slice().to_vec())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn flat_chain_to_python(chain: &[crate::DVector<crate::Float>]) -> Vec<Vec<crate::Float>> {
    chain.iter()
        .map(|position| position.as_slice().to_vec())
        .collect()
}

/// Python-facing typed wrapper for [`MinimizationSummary`].
#[pyclass(module = "ganesh", name = "MinimizationSummary")]
#[derive(Clone)]
pub struct PyMinimizationSummary {
    summary: MinimizationSummary,
}

#[pymethods]
impl PyMinimizationSummary {
    /// Get the optional parameter bounds.
    ///
    /// Each bound is represented as `(lower, upper)`, where `None` means unbounded on that side.
    #[getter]
    pub fn bounds(&self) -> Option<Vec<(Option<crate::Float>, Option<crate::Float>)>> {
        self.summary.bounds.as_ref().map(bounds_to_python)
    }

    /// Get the optional parameter names.
    #[getter]
    pub fn parameter_names(&self) -> Option<Vec<String>> {
        self.summary.parameter_names.clone()
    }

    /// Get the status type as a string.
    #[getter]
    pub fn status_type(&self) -> String {
        self.summary.message.status_type.to_string()
    }

    /// Get the message text.
    #[getter]
    pub fn message_text(&self) -> String {
        self.summary.message.text.clone()
    }

    /// Return `True` when the minimization succeeded.
    #[getter]
    pub fn success(&self) -> bool {
        self.summary.message.success()
    }

    /// Get the initial parameters.
    #[getter]
    pub fn x0<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.summary.x0.as_slice())
    }

    /// Get the final parameters.
    #[getter]
    pub fn x<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.summary.x.as_slice())
    }

    /// Get the parameter standard deviations.
    #[getter]
    pub fn std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.summary.std.as_slice())
    }

    /// Get the final objective value.
    #[getter]
    pub fn fx(&self) -> crate::Float {
        self.summary.fx
    }

    /// Get the number of cost evaluations.
    #[getter]
    pub const fn cost_evals(&self) -> usize {
        self.summary.cost_evals
    }

    /// Get the number of gradient evaluations.
    #[getter]
    pub const fn gradient_evals(&self) -> usize {
        self.summary.gradient_evals
    }

    /// Get the covariance matrix as nested Python lists.
    #[getter]
    pub fn covariance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let covariance = self.summary
            .covariance
            .row_iter()
            .map(|row| row.iter().copied().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        matrix_to_python(py, &covariance)
    }

    /// Export the wrapped summary as a Python dictionary.
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
#[pyclass(module = "ganesh", name = "MCMCSummary")]
#[derive(Clone)]
pub struct PyMCMCSummary {
    summary: MCMCSummary,
}

#[pymethods]
impl PyMCMCSummary {
    /// Get the optional parameter bounds.
    #[getter]
    pub fn bounds(&self) -> Option<Vec<(Option<crate::Float>, Option<crate::Float>)>> {
        self.summary.bounds.as_ref().map(bounds_to_python)
    }

    /// Get the optional parameter names.
    #[getter]
    pub fn parameter_names(&self) -> Option<Vec<String>> {
        self.summary.parameter_names.clone()
    }

    /// Get the status type as a string.
    #[getter]
    pub fn status_type(&self) -> String {
        self.summary.message.status_type.to_string()
    }

    /// Get the message text.
    #[getter]
    pub fn message_text(&self) -> String {
        self.summary.message.text.clone()
    }

    /// Return `True` when the sampler reports success.
    #[getter]
    pub fn success(&self) -> bool {
        self.summary.message.success()
    }

    /// Get the full retained chain as nested Python lists.
    pub fn chain<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        tensor3_to_python(py, &chain_to_python(&self.summary.chain))
    }

    /// Get the chain storage description.
    pub fn chain_storage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        chain_storage_to_python(py, self.summary.chain_storage)
    }

    /// Get the number of cost evaluations.
    #[getter]
    pub const fn cost_evals(&self) -> usize {
        self.summary.cost_evals
    }

    /// Get the number of gradient evaluations.
    #[getter]
    pub const fn gradient_evals(&self) -> usize {
        self.summary.gradient_evals
    }

    /// Get the retained chain dimensions `(n_walkers, n_steps, n_variables)`.
    #[getter]
    pub const fn dimension(&self) -> (usize, usize, usize) {
        self.summary.dimension
    }

    /// Compute diagnostics from the retained chain after optional burn-in and thinning.
    #[pyo3(signature = (burn=None, thin=None))]
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
    #[pyo3(signature = (burn=None, thin=None))]
    pub fn get_chain<'py>(
        &self,
        py: Python<'py>,
        burn: Option<usize>,
        thin: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        tensor3_to_python(py, &chain_to_python(&self.summary.get_chain(burn, thin)))
    }

    /// Get the flattened retained chain after optional burn-in and thinning.
    #[pyo3(signature = (burn=None, thin=None))]
    pub fn get_flat_chain<'py>(
        &self,
        py: Python<'py>,
        burn: Option<usize>,
        thin: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        matrix_to_python(py, &flat_chain_to_python(&self.summary.get_flat_chain(burn, thin)))
    }

    /// Export the wrapped summary as a Python dictionary.
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.summary.to_py_dict(py)
    }
}

impl From<MCMCSummary> for PyMCMCSummary {
    fn from(summary: MCMCSummary) -> Self {
        Self { summary }
    }
}

/// Python-facing typed wrapper for [`SimulatedAnnealingSummary<crate::DVector<crate::Float>>`].
#[pyclass(module = "ganesh", name = "SimulatedAnnealingSummary")]
#[derive(Clone)]
pub struct PySimulatedAnnealingSummary {
    summary: SimulatedAnnealingSummary<crate::DVector<crate::Float>>,
}

#[pymethods]
impl PySimulatedAnnealingSummary {
    /// Get the optional parameter bounds.
    #[getter]
    pub fn bounds(&self) -> Option<Vec<(Option<crate::Float>, Option<crate::Float>)>> {
        self.summary.bounds.as_ref().map(bounds_to_python)
    }

    /// Get the status type as a string.
    #[getter]
    pub fn status_type(&self) -> String {
        self.summary.message.status_type.to_string()
    }

    /// Get the message text.
    #[getter]
    pub fn message_text(&self) -> String {
        self.summary.message.text.clone()
    }

    /// Return `True` when the run reports success.
    #[getter]
    pub fn success(&self) -> bool {
        self.summary.message.success()
    }

    /// Get the initial state.
    #[getter]
    pub fn x0<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.summary.x0.as_slice())
    }

    /// Get the best state found.
    #[getter]
    pub fn x<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.summary.x.as_slice())
    }

    /// Get the best objective value.
    #[getter]
    pub fn fx(&self) -> crate::Float {
        self.summary.fx
    }

    /// Get the number of cost evaluations.
    #[getter]
    pub const fn cost_evals(&self) -> usize {
        self.summary.cost_evals
    }

    /// Export the wrapped summary as a Python dictionary.
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
        dict.set_item("chain", tensor3_to_python(py, &chain_to_python(&self.chain))?)?;
        dict.set_item("chain_storage", chain_storage_to_python(py, self.chain_storage)?)?;
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

#[cfg(test)]
mod tests {
    use pyo3::{
        prepare_freethreaded_python,
        types::PyAnyMethods,
        Py, Python,
    };

    use super::*;
    use crate::{
        algorithms::mcmc::ChainStorageMode,
        core::transforms::Bounds,
        traits::StatusMessage,
        DMatrix, DVector,
    };

    fn extract_vector_like(obj: &Bound<'_, PyAny>) -> Vec<crate::Float> {
        crate::python::numeric::extract_vector(obj).unwrap()
    }

    fn extract_matrix_like(obj: &Bound<'_, PyAny>) -> Vec<Vec<crate::Float>> {
        crate::python::numeric::extract_matrix(obj).unwrap()
    }

    fn extract_tensor3_like(obj: &Bound<'_, PyAny>) -> Vec<Vec<Vec<crate::Float>>> {
        crate::python::numeric::extract_tensor3(obj).unwrap()
    }

    #[cfg(feature = "python-numpy")]
    fn ensure_numpy_initialized() {
        crate::python::numeric::ensure_numpy_test_runtime();
    }

    #[cfg(not(feature = "python-numpy"))]
    fn ensure_numpy_initialized() {}

    fn sample_summary() -> MinimizationSummary {
        MinimizationSummary {
            bounds: Some(Bounds::new_default([(Some(-1.0), Some(1.0)), (None, Some(2.0))])),
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

    fn sample_mcmc_summary() -> MCMCSummary {
        MCMCSummary {
            bounds: Some(Bounds::new_default([(Some(-1.0), Some(1.0))])),
            parameter_names: Some(vec!["theta".into()]),
            message: StatusMessage::default().set_initialized_with_message("warmup"),
            chain: vec![vec![DVector::from_vec(vec![0.0]), DVector::from_vec(vec![0.5])]],
            chain_storage: ChainStorageMode::Rolling { window: 16 },
            cost_evals: 8,
            gradient_evals: 0,
            dimension: (1, 2, 1),
        }
    }

    fn sample_simulated_annealing_summary() -> SimulatedAnnealingSummary<DVector<crate::Float>> {
        SimulatedAnnealingSummary {
            bounds: Some(Bounds::new_default([(Some(-2.0), Some(2.0)), (None, Some(3.0))])),
            message: StatusMessage::default().set_success_with_message("cooled"),
            x0: DVector::from_vec(vec![1.5, -0.5]),
            x: DVector::from_vec(vec![0.25, 1.25]),
            fx: 0.125,
            cost_evals: 42,
        }
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn minimization_summary_exports_to_python_dict() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let summary = sample_summary();
            let dict = summary.to_py_dict(py).unwrap();

            let bounds = dict.get_item("bounds").unwrap().unwrap().extract::<Vec<(Option<crate::Float>, Option<crate::Float>)>>().unwrap();
            let names = dict.get_item("parameter_names").unwrap().unwrap().extract::<Vec<String>>().unwrap();
            let x = extract_vector_like(dict.get_item("x").unwrap().unwrap().as_any());
            let covariance = extract_matrix_like(dict.get_item("covariance").unwrap().unwrap().as_any());
            let message = dict.get_item("message").unwrap().unwrap().downcast_into::<PyDict>().unwrap();

            assert_eq!(bounds, vec![(Some(-1.0), Some(1.0)), (None, Some(2.0))]);
            assert_eq!(names, vec!["alpha", "beta"]);
            assert_eq!(x, vec![0.5, 1.5]);
            assert_eq!(covariance, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
            assert_eq!(
                message.get_item("status_type").unwrap().unwrap().extract::<String>().unwrap(),
                "Success"
            );
            assert!(
                message.get_item("success").unwrap().unwrap().extract::<bool>().unwrap()
            );
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn minimization_summary_exports_to_python_class() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let summary = sample_summary();
            let wrapper = summary.to_py_class(py).unwrap();
            let wrapper = wrapper.extract::<Py<PyMinimizationSummary>>().unwrap();
            let wrapper = wrapper.bind(py).borrow();

            assert_eq!(wrapper.bounds(), Some(vec![(Some(-1.0), Some(1.0)), (None, Some(2.0))]));
            assert_eq!(wrapper.parameter_names(), Some(vec!["alpha".into(), "beta".into()]));
            assert_eq!(wrapper.status_type(), "Success");
            assert_eq!(wrapper.message_text(), "ok");
            assert!(wrapper.success());
            assert_eq!(extract_vector_like(wrapper.x0(py).unwrap().as_any()), vec![1.0, 2.0]);
            assert_eq!(extract_vector_like(wrapper.x(py).unwrap().as_any()), vec![0.5, 1.5]);
            assert_eq!(extract_vector_like(wrapper.std(py).unwrap().as_any()), vec![0.1, 0.2]);
            assert_eq!(wrapper.fx(), 1.25);
            assert_eq!(wrapper.cost_evals(), 10);
            assert_eq!(wrapper.gradient_evals(), 4);
            assert_eq!(
                extract_matrix_like(wrapper.covariance(py).unwrap().as_any()),
                vec![vec![1.0, 0.0], vec![0.0, 1.0]]
            );
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn mcmc_summary_exports_to_python_dict() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let summary = sample_mcmc_summary();
            let dict = summary.to_py_dict(py).unwrap();

            let bounds = dict
                .get_item("bounds")
                .unwrap()
                .unwrap()
                .extract::<Vec<(Option<crate::Float>, Option<crate::Float>)>>()
                .unwrap();
            let names = dict
                .get_item("parameter_names")
                .unwrap()
                .unwrap()
                .extract::<Vec<String>>()
                .unwrap();
            let chain = extract_tensor3_like(dict.get_item("chain").unwrap().unwrap().as_any());
            let chain_storage = dict
                .get_item("chain_storage")
                .unwrap()
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();
            let message = dict
                .get_item("message")
                .unwrap()
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();

            assert_eq!(bounds, vec![(Some(-1.0), Some(1.0))]);
            assert_eq!(names, vec!["theta"]);
            assert_eq!(chain, vec![vec![vec![0.0], vec![0.5]]]);
            assert_eq!(
                chain_storage
                    .get_item("mode")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "Rolling"
            );
            assert_eq!(
                chain_storage
                    .get_item("window")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                16
            );
            assert_eq!(
                message
                    .get_item("status_type")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "Initialized"
            );
            assert_eq!(
                message
                    .get_item("text")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "warmup"
            );
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn mcmc_summary_exports_to_python_class_and_helpers() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let summary = sample_mcmc_summary();
            let wrapper = summary.to_py_class(py).unwrap();
            let wrapper = wrapper.extract::<Py<PyMCMCSummary>>().unwrap();
            let wrapper = wrapper.bind(py).borrow();

            assert_eq!(wrapper.bounds(), Some(vec![(Some(-1.0), Some(1.0))]));
            assert_eq!(wrapper.parameter_names(), Some(vec!["theta".into()]));
            assert_eq!(wrapper.status_type(), "Initialized");
            assert_eq!(wrapper.message_text(), "warmup");
            assert!(!wrapper.success());
            assert_eq!(
                extract_tensor3_like(wrapper.chain(py).unwrap().as_any()),
                vec![vec![vec![0.0], vec![0.5]]]
            );
            assert_eq!(wrapper.dimension(), (1, 2, 1));
            assert_eq!(wrapper.cost_evals(), 8);
            assert_eq!(wrapper.gradient_evals(), 0);

            let retained = extract_tensor3_like(wrapper.get_chain(py, Some(1), None).unwrap().as_any());
            let flat = extract_matrix_like(wrapper.get_flat_chain(py, None, None).unwrap().as_any());
            assert_eq!(retained, vec![vec![vec![0.5]]]);
            assert_eq!(flat, vec![vec![0.0], vec![0.5]]);

            let chain_storage = wrapper.chain_storage(py).unwrap();
            assert_eq!(
                chain_storage
                    .get_item("mode")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "Rolling"
            );

            let diagnostics = wrapper.diagnostics(py, None, None).unwrap();
            let r_hat = extract_vector_like(diagnostics.get_item("r_hat").unwrap().unwrap().as_any());
            assert_eq!(r_hat.len(), 1);
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn simulated_annealing_summary_exports_to_python_dict() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let summary = sample_simulated_annealing_summary();
            let dict = summary.to_py_dict(py).unwrap();

            let bounds = dict
                .get_item("bounds")
                .unwrap()
                .unwrap()
                .extract::<Vec<(Option<crate::Float>, Option<crate::Float>)>>()
                .unwrap();
            let x0 = extract_vector_like(dict.get_item("x0").unwrap().unwrap().as_any());
            let x = extract_vector_like(dict.get_item("x").unwrap().unwrap().as_any());
            let message = dict
                .get_item("message")
                .unwrap()
                .unwrap()
                .downcast_into::<PyDict>()
                .unwrap();

            assert_eq!(bounds, vec![(Some(-2.0), Some(2.0)), (None, Some(3.0))]);
            assert_eq!(x0, vec![1.5, -0.5]);
            assert_eq!(x, vec![0.25, 1.25]);
            assert_eq!(dict.get_item("fx").unwrap().unwrap().extract::<crate::Float>().unwrap(), 0.125);
            assert_eq!(dict.get_item("cost_evals").unwrap().unwrap().extract::<usize>().unwrap(), 42);
            assert_eq!(
                message.get_item("status_type").unwrap().unwrap().extract::<String>().unwrap(),
                "Success"
            );
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn simulated_annealing_summary_exports_to_python_class() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let summary = sample_simulated_annealing_summary();
            let wrapper = summary.to_py_class(py).unwrap();
            let wrapper = wrapper.extract::<Py<PySimulatedAnnealingSummary>>().unwrap();
            let wrapper = wrapper.bind(py).borrow();

            assert_eq!(wrapper.bounds(), Some(vec![(Some(-2.0), Some(2.0)), (None, Some(3.0))]));
            assert_eq!(wrapper.status_type(), "Success");
            assert_eq!(wrapper.message_text(), "cooled");
            assert!(wrapper.success());
            assert_eq!(extract_vector_like(wrapper.x0(py).unwrap().as_any()), vec![1.5, -0.5]);
            assert_eq!(extract_vector_like(wrapper.x(py).unwrap().as_any()), vec![0.25, 1.25]);
            assert_eq!(wrapper.fx(), 0.125);
            assert_eq!(wrapper.cost_evals(), 42);
        });
    }
}
