//! Python-facing summary export traits for downstream wrapper crates.

use pyo3::{
    pyclass, pymethods, Py,
    types::{PyDict, PyDictMethods},
    Bound, PyAny, PyResult, Python,
};

use crate::core::MinimizationSummary;

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
    pub fn x0(&self) -> Vec<crate::Float> {
        self.summary.x0.as_slice().to_vec()
    }

    /// Get the final parameters.
    #[getter]
    pub fn x(&self) -> Vec<crate::Float> {
        self.summary.x.as_slice().to_vec()
    }

    /// Get the parameter standard deviations.
    #[getter]
    pub fn std(&self) -> Vec<crate::Float> {
        self.summary.std.as_slice().to_vec()
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
    pub fn covariance(&self) -> Vec<Vec<crate::Float>> {
        self.summary
            .covariance
            .row_iter()
            .map(|row| row.iter().copied().collect::<Vec<_>>())
            .collect()
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

impl IntoPySummary for MinimizationSummary {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("bounds", self.bounds.as_ref().map(bounds_to_python))?;
        dict.set_item("parameter_names", self.parameter_names.clone())?;

        let message = PyDict::new(py);
        message.set_item("status_type", self.message.status_type.to_string())?;
        message.set_item("text", self.message.text.clone())?;
        message.set_item("success", self.message.success())?;
        dict.set_item("message", message)?;

        dict.set_item("x0", self.x0.as_slice().to_vec())?;
        dict.set_item("x", self.x.as_slice().to_vec())?;
        dict.set_item("std", self.std.as_slice().to_vec())?;
        dict.set_item("fx", self.fx)?;
        dict.set_item("cost_evals", self.cost_evals)?;
        dict.set_item("gradient_evals", self.gradient_evals)?;
        dict.set_item("covariance", self.covariance.row_iter().map(|row| row.iter().copied().collect::<Vec<_>>()).collect::<Vec<_>>())?;
        Ok(dict)
    }

    fn to_py_class<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let wrapper = Py::new(py, PyMinimizationSummary::from(self.clone()))?;
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
        core::transforms::Bounds,
        traits::StatusMessage,
        DMatrix, DVector,
    };

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

    #[test]
    fn minimization_summary_exports_to_python_dict() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            let summary = sample_summary();
            let dict = summary.to_py_dict(py).unwrap();

            let bounds = dict.get_item("bounds").unwrap().unwrap().extract::<Vec<(Option<crate::Float>, Option<crate::Float>)>>().unwrap();
            let names = dict.get_item("parameter_names").unwrap().unwrap().extract::<Vec<String>>().unwrap();
            let x = dict.get_item("x").unwrap().unwrap().extract::<Vec<crate::Float>>().unwrap();
            let covariance = dict.get_item("covariance").unwrap().unwrap().extract::<Vec<Vec<crate::Float>>>().unwrap();
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
    fn minimization_summary_exports_to_python_class() {
        prepare_freethreaded_python();
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
            assert_eq!(wrapper.x0(), vec![1.0, 2.0]);
            assert_eq!(wrapper.x(), vec![0.5, 1.5]);
            assert_eq!(wrapper.std(), vec![0.1, 0.2]);
            assert_eq!(wrapper.fx(), 1.25);
            assert_eq!(wrapper.cost_evals(), 10);
            assert_eq!(wrapper.gradient_evals(), 4);
            assert_eq!(wrapper.covariance(), vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        });
    }
}
