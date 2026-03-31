//! Python-facing config conversion traits for downstream wrapper crates.

use pyo3::{
    pyclass, pymethods,
    types::PyAnyMethods,
    Bound, PyAny, PyRef, PyResult,
};

use crate::{
    algorithms::gradient::LBFGSBConfig,
    error::GaneshError,
    traits::{Bound as GaneshBound, SupportsBounds, SupportsParameterNames},
    Float,
};

/// Convert a Python-facing wrapper object into a native `ganesh` config.
///
/// This trait is intended for typed `#[pyclass]` wrapper objects used by downstream crates. The
/// goal is to keep config parsing typed rather than dictionary-based.
pub trait FromPyConfig<'py>: Sized {
    /// Construct `Self` from a Python-facing wrapper object.
    fn from_py_config(obj: &Bound<'py, PyAny>) -> PyResult<Self>;
}

fn apply_python_bounds<C>(
    mut config: C,
    bounds: &Option<Vec<(Option<Float>, Option<Float>)>>,
) -> C
where
    C: SupportsBounds,
{
    if let Some(bounds) = bounds {
        config = config.with_bounds(bounds.iter().copied().map(GaneshBound::from));
    }
    config
}

fn apply_python_parameter_names<C>(mut config: C, parameter_names: &Option<Vec<String>>) -> C
where
    C: SupportsParameterNames,
{
    if let Some(parameter_names) = parameter_names {
        config = config.with_parameter_names(parameter_names.clone());
    }
    config
}

/// Python-facing typed wrapper for [`LBFGSBConfig`].
#[pyclass(module = "ganesh", name = "LBFGSBConfig")]
#[derive(Clone)]
pub struct PyLBFGSBConfig {
    x0: Vec<Float>,
    memory_limit: usize,
    bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
    parameter_names: Option<Vec<String>>,
}

#[pymethods]
impl PyLBFGSBConfig {
    /// Create a new Python-facing L-BFGS-B config wrapper.
    #[new]
    pub fn new(x0: Vec<Float>) -> Self {
        Self {
            x0,
            memory_limit: 10,
            bounds: None,
            parameter_names: None,
        }
    }

    /// Get the starting position of the optimizer.
    #[getter]
    pub fn x0(&self) -> Vec<Float> {
        self.x0.clone()
    }

    /// Set the starting position of the optimizer.
    #[setter]
    pub fn set_x0(&mut self, x0: Vec<Float>) {
        self.x0 = x0;
    }

    /// Get the number of stored L-BFGS-B correction steps.
    #[getter]
    pub const fn memory_limit(&self) -> usize {
        self.memory_limit
    }

    /// Set the number of stored L-BFGS-B correction steps.
    #[setter]
    pub fn set_memory_limit(&mut self, memory_limit: usize) {
        self.memory_limit = memory_limit;
    }

    /// Get the optional parameter bounds.
    ///
    /// Each bound is represented as `(lower, upper)`, where `None` means unbounded on that side.
    #[getter]
    pub fn bounds(&self) -> Option<Vec<(Option<Float>, Option<Float>)>> {
        self.bounds.clone()
    }

    /// Set the optional parameter bounds.
    ///
    /// Each bound is represented as `(lower, upper)`, where `None` means unbounded on that side.
    #[setter]
    pub fn set_bounds(&mut self, bounds: Option<Vec<(Option<Float>, Option<Float>)>>) {
        self.bounds = bounds;
    }

    /// Get optional parameter names to propagate into the summary.
    #[getter]
    pub fn parameter_names(&self) -> Option<Vec<String>> {
        self.parameter_names.clone()
    }

    /// Set optional parameter names to propagate into the summary.
    #[setter]
    pub fn set_parameter_names(&mut self, parameter_names: Option<Vec<String>>) {
        self.parameter_names = parameter_names;
    }
}

impl TryFrom<&PyLBFGSBConfig> for LBFGSBConfig {
    type Error = GaneshError;

    fn try_from(config: &PyLBFGSBConfig) -> Result<Self, Self::Error> {
        let native = LBFGSBConfig::new(&config.x0).with_memory_limit(config.memory_limit)?;
        let native = apply_python_bounds(native, &config.bounds);
        Ok(apply_python_parameter_names(native, &config.parameter_names))
    }
}

impl<'py> FromPyConfig<'py> for LBFGSBConfig {
    fn from_py_config(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let config = obj.extract::<PyRef<'py, PyLBFGSBConfig>>()?;
        LBFGSBConfig::try_from(&*config).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{prepare_freethreaded_python, Py, Python};

    use super::*;
    use crate::{
        algorithms::gradient::LBFGSB,
        core::MinimizationSummary,
        traits::{Algorithm, CostFunction, Gradient},
        DVector,
    };
    use std::convert::Infallible;

    struct Quadratic;

    impl CostFunction for Quadratic {
        fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(x.dot(x))
        }
    }

    impl Gradient for Quadratic {
        fn gradient(&self, x: &DVector<Float>, _args: &()) -> Result<DVector<Float>, Infallible> {
            Ok(x.scale(2.0))
        }
    }

    fn run_summary(config: LBFGSBConfig) -> MinimizationSummary {
        let mut solver = LBFGSB::default();
        solver.process_default(&Quadratic, &(), config).unwrap()
    }

    #[test]
    fn python_lbfgsb_config_converts_to_native_config() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut wrapper = PyLBFGSBConfig::new(vec![2.0, -1.0]);
            wrapper.set_memory_limit(5);
            wrapper.set_bounds(Some(vec![(Some(-3.0), Some(3.0)), (None, Some(2.0))]));
            wrapper.set_parameter_names(Some(vec!["alpha".into(), "beta".into()]));

            let wrapper = Py::new(py, wrapper).unwrap();
            let config = LBFGSBConfig::from_py_config(wrapper.bind(py).as_any()).unwrap();
            let summary = run_summary(config);

            assert!(summary.message.success());
            assert_eq!(
                summary.bounds.as_ref().unwrap()[0].0.as_options(),
                (Some(-3.0), Some(3.0))
            );
            assert_eq!(
                summary.bounds.as_ref().unwrap()[1].0.as_options(),
                (None, Some(2.0))
            );
            assert_eq!(summary.parameter_names.as_deref(), Some(&["alpha".into(), "beta".into()][..]));
        });
    }

    #[test]
    fn python_lbfgsb_config_invalid_memory_limit_maps_to_python_config_error() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut wrapper = PyLBFGSBConfig::new(vec![1.0, 1.0]);
            wrapper.set_memory_limit(0);

            let wrapper = Py::new(py, wrapper).unwrap();
            let err = LBFGSBConfig::from_py_config(wrapper.bind(py).as_any()).err().unwrap();
            assert!(err.is_instance_of::<crate::python::GaneshConfigError>(py));
        });
    }

    #[test]
    fn python_lbfgsb_config_accessors_roundtrip_python_facing_state() {
        let mut wrapper = PyLBFGSBConfig::new(vec![1.0, 2.0]);
        assert_eq!(wrapper.x0(), vec![1.0, 2.0]);
        assert_eq!(wrapper.memory_limit(), 10);
        assert_eq!(wrapper.bounds(), None);
        assert_eq!(wrapper.parameter_names(), None);

        wrapper.set_bounds(Some(vec![(Some(0.0), Some(3.0)), (Some(-1.0), None)]));
        wrapper.set_parameter_names(Some(vec!["x".into(), "y".into()]));

        assert_eq!(wrapper.bounds(), Some(vec![(Some(0.0), Some(3.0)), (Some(-1.0), None)]));
        assert_eq!(wrapper.parameter_names(), Some(vec!["x".into(), "y".into()]));
    }
}
