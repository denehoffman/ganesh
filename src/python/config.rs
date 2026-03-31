//! Python-facing config conversion traits for downstream wrapper crates.

use pyo3::{
    pyclass, pymethods,
    types::PyAnyMethods,
    Bound, PyAny, PyRef, PyResult, Python,
};

use crate::{
    algorithms::{
        gradient::LBFGSBConfig,
        gradient_free::{CMAESConfig, DifferentialEvolutionConfig, NelderMeadConfig},
        mcmc::{AIESConfig, ESSConfig},
        particles::{PSOConfig, Swarm, SwarmPositionInitializer},
    },
    error::GaneshError,
    python::numeric::{extract_matrix, extract_vector, matrix_to_python, vector_to_python},
    traits::{Bound as GaneshBound, SupportsBounds, SupportsParameterNames},
    DVector, Float,
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

fn vectors_to_dvectors(vectors: &[Vec<Float>]) -> Vec<DVector<Float>> {
    vectors.iter().cloned().map(DVector::from_vec).collect()
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
    pub fn new(x0: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            x0: extract_vector(x0)?,
            memory_limit: 10,
            bounds: None,
            parameter_names: None,
        })
    }

    /// Get the starting position of the optimizer.
    #[getter]
    pub fn x0<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, &self.x0)
    }

    /// Set the starting position of the optimizer.
    #[setter]
    pub fn set_x0(&mut self, x0: &Bound<'_, PyAny>) -> PyResult<()> {
        self.x0 = extract_vector(x0)?;
        Ok(())
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

/// Python-facing typed wrapper for [`NelderMeadConfig`].
#[pyclass(module = "ganesh", name = "NelderMeadConfig")]
#[derive(Clone)]
pub struct PyNelderMeadConfig {
    x0: Vec<Float>,
    bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
    parameter_names: Option<Vec<String>>,
}

#[allow(missing_docs)]
#[pymethods]
impl PyNelderMeadConfig {
    #[new]
    pub fn new(x0: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            x0: extract_vector(x0)?,
            bounds: None,
            parameter_names: None,
        })
    }

    #[getter]
    pub fn x0<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, &self.x0)
    }

    #[setter]
    pub fn set_x0(&mut self, x0: &Bound<'_, PyAny>) -> PyResult<()> {
        self.x0 = extract_vector(x0)?;
        Ok(())
    }

    #[getter]
    pub fn bounds(&self) -> Option<Vec<(Option<Float>, Option<Float>)>> {
        self.bounds.clone()
    }

    #[setter]
    pub fn set_bounds(&mut self, bounds: Option<Vec<(Option<Float>, Option<Float>)>>) {
        self.bounds = bounds;
    }

    #[getter]
    pub fn parameter_names(&self) -> Option<Vec<String>> {
        self.parameter_names.clone()
    }

    #[setter]
    pub fn set_parameter_names(&mut self, parameter_names: Option<Vec<String>>) {
        self.parameter_names = parameter_names;
    }
}

impl TryFrom<&PyNelderMeadConfig> for NelderMeadConfig {
    type Error = GaneshError;

    fn try_from(config: &PyNelderMeadConfig) -> Result<Self, Self::Error> {
        let native = NelderMeadConfig::new(&config.x0);
        let native = apply_python_bounds(native, &config.bounds);
        Ok(apply_python_parameter_names(native, &config.parameter_names))
    }
}

impl<'py> FromPyConfig<'py> for NelderMeadConfig {
    fn from_py_config(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let config = obj.extract::<PyRef<'py, PyNelderMeadConfig>>()?;
        NelderMeadConfig::try_from(&*config).map_err(Into::into)
    }
}

/// Python-facing typed wrapper for [`PSOConfig`].
#[pyclass(module = "ganesh", name = "PSOConfig")]
#[derive(Clone)]
pub struct PyPSOConfig {
    positions: Vec<Vec<Float>>,
    bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
    parameter_names: Option<Vec<String>>,
}

#[allow(missing_docs)]
#[pymethods]
impl PyPSOConfig {
    #[new]
    pub fn new(positions: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            positions: extract_matrix(positions)?,
            bounds: None,
            parameter_names: None,
        })
    }

    #[getter]
    pub fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        matrix_to_python(py, &self.positions)
    }

    #[setter]
    pub fn set_positions(&mut self, positions: &Bound<'_, PyAny>) -> PyResult<()> {
        self.positions = extract_matrix(positions)?;
        Ok(())
    }

    #[getter]
    pub fn bounds(&self) -> Option<Vec<(Option<Float>, Option<Float>)>> {
        self.bounds.clone()
    }

    #[setter]
    pub fn set_bounds(&mut self, bounds: Option<Vec<(Option<Float>, Option<Float>)>>) {
        self.bounds = bounds;
    }

    #[getter]
    pub fn parameter_names(&self) -> Option<Vec<String>> {
        self.parameter_names.clone()
    }

    #[setter]
    pub fn set_parameter_names(&mut self, parameter_names: Option<Vec<String>>) {
        self.parameter_names = parameter_names;
    }
}

impl TryFrom<&PyPSOConfig> for PSOConfig {
    type Error = GaneshError;

    fn try_from(config: &PyPSOConfig) -> Result<Self, Self::Error> {
        let swarm = Swarm::new(SwarmPositionInitializer::Custom(vectors_to_dvectors(&config.positions)));
        let native = PSOConfig::new(swarm);
        let native = apply_python_bounds(native, &config.bounds);
        Ok(apply_python_parameter_names(native, &config.parameter_names))
    }
}

impl<'py> FromPyConfig<'py> for PSOConfig {
    fn from_py_config(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let config = obj.extract::<PyRef<'py, PyPSOConfig>>()?;
        PSOConfig::try_from(&*config).map_err(Into::into)
    }
}

/// Python-facing typed wrapper for [`AIESConfig`].
#[pyclass(module = "ganesh", name = "AIESConfig")]
#[derive(Clone)]
pub struct PyAIESConfig {
    walkers: Vec<Vec<Float>>,
    parameter_names: Option<Vec<String>>,
}

#[allow(missing_docs)]
#[pymethods]
impl PyAIESConfig {
    #[new]
    pub fn new(walkers: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            walkers: extract_matrix(walkers)?,
            parameter_names: None,
        })
    }

    #[getter]
    pub fn walkers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        matrix_to_python(py, &self.walkers)
    }

    #[setter]
    pub fn set_walkers(&mut self, walkers: &Bound<'_, PyAny>) -> PyResult<()> {
        self.walkers = extract_matrix(walkers)?;
        Ok(())
    }

    #[getter]
    pub fn parameter_names(&self) -> Option<Vec<String>> {
        self.parameter_names.clone()
    }

    #[setter]
    pub fn set_parameter_names(&mut self, parameter_names: Option<Vec<String>>) {
        self.parameter_names = parameter_names;
    }
}

impl TryFrom<&PyAIESConfig> for AIESConfig {
    type Error = GaneshError;

    fn try_from(config: &PyAIESConfig) -> Result<Self, Self::Error> {
        let native = AIESConfig::new(vectors_to_dvectors(&config.walkers))?;
        Ok(apply_python_parameter_names(native, &config.parameter_names))
    }
}

impl<'py> FromPyConfig<'py> for AIESConfig {
    fn from_py_config(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let config = obj.extract::<PyRef<'py, PyAIESConfig>>()?;
        AIESConfig::try_from(&*config).map_err(Into::into)
    }
}

/// Python-facing typed wrapper for [`ESSConfig`].
#[pyclass(module = "ganesh", name = "ESSConfig")]
#[derive(Clone)]
pub struct PyESSConfig {
    walkers: Vec<Vec<Float>>,
    parameter_names: Option<Vec<String>>,
    n_adaptive: usize,
    max_steps: usize,
    mu: Float,
}

#[allow(missing_docs)]
#[pymethods]
impl PyESSConfig {
    #[new]
    pub fn new(walkers: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            walkers: extract_matrix(walkers)?,
            parameter_names: None,
            n_adaptive: 0,
            max_steps: 10000,
            mu: 1.0,
        })
    }

    #[getter]
    pub fn walkers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        matrix_to_python(py, &self.walkers)
    }

    #[setter]
    pub fn set_walkers(&mut self, walkers: &Bound<'_, PyAny>) -> PyResult<()> {
        self.walkers = extract_matrix(walkers)?;
        Ok(())
    }

    #[getter]
    pub fn parameter_names(&self) -> Option<Vec<String>> {
        self.parameter_names.clone()
    }

    #[setter]
    pub fn set_parameter_names(&mut self, parameter_names: Option<Vec<String>>) {
        self.parameter_names = parameter_names;
    }

    #[getter]
    pub const fn n_adaptive(&self) -> usize {
        self.n_adaptive
    }

    #[setter]
    pub fn set_n_adaptive(&mut self, n_adaptive: usize) {
        self.n_adaptive = n_adaptive;
    }

    #[getter]
    pub const fn max_steps(&self) -> usize {
        self.max_steps
    }

    #[setter]
    pub fn set_max_steps(&mut self, max_steps: usize) {
        self.max_steps = max_steps;
    }

    #[getter]
    pub const fn mu(&self) -> Float {
        self.mu
    }

    #[setter]
    pub fn set_mu(&mut self, mu: Float) {
        self.mu = mu;
    }
}

impl TryFrom<&PyESSConfig> for ESSConfig {
    type Error = GaneshError;

    fn try_from(config: &PyESSConfig) -> Result<Self, Self::Error> {
        let native = ESSConfig::new(vectors_to_dvectors(&config.walkers))?
            .with_n_adaptive(config.n_adaptive)
            .with_max_steps(config.max_steps)
            .with_mu(config.mu)?;
        Ok(apply_python_parameter_names(native, &config.parameter_names))
    }
}

impl<'py> FromPyConfig<'py> for ESSConfig {
    fn from_py_config(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let config = obj.extract::<PyRef<'py, PyESSConfig>>()?;
        ESSConfig::try_from(&*config).map_err(Into::into)
    }
}

/// Python-facing typed wrapper for [`DifferentialEvolutionConfig`].
#[pyclass(module = "ganesh", name = "DifferentialEvolutionConfig")]
#[derive(Clone)]
pub struct PyDifferentialEvolutionConfig {
    x0: Vec<Float>,
    population_size: Option<usize>,
    differential_weight: Float,
    crossover_probability: Float,
    initial_scale: Float,
    bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
    parameter_names: Option<Vec<String>>,
}

#[allow(missing_docs)]
#[pymethods]
impl PyDifferentialEvolutionConfig {
    #[new]
    pub fn new(x0: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            x0: extract_vector(x0)?,
            population_size: None,
            differential_weight: 0.8,
            crossover_probability: 0.9,
            initial_scale: 1.0,
            bounds: None,
            parameter_names: None,
        })
    }

    #[getter]
    pub fn x0<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, &self.x0)
    }

    #[setter]
    pub fn set_x0(&mut self, x0: &Bound<'_, PyAny>) -> PyResult<()> {
        self.x0 = extract_vector(x0)?;
        Ok(())
    }

    #[getter]
    pub const fn population_size(&self) -> Option<usize> {
        self.population_size
    }

    #[setter]
    pub fn set_population_size(&mut self, population_size: Option<usize>) {
        self.population_size = population_size;
    }

    #[getter]
    pub const fn differential_weight(&self) -> Float {
        self.differential_weight
    }

    #[setter]
    pub fn set_differential_weight(&mut self, differential_weight: Float) {
        self.differential_weight = differential_weight;
    }

    #[getter]
    pub const fn crossover_probability(&self) -> Float {
        self.crossover_probability
    }

    #[setter]
    pub fn set_crossover_probability(&mut self, crossover_probability: Float) {
        self.crossover_probability = crossover_probability;
    }

    #[getter]
    pub const fn initial_scale(&self) -> Float {
        self.initial_scale
    }

    #[setter]
    pub fn set_initial_scale(&mut self, initial_scale: Float) {
        self.initial_scale = initial_scale;
    }

    #[getter]
    pub fn bounds(&self) -> Option<Vec<(Option<Float>, Option<Float>)>> {
        self.bounds.clone()
    }

    #[setter]
    pub fn set_bounds(&mut self, bounds: Option<Vec<(Option<Float>, Option<Float>)>>) {
        self.bounds = bounds;
    }

    #[getter]
    pub fn parameter_names(&self) -> Option<Vec<String>> {
        self.parameter_names.clone()
    }

    #[setter]
    pub fn set_parameter_names(&mut self, parameter_names: Option<Vec<String>>) {
        self.parameter_names = parameter_names;
    }
}

impl TryFrom<&PyDifferentialEvolutionConfig> for DifferentialEvolutionConfig {
    type Error = GaneshError;

    fn try_from(config: &PyDifferentialEvolutionConfig) -> Result<Self, Self::Error> {
        let mut native = DifferentialEvolutionConfig::new(&config.x0)?;
        if let Some(population_size) = config.population_size {
            native = native.with_population_size(population_size)?;
        }
        native = native
            .with_differential_weight(config.differential_weight)?
            .with_crossover_probability(config.crossover_probability)?
            .with_initial_scale(config.initial_scale)?;
        let native = apply_python_bounds(native, &config.bounds);
        Ok(apply_python_parameter_names(native, &config.parameter_names))
    }
}

impl<'py> FromPyConfig<'py> for DifferentialEvolutionConfig {
    fn from_py_config(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let config = obj.extract::<PyRef<'py, PyDifferentialEvolutionConfig>>()?;
        DifferentialEvolutionConfig::try_from(&*config).map_err(Into::into)
    }
}

/// Python-facing typed wrapper for [`CMAESConfig`].
#[pyclass(module = "ganesh", name = "CMAESConfig")]
#[derive(Clone)]
pub struct PyCMAESConfig {
    x0: Vec<Float>,
    sigma: Float,
    population_size: Option<usize>,
    bounds: Option<Vec<(Option<Float>, Option<Float>)>>,
    parameter_names: Option<Vec<String>>,
}

#[allow(missing_docs)]
#[pymethods]
impl PyCMAESConfig {
    #[new]
    pub fn new(x0: &Bound<'_, PyAny>, sigma: Float) -> PyResult<Self> {
        Ok(Self {
            x0: extract_vector(x0)?,
            sigma,
            population_size: None,
            bounds: None,
            parameter_names: None,
        })
    }

    #[getter]
    pub fn x0<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, &self.x0)
    }

    #[setter]
    pub fn set_x0(&mut self, x0: &Bound<'_, PyAny>) -> PyResult<()> {
        self.x0 = extract_vector(x0)?;
        Ok(())
    }

    #[getter]
    pub const fn sigma(&self) -> Float {
        self.sigma
    }

    #[setter]
    pub fn set_sigma(&mut self, sigma: Float) {
        self.sigma = sigma;
    }

    #[getter]
    pub const fn population_size(&self) -> Option<usize> {
        self.population_size
    }

    #[setter]
    pub fn set_population_size(&mut self, population_size: Option<usize>) {
        self.population_size = population_size;
    }

    #[getter]
    pub fn bounds(&self) -> Option<Vec<(Option<Float>, Option<Float>)>> {
        self.bounds.clone()
    }

    #[setter]
    pub fn set_bounds(&mut self, bounds: Option<Vec<(Option<Float>, Option<Float>)>>) {
        self.bounds = bounds;
    }

    #[getter]
    pub fn parameter_names(&self) -> Option<Vec<String>> {
        self.parameter_names.clone()
    }

    #[setter]
    pub fn set_parameter_names(&mut self, parameter_names: Option<Vec<String>>) {
        self.parameter_names = parameter_names;
    }
}

impl TryFrom<&PyCMAESConfig> for CMAESConfig {
    type Error = GaneshError;

    fn try_from(config: &PyCMAESConfig) -> Result<Self, Self::Error> {
        let mut native = CMAESConfig::new(&config.x0, config.sigma)?;
        if let Some(population_size) = config.population_size {
            native = native.with_population_size(population_size)?;
        }
        let native = apply_python_bounds(native, &config.bounds);
        Ok(apply_python_parameter_names(native, &config.parameter_names))
    }
}

impl<'py> FromPyConfig<'py> for CMAESConfig {
    fn from_py_config(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let config = obj.extract::<PyRef<'py, PyCMAESConfig>>()?;
        CMAESConfig::try_from(&*config).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{prepare_freethreaded_python, types::PyList, Py, Python};

    use super::*;
    use crate::{
        algorithms::{
            gradient::LBFGSB,
            gradient_free::{CMAES, DifferentialEvolution, NelderMead},
            mcmc::{AIES, ESS},
            particles::PSO,
        },
        core::{Callbacks, MaxSteps, MinimizationSummary},
        traits::{Algorithm, CostFunction, Gradient, LogDensity},
        DVector,
    };
    use std::convert::Infallible;

    fn py_vector<'py>(py: Python<'py>, values: &[Float]) -> Bound<'py, PyAny> {
        PyList::new(py, values).unwrap().into_any()
    }

    fn py_matrix<'py>(py: Python<'py>, values: &[Vec<Float>]) -> Bound<'py, PyAny> {
        PyList::new(py, values).unwrap().into_any()
    }

    #[cfg(feature = "python-numpy")]
    fn ensure_numpy_initialized() {
        crate::python::numeric::ensure_numpy_test_runtime();
    }

    #[cfg(not(feature = "python-numpy"))]
    fn ensure_numpy_initialized() {}

    struct Quadratic;
    struct GaussianLogDensity;

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

    impl LogDensity for GaussianLogDensity {
        fn log_density(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(-0.5 * x.dot(x))
        }
    }

    fn run_summary(config: LBFGSBConfig) -> MinimizationSummary {
        let mut solver = LBFGSB::default();
        solver.process_default(&Quadratic, &(), config).unwrap()
    }

    fn run_nm_summary(config: NelderMeadConfig) -> MinimizationSummary {
        let mut solver = NelderMead::default();
        solver
            .process(
                &Quadratic,
                &(),
                config,
                Callbacks::empty().with_terminator(MaxSteps(8)),
            )
            .unwrap()
    }

    fn run_pso_summary(config: PSOConfig) -> MinimizationSummary {
        let mut solver = PSO::default();
        solver
            .process(
                &Quadratic,
                &(),
                config,
                Callbacks::empty().with_terminator(MaxSteps(2)),
            )
            .unwrap()
    }

    fn run_de_summary(config: DifferentialEvolutionConfig) -> MinimizationSummary {
        let mut solver = DifferentialEvolution::default();
        solver
            .process(
                &Quadratic,
                &(),
                config,
                Callbacks::empty().with_terminator(MaxSteps(8)),
            )
            .unwrap()
    }

    fn run_cmaes_summary(config: CMAESConfig) -> MinimizationSummary {
        let mut solver = CMAES::default();
        solver
            .process(
                &Quadratic,
                &(),
                config,
                Callbacks::empty().with_terminator(MaxSteps(8)),
            )
            .unwrap()
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn python_lbfgsb_config_converts_to_native_config() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let mut wrapper = PyLBFGSBConfig::new(&py_vector(py, &[2.0, -1.0])).unwrap();
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
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn python_lbfgsb_config_invalid_memory_limit_maps_to_python_config_error() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let mut wrapper = PyLBFGSBConfig::new(&py_vector(py, &[1.0, 1.0])).unwrap();
            wrapper.set_memory_limit(0);

            let wrapper = Py::new(py, wrapper).unwrap();
            let err = LBFGSBConfig::from_py_config(wrapper.bind(py).as_any()).err().unwrap();
            assert!(err.is_instance_of::<crate::python::GaneshConfigError>(py));
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn python_lbfgsb_config_accessors_roundtrip_python_facing_state() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            ensure_numpy_initialized();
            let mut wrapper = PyLBFGSBConfig::new(&py_vector(py, &[1.0, 2.0])).unwrap();
            assert_eq!(
                crate::python::numeric::extract_vector(wrapper.x0(py).unwrap().as_any()).unwrap(),
                vec![1.0, 2.0]
            );
            assert_eq!(wrapper.memory_limit(), 10);
            assert_eq!(wrapper.bounds(), None);
            assert_eq!(wrapper.parameter_names(), None);

            wrapper.set_bounds(Some(vec![(Some(0.0), Some(3.0)), (Some(-1.0), None)]));
            wrapper.set_parameter_names(Some(vec!["x".into(), "y".into()]));

            assert_eq!(wrapper.bounds(), Some(vec![(Some(0.0), Some(3.0)), (Some(-1.0), None)]));
            assert_eq!(wrapper.parameter_names(), Some(vec!["x".into(), "y".into()]));
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn python_nelder_mead_config_converts_to_native_config() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let mut wrapper = PyNelderMeadConfig::new(&py_vector(py, &[2.0, 2.0])).unwrap();
            wrapper.set_bounds(Some(vec![(Some(-3.0), Some(3.0)), (None, Some(2.0))]));
            wrapper.set_parameter_names(Some(vec!["alpha".into(), "beta".into()]));

            let wrapper = Py::new(py, wrapper).unwrap();
            let config = NelderMeadConfig::from_py_config(wrapper.bind(py).as_any()).unwrap();
            let summary = run_nm_summary(config);

            assert_eq!(summary.parameter_names.as_deref(), Some(&["alpha".into(), "beta".into()][..]));
            assert_eq!(
                summary.bounds.as_ref().unwrap()[0].0.as_options(),
                (Some(-3.0), Some(3.0))
            );
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn python_pso_config_converts_to_native_config() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let mut wrapper = PyPSOConfig::new(&py_matrix(
                py,
                &[vec![2.0, 2.0], vec![1.0, 1.0], vec![0.5, 0.5]],
            ))
            .unwrap();
            wrapper.set_bounds(Some(vec![(Some(-3.0), Some(3.0)), (Some(-3.0), Some(3.0))]));
            wrapper.set_parameter_names(Some(vec!["alpha".into(), "beta".into()]));

            let wrapper = Py::new(py, wrapper).unwrap();
            let config = PSOConfig::from_py_config(wrapper.bind(py).as_any()).unwrap();
            let summary = run_pso_summary(config);

            assert_eq!(summary.parameter_names.as_deref(), Some(&["alpha".into(), "beta".into()][..]));
            assert_eq!(
                summary.bounds.as_ref().unwrap()[0].0.as_options(),
                (Some(-3.0), Some(3.0))
            );
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn python_aies_config_converts_to_native_config() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let mut wrapper = PyAIESConfig::new(&py_matrix(py, &[
                vec![0.0, 0.0],
                vec![0.1, 0.0],
                vec![0.0, 0.1],
                vec![0.1, 0.1],
            ]))
            .unwrap();
            wrapper.set_parameter_names(Some(vec!["alpha".into(), "beta".into()]));

            let wrapper = Py::new(py, wrapper).unwrap();
            let config = AIESConfig::from_py_config(wrapper.bind(py).as_any()).unwrap();
            let mut solver = AIES::default();
            let summary = solver
                .process(
                    &GaussianLogDensity,
                    &(),
                    config,
                    AIES::default_callbacks().with_terminator(MaxSteps(1)),
                )
                .unwrap();

            assert_eq!(summary.parameter_names.as_deref(), Some(&["alpha".into(), "beta".into()][..]));
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn python_ess_config_converts_to_native_config() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let mut wrapper = PyESSConfig::new(&py_matrix(py, &[
                vec![0.0, 0.0],
                vec![0.1, 0.0],
                vec![0.0, 0.1],
            ]))
            .unwrap();
            wrapper.set_parameter_names(Some(vec!["alpha".into(), "beta".into()]));
            wrapper.set_n_adaptive(2);
            wrapper.set_max_steps(32);
            wrapper.set_mu(0.75);

            let wrapper = Py::new(py, wrapper).unwrap();
            let config = ESSConfig::from_py_config(wrapper.bind(py).as_any()).unwrap();
            let mut solver = ESS::default();
            let summary = solver
                .process(
                    &GaussianLogDensity,
                    &(),
                    config,
                    ESS::default_callbacks().with_terminator(MaxSteps(1)),
                )
                .unwrap();

            assert_eq!(summary.parameter_names.as_deref(), Some(&["alpha".into(), "beta".into()][..]));
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn python_ess_config_invalid_mu_maps_to_python_config_error() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let mut wrapper = PyESSConfig::new(&py_matrix(py, &[
                vec![0.0, 0.0],
                vec![0.1, 0.0],
                vec![0.0, 0.1],
            ]))
            .unwrap();
            wrapper.set_mu(0.0);

            let wrapper = Py::new(py, wrapper).unwrap();
            let err = ESSConfig::from_py_config(wrapper.bind(py).as_any()).err().unwrap();
            assert!(err.is_instance_of::<crate::python::GaneshConfigError>(py));
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn python_de_config_converts_to_native_config() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let mut wrapper =
                PyDifferentialEvolutionConfig::new(&py_vector(py, &[2.0, -2.0])).unwrap();
            wrapper.set_population_size(Some(8));
            wrapper.set_differential_weight(0.7);
            wrapper.set_crossover_probability(0.85);
            wrapper.set_initial_scale(0.5);
            wrapper.set_bounds(Some(vec![(Some(-3.0), Some(3.0)), (Some(-3.0), Some(3.0))]));
            wrapper.set_parameter_names(Some(vec!["alpha".into(), "beta".into()]));

            let wrapper = Py::new(py, wrapper).unwrap();
            let config =
                DifferentialEvolutionConfig::from_py_config(wrapper.bind(py).as_any()).unwrap();
            let summary = run_de_summary(config);

            assert_eq!(
                summary.parameter_names.as_deref(),
                Some(&["alpha".into(), "beta".into()][..])
            );
            assert_eq!(
                summary.bounds.as_ref().unwrap()[0].0.as_options(),
                (Some(-3.0), Some(3.0))
            );
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn python_cmaes_config_converts_to_native_config() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let mut wrapper = PyCMAESConfig::new(&py_vector(py, &[2.0, -2.0]), 0.8).unwrap();
            wrapper.set_population_size(Some(6));
            wrapper.set_bounds(Some(vec![(Some(-3.0), Some(3.0)), (Some(-3.0), Some(3.0))]));
            wrapper.set_parameter_names(Some(vec!["alpha".into(), "beta".into()]));

            let wrapper = Py::new(py, wrapper).unwrap();
            let config = CMAESConfig::from_py_config(wrapper.bind(py).as_any()).unwrap();
            let summary = run_cmaes_summary(config);

            assert_eq!(
                summary.parameter_names.as_deref(),
                Some(&["alpha".into(), "beta".into()][..])
            );
            assert_eq!(
                summary.bounds.as_ref().unwrap()[0].0.as_options(),
                (Some(-3.0), Some(3.0))
            );
        });
    }

    #[test]
    #[cfg_attr(feature = "python-numpy", ignore = "NumPy runtime is unavailable in this test environment")]
    fn python_cmaes_invalid_sigma_maps_to_python_config_error() {
        prepare_freethreaded_python();
        ensure_numpy_initialized();
        Python::with_gil(|py| {
            let wrapper = Py::new(py, PyCMAESConfig::new(&py_vector(py, &[1.0, 1.0]), 0.0).unwrap()).unwrap();
            let err = CMAESConfig::from_py_config(wrapper.bind(py).as_any()).err().unwrap();
            assert!(err.is_instance_of::<crate::python::GaneshConfigError>(py));
        });
    }
}
