#![allow(clippy::too_many_arguments)] // Python keyword-only constructors mirror algorithm options.

use pyo3::{exceptions::PyValueError, prelude::*, types::PyAnyMethods};

use crate::{
    algorithms::{
        gradient::{
            AdamConfig, ConjugateGradientConfig, ConjugateGradientUpdate, LBFGSBConfig,
            LBFGSBErrorMode, TrustRegionConfig, TrustRegionSubproblem,
        },
        gradient_free::{
            CMAESConfig, DifferentialEvolutionConfig, NelderMeadConfig, SimplexExpansionMethod,
            SimulatedAnnealingConfig,
        },
        line_search::{HagerZhangLineSearch, MoreThuenteLineSearch, StrongWolfeLineSearch},
        mcmc::{AIESConfig, AIESMove, ChainStorageMode, ESSConfig, ESSMove},
        particles::{PSOConfig, SwarmTopology, SwarmUpdateMethod, SwarmVelocityInitializer},
    },
    traits::{PeriodicTransform, TransformChain},
    Bounds, NalgebraProvider, ScaleTransform, Transform,
};

fn finite_positive(value: f64, name: &str) -> PyResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be finite and positive"
        )))
    }
}

#[derive(Clone, Debug)]
enum TransformSpec {
    Bounds(Vec<(f64, f64)>),
    Scale(Vec<f64>),
    Periodic(Vec<Option<(f64, f64)>>),
    Chain(Vec<Self>),
}

impl TransformSpec {
    fn build(&self) -> crate::error::GaneshResult<Box<dyn Transform<f64, NalgebraProvider>>> {
        match self {
            Self::Bounds(bounds) => Ok(Box::new(Bounds::new(bounds.clone())?)),
            Self::Scale(scales) => Ok(Box::new(ScaleTransform::from_parameter_scales(
                scales.clone(),
            )?)),
            Self::Periodic(intervals) => Ok(Box::new(PeriodicTransform::new(intervals.clone())?)),
            Self::Chain(items) => {
                let mut items = items.iter();
                let Some(first) = items.next() else {
                    return Err(crate::error::GaneshError::ConfigError(
                        "a transform chain requires at least one transform".into(),
                    ));
                };
                let mut chain = first.build()?;
                for item in items {
                    chain = Box::new(TransformChain::new(chain, item.build()?));
                }
                Ok(chain)
            }
        }
    }
}

fn extract_transform(value: &Bound<'_, PyAny>) -> PyResult<TransformSpec> {
    if let Ok(value) = value.extract::<PyRef<'_, PyBounds>>() {
        return Ok(value.spec.clone());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyScaleTransform>>() {
        return Ok(value.spec.clone());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyPeriodicTransform>>() {
        return Ok(value.spec.clone());
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyTransformChain>>() {
        return Ok(value.spec.clone());
    }
    Err(PyValueError::new_err(
        "expected a Ganesh bounds, scale, periodic, or transform-chain object",
    ))
}

/// Smooth parameter bounds.
#[pyclass(name = "Bounds", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyBounds {
    spec: TransformSpec,
}

#[pymethods]
impl PyBounds {
    #[new]
    fn new(bounds: Vec<(f64, f64)>) -> PyResult<Self> {
        Bounds::<f64, NalgebraProvider>::new(bounds.clone()).map_err(super::ganesh_error)?;
        Ok(Self {
            spec: TransformSpec::Bounds(bounds),
        })
    }
}

/// Per-parameter characteristic scales.
#[pyclass(name = "ScaleTransform", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyScaleTransform {
    spec: TransformSpec,
}

#[pymethods]
impl PyScaleTransform {
    #[new]
    fn new(scales: Vec<f64>) -> PyResult<Self> {
        ScaleTransform::<f64, NalgebraProvider>::from_parameter_scales(scales.clone())
            .map_err(super::ganesh_error)?;
        Ok(Self {
            spec: TransformSpec::Scale(scales),
        })
    }
}

/// Per-parameter optional periodic intervals.
#[pyclass(name = "PeriodicTransform", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPeriodicTransform {
    spec: TransformSpec,
}

#[pymethods]
impl PyPeriodicTransform {
    #[new]
    fn new(intervals: Vec<Option<(f64, f64)>>) -> PyResult<Self> {
        PeriodicTransform::<f64, NalgebraProvider>::new(intervals.clone())
            .map_err(super::ganesh_error)?;
        Ok(Self {
            spec: TransformSpec::Periodic(intervals),
        })
    }
}

/// A transform sequence applied in the supplied order.
#[pyclass(name = "TransformChain", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTransformChain {
    spec: TransformSpec,
}

#[pymethods]
impl PyTransformChain {
    #[new]
    fn new(transforms: &Bound<'_, PyAny>) -> PyResult<Self> {
        let specs = transforms
            .try_iter()?
            .map(|item| item.and_then(|item| extract_transform(&item)))
            .collect::<PyResult<Vec<_>>>()?;
        if specs.is_empty() {
            return Err(PyValueError::new_err(
                "a transform chain requires at least one transform",
            ));
        }
        let spec = TransformSpec::Chain(specs);
        spec.build().map_err(super::ganesh_error)?;
        Ok(Self { spec })
    }
}

fn parse_transform(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<TransformSpec>> {
    value.map(extract_transform).transpose()
}

#[derive(Clone, Debug)]
enum LineSearchSpec {
    MoreThuente {
        c1: f64,
        c2: f64,
        max_iterations: usize,
        max_zoom: usize,
    },
    HagerZhang {
        delta: f64,
        sigma: f64,
        epsilon: f64,
        theta: f64,
        gamma: f64,
        max_iterations: usize,
        max_bisects: usize,
    },
}

impl LineSearchSpec {
    fn build(&self) -> crate::error::GaneshResult<StrongWolfeLineSearch> {
        Ok(match *self {
            Self::MoreThuente {
                c1,
                c2,
                max_iterations,
                max_zoom,
            } => MoreThuenteLineSearch::default()
                .with_c1_c2(c1, c2)?
                .with_max_iterations(max_iterations)
                .with_max_zoom(max_zoom)
                .into(),
            Self::HagerZhang {
                delta,
                sigma,
                epsilon,
                theta,
                gamma,
                max_iterations,
                max_bisects,
            } => HagerZhangLineSearch::default()
                .with_delta_sigma(delta, sigma)?
                .with_epsilon(epsilon)?
                .with_theta(theta)?
                .with_gamma(gamma)?
                .with_max_iterations(max_iterations)
                .with_max_bisects(max_bisects)
                .into(),
        })
    }
}

/// Moré–Thuente strong-Wolfe line-search settings.
#[pyclass(name = "MoreThuenteLineSearch", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMoreThuenteLineSearch {
    spec: LineSearchSpec,
}

#[pymethods]
impl PyMoreThuenteLineSearch {
    #[new]
    #[pyo3(signature = (*, c1=1e-4, c2=0.9, max_iterations=20, max_zoom=20))]
    fn new(c1: f64, c2: f64, max_iterations: usize, max_zoom: usize) -> PyResult<Self> {
        let spec = LineSearchSpec::MoreThuente {
            c1,
            c2,
            max_iterations,
            max_zoom,
        };
        spec.build().map_err(super::ganesh_error)?;
        Ok(Self { spec })
    }
}

/// Hager–Zhang strong-Wolfe line-search settings.
#[pyclass(name = "HagerZhangLineSearch", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyHagerZhangLineSearch {
    spec: LineSearchSpec,
}

#[pymethods]
impl PyHagerZhangLineSearch {
    #[new]
    #[pyo3(signature = (*, delta=0.1, sigma=0.9, epsilon=1e-6, theta=0.5, gamma=0.66, max_iterations=100, max_bisects=50))]
    fn new(
        delta: f64,
        sigma: f64,
        epsilon: f64,
        theta: f64,
        gamma: f64,
        max_iterations: usize,
        max_bisects: usize,
    ) -> PyResult<Self> {
        let spec = LineSearchSpec::HagerZhang {
            delta,
            sigma,
            epsilon,
            theta,
            gamma,
            max_iterations,
            max_bisects,
        };
        spec.build().map_err(super::ganesh_error)?;
        Ok(Self { spec })
    }
}

fn parse_line_search(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<LineSearchSpec>> {
    let Some(value) = value else { return Ok(None) };
    if let Ok(value) = value.extract::<PyRef<'_, PyMoreThuenteLineSearch>>() {
        return Ok(Some(value.spec.clone()));
    }
    if let Ok(value) = value.extract::<PyRef<'_, PyHagerZhangLineSearch>>() {
        return Ok(Some(value.spec.clone()));
    }
    Err(PyValueError::new_err(
        "expected MoreThuenteLineSearch or HagerZhangLineSearch",
    ))
}

macro_rules! apply_common {
    ($config:expr, $names:expr, $transform:expr) => {{
        let mut config = $config;
        if let Some(names) = &$names {
            config =
                crate::traits::SupportsParameterNames::with_parameter_names(config, names.clone());
        }
        if let Some(transform) = &$transform {
            config = config.with_transform(transform.build()?);
        }
        config
    }};
}

/// Adam configuration.
#[pyclass(name = "AdamConfig", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyAdamConfig {
    alpha: f64,
    beta_1: f64,
    beta_2: f64,
    epsilon: f64,
    parameter_names: Option<Vec<String>>,
    transform: Option<TransformSpec>,
}

#[pymethods]
impl PyAdamConfig {
    #[new]
    #[pyo3(signature = (*, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, parameter_names=None, transform=None))]
    fn new(
        alpha: f64,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        parameter_names: Option<Vec<String>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let value = Self {
            alpha,
            beta_1,
            beta_2,
            epsilon,
            parameter_names,
            transform: parse_transform(transform)?,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}

impl PyAdamConfig {
    /// Convert to the default Rust Adam configuration.
    pub fn to_rust(&self) -> crate::error::GaneshResult<AdamConfig> {
        let config = AdamConfig::default()
            .with_alpha(self.alpha)?
            .with_beta_1(self.beta_1)?
            .with_beta_2(self.beta_2)?
            .with_epsilon(self.epsilon)?;
        Ok(apply_common!(config, self.parameter_names, self.transform))
    }
}

/// Nonlinear conjugate-gradient configuration.
#[pyclass(name = "ConjugateGradientConfig", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyConjugateGradientConfig {
    update: String,
    line_search: Option<LineSearchSpec>,
    parameter_names: Option<Vec<String>>,
    transform: Option<TransformSpec>,
}

#[pymethods]
impl PyConjugateGradientConfig {
    #[new]
    #[pyo3(signature = (*, update="polak_ribiere_plus", line_search=None, parameter_names=None, transform=None))]
    fn new(
        update: &str,
        line_search: Option<&Bound<'_, PyAny>>,
        parameter_names: Option<Vec<String>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let value = Self {
            update: update.to_owned(),
            line_search: parse_line_search(line_search)?,
            parameter_names,
            transform: parse_transform(transform)?,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}

impl PyConjugateGradientConfig {
    /// Convert to the default Rust conjugate-gradient configuration.
    pub fn to_rust(&self) -> crate::error::GaneshResult<ConjugateGradientConfig> {
        let update = match self.update.as_str() {
            "fletcher_reeves" => ConjugateGradientUpdate::FletcherReeves,
            "polak_ribiere_plus" => ConjugateGradientUpdate::PolakRibierePlus,
            "hestenes_stiefel_plus" => ConjugateGradientUpdate::HestenesStiefelPlus,
            "dai_yuan" => ConjugateGradientUpdate::DaiYuan,
            "hager_zhang" => ConjugateGradientUpdate::HagerZhang,
            other => {
                return Err(crate::error::GaneshError::ConfigError(format!(
                    "unknown conjugate-gradient update `{other}`"
                )))
            }
        };
        let mut config = ConjugateGradientConfig::default().with_update(update);
        if let Some(line_search) = &self.line_search {
            config = config.with_line_search(line_search.build()?);
        }
        Ok(apply_common!(config, self.parameter_names, self.transform))
    }
}

/// L-BFGS-B configuration.
#[pyclass(name = "LBFGSBConfig", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyLBFGSBConfig {
    history_size: usize,
    max_step: f64,
    error_mode: String,
    line_search: Option<LineSearchSpec>,
    bounds: Option<Vec<(f64, f64)>>,
    parameter_names: Option<Vec<String>>,
    transform: Option<TransformSpec>,
}

#[pymethods]
impl PyLBFGSBConfig {
    #[new]
    #[pyo3(signature = (*, history_size=10, max_step=1e8, error_mode="exact_hessian", line_search=None, bounds=None, parameter_names=None, transform=None))]
    fn new(
        history_size: usize,
        max_step: f64,
        error_mode: &str,
        line_search: Option<&Bound<'_, PyAny>>,
        bounds: Option<Vec<(f64, f64)>>,
        parameter_names: Option<Vec<String>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let value = Self {
            history_size,
            max_step,
            error_mode: error_mode.to_owned(),
            line_search: parse_line_search(line_search)?,
            bounds,
            parameter_names,
            transform: parse_transform(transform)?,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}

impl PyLBFGSBConfig {
    /// Convert to the default Rust L-BFGS-B configuration.
    pub fn to_rust(&self) -> crate::error::GaneshResult<LBFGSBConfig> {
        let mode = match self.error_mode.as_str() {
            "exact_hessian" => LBFGSBErrorMode::ExactHessian,
            "skip" => LBFGSBErrorMode::Skip,
            other => {
                return Err(crate::error::GaneshError::ConfigError(format!(
                    "unknown L-BFGS-B error mode `{other}`"
                )))
            }
        };
        let mut config = LBFGSBConfig::default()
            .with_memory_limit(self.history_size)?
            .with_max_step(self.max_step)?
            .with_error_mode(mode);
        if let Some(line_search) = &self.line_search {
            config = config.with_line_search(line_search.build()?);
        }
        if let Some(bounds) = &self.bounds {
            config = config.with_bounds(bounds.clone())?;
        }
        if let Some(names) = &self.parameter_names {
            config =
                crate::traits::SupportsParameterNames::with_parameter_names(config, names.clone());
        }
        if let Some(transform) = &self.transform {
            config = config.with_transform(transform.build()?)?;
        }
        Ok(config)
    }
}

/// Trust-region configuration.
#[pyclass(name = "TrustRegionConfig", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTrustRegionConfig {
    subproblem: String,
    initial_radius: f64,
    max_radius: f64,
    eta: f64,
    parameter_names: Option<Vec<String>>,
    transform: Option<TransformSpec>,
}

#[pymethods]
impl PyTrustRegionConfig {
    #[new]
    #[pyo3(signature = (*, subproblem="dogleg", initial_radius=1.0, max_radius=1000.0, eta=1e-4, parameter_names=None, transform=None))]
    fn new(
        subproblem: &str,
        initial_radius: f64,
        max_radius: f64,
        eta: f64,
        parameter_names: Option<Vec<String>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let value = Self {
            subproblem: subproblem.to_owned(),
            initial_radius,
            max_radius,
            eta,
            parameter_names,
            transform: parse_transform(transform)?,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}

impl PyTrustRegionConfig {
    /// Convert to the default Rust trust-region configuration.
    pub fn to_rust(&self) -> crate::error::GaneshResult<TrustRegionConfig> {
        let subproblem = match self.subproblem.as_str() {
            "dogleg" => TrustRegionSubproblem::Dogleg,
            "cauchy_point" => TrustRegionSubproblem::CauchyPoint,
            other => {
                return Err(crate::error::GaneshError::ConfigError(format!(
                    "unknown trust-region subproblem `{other}`"
                )))
            }
        };
        let config = TrustRegionConfig::default()
            .with_subproblem(subproblem)
            .with_initial_radius(self.initial_radius)?
            .with_max_radius(self.max_radius)?
            .with_eta(self.eta)?;
        Ok(apply_common!(config, self.parameter_names, self.transform))
    }
}

/// CMA-ES configuration.
#[pyclass(name = "CMAESConfig", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyCMAESConfig {
    population_size: usize,
    initial_sigma: f64,
    parameter_names: Option<Vec<String>>,
    transform: Option<TransformSpec>,
}
#[pymethods]
impl PyCMAESConfig {
    #[new]
    #[pyo3(signature = (*, population_size=0, initial_sigma=0.5, parameter_names=None, transform=None))]
    fn new(
        population_size: usize,
        initial_sigma: f64,
        parameter_names: Option<Vec<String>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let value = Self {
            population_size,
            initial_sigma,
            parameter_names,
            transform: parse_transform(transform)?,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}
impl PyCMAESConfig {
    /// Convert to Rust.
    pub fn to_rust(&self) -> crate::error::GaneshResult<CMAESConfig> {
        let mut config = CMAESConfig::default().with_initial_sigma(self.initial_sigma)?;
        if self.population_size != 0 {
            config = config.with_population_size(self.population_size)?;
        }
        Ok(apply_common!(config, self.parameter_names, self.transform))
    }
}

/// Differential-evolution configuration.
#[pyclass(name = "DifferentialEvolutionConfig", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDifferentialEvolutionConfig {
    population_size: usize,
    differential_weight: f64,
    crossover_probability: f64,
    initial_scale: f64,
    parameter_names: Option<Vec<String>>,
    transform: Option<TransformSpec>,
}
#[pymethods]
impl PyDifferentialEvolutionConfig {
    #[new]
    #[pyo3(signature = (*, population_size=0, differential_weight=0.8, crossover_probability=0.9, initial_scale=1.0, parameter_names=None, transform=None))]
    fn new(
        population_size: usize,
        differential_weight: f64,
        crossover_probability: f64,
        initial_scale: f64,
        parameter_names: Option<Vec<String>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let value = Self {
            population_size,
            differential_weight,
            crossover_probability,
            initial_scale,
            parameter_names,
            transform: parse_transform(transform)?,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}
impl PyDifferentialEvolutionConfig {
    /// Convert to Rust.
    pub fn to_rust(&self) -> crate::error::GaneshResult<DifferentialEvolutionConfig> {
        let mut config = DifferentialEvolutionConfig::default()
            .with_differential_weight(self.differential_weight)?
            .with_crossover_probability(self.crossover_probability)?
            .with_initial_scale(self.initial_scale)?;
        if self.population_size != 0 {
            config = config.with_population_size(self.population_size)?;
        }
        Ok(apply_common!(config, self.parameter_names, self.transform))
    }
}

/// Nelder-Mead configuration.
#[pyclass(name = "NelderMeadConfig", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyNelderMeadConfig {
    reflection: f64,
    expansion: f64,
    contraction: f64,
    shrink: f64,
    initial_step: f64,
    initial_zero_step: f64,
    expansion_method: String,
    initial_simplex: Option<Vec<Vec<f64>>>,
    parameter_names: Option<Vec<String>>,
    transform: Option<TransformSpec>,
}
#[pymethods]
impl PyNelderMeadConfig {
    #[new]
    #[pyo3(signature = (*, reflection=1.0, expansion=2.0, contraction=0.5, shrink=0.5, initial_step=0.05, initial_zero_step=0.00025, expansion_method="greedy_minimization", initial_simplex=None, parameter_names=None, transform=None))]
    fn new(
        reflection: f64,
        expansion: f64,
        contraction: f64,
        shrink: f64,
        initial_step: f64,
        initial_zero_step: f64,
        expansion_method: &str,
        initial_simplex: Option<Vec<Vec<f64>>>,
        parameter_names: Option<Vec<String>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let value = Self {
            reflection,
            expansion,
            contraction,
            shrink,
            initial_step,
            initial_zero_step,
            expansion_method: expansion_method.to_owned(),
            initial_simplex,
            parameter_names,
            transform: parse_transform(transform)?,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}
impl PyNelderMeadConfig {
    /// Convert to Rust.
    pub fn to_rust(&self) -> crate::error::GaneshResult<NelderMeadConfig> {
        let method = match self.expansion_method.as_str() {
            "greedy_minimization" => SimplexExpansionMethod::GreedyMinimization,
            "greedy_expansion" => SimplexExpansionMethod::GreedyExpansion,
            other => {
                return Err(crate::error::GaneshError::ConfigError(format!(
                    "unknown simplex expansion method `{other}`"
                )))
            }
        };
        let mut config = NelderMeadConfig::default()
            .with_alpha_beta(self.reflection, self.expansion)?
            .with_gamma(self.contraction)?
            .with_delta(self.shrink)?
            .with_initial_step(self.initial_step)?
            .with_initial_zero_step(self.initial_zero_step)?
            .with_expansion_method(method);
        if let Some(simplex) = &self.initial_simplex {
            config = config.with_initial_simplex(
                simplex
                    .iter()
                    .cloned()
                    .map(crate::Vector::from_vec)
                    .collect(),
            )?;
        }
        Ok(apply_common!(config, self.parameter_names, self.transform))
    }
}

/// Simulated-annealing configuration.
#[pyclass(name = "SimulatedAnnealingConfig", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PySimulatedAnnealingConfig {
    initial_temperature: f64,
    cooling_rate: f64,
    proposal_scale: f64,
    parameter_names: Option<Vec<String>>,
    transform: Option<TransformSpec>,
}
#[pymethods]
impl PySimulatedAnnealingConfig {
    #[new]
    #[pyo3(signature = (*, initial_temperature=1.0, cooling_rate=0.999, proposal_scale=0.1, parameter_names=None, transform=None))]
    fn new(
        initial_temperature: f64,
        cooling_rate: f64,
        proposal_scale: f64,
        parameter_names: Option<Vec<String>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let value = Self {
            initial_temperature,
            cooling_rate,
            proposal_scale,
            parameter_names,
            transform: parse_transform(transform)?,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}
impl PySimulatedAnnealingConfig {
    /// Convert to Rust.
    pub fn to_rust(&self) -> crate::error::GaneshResult<SimulatedAnnealingConfig> {
        let config = SimulatedAnnealingConfig::new(self.initial_temperature, self.cooling_rate)?
            .with_proposal_scale(self.proposal_scale)?;
        Ok(apply_common!(config, self.parameter_names, self.transform))
    }
}

/// Weighted affine-invariant ensemble move.
#[pyclass(name = "AIESMove", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyAIESMove {
    kind: String,
    scale: f64,
    weight: f64,
}

#[pymethods]
impl PyAIESMove {
    #[staticmethod]
    #[pyo3(signature = (*, scale=2.0, weight=1.0))]
    fn stretch(scale: f64, weight: f64) -> PyResult<Self> {
        let value = Self {
            kind: "stretch".into(),
            scale,
            weight,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
    #[staticmethod]
    #[pyo3(signature = (*, weight=1.0))]
    fn walk(weight: f64) -> PyResult<Self> {
        let value = Self {
            kind: "walk".into(),
            scale: 2.0,
            weight,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}

impl PyAIESMove {
    /// Convert to a weighted Rust AIES move.
    pub fn to_rust(&self) -> crate::error::GaneshResult<(AIESMove, f64)> {
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(crate::error::GaneshError::ConfigError(
                "move weight must be finite and non-negative".into(),
            ));
        }
        match self.kind.as_str() {
            "stretch" => AIESMove::custom_stretch(self.scale, self.weight),
            "walk" => Ok(AIESMove::walk(self.weight)),
            _ => unreachable!(),
        }
    }
}

/// Weighted ensemble-slice move.
#[pyclass(name = "ESSMove", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyESSMove {
    kind: String,
    scale: f64,
    rescale_covariance: f64,
    components: usize,
    weight: f64,
}

#[pymethods]
impl PyESSMove {
    #[staticmethod]
    #[pyo3(signature = (*, weight=1.0))]
    fn differential(weight: f64) -> PyResult<Self> {
        Self::make("differential", 1.0, 1.0, 1, weight)
    }
    #[staticmethod]
    #[pyo3(signature = (*, weight=1.0))]
    fn gaussian(weight: f64) -> PyResult<Self> {
        Self::make("gaussian", 1.0, 1.0, 1, weight)
    }
    #[staticmethod]
    #[pyo3(signature = (*, scale=1.0, rescale_covariance=0.001, components=5, weight=1.0))]
    fn global(
        scale: f64,
        rescale_covariance: f64,
        components: usize,
        weight: f64,
    ) -> PyResult<Self> {
        Self::make("global", scale, rescale_covariance, components, weight)
    }
}

impl PyESSMove {
    fn make(
        kind: &str,
        scale: f64,
        rescale_covariance: f64,
        components: usize,
        weight: f64,
    ) -> PyResult<Self> {
        let value = Self {
            kind: kind.into(),
            scale,
            rescale_covariance,
            components,
            weight,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
    /// Convert to a weighted Rust ESS move.
    pub fn to_rust(&self) -> crate::error::GaneshResult<(ESSMove, f64)> {
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(crate::error::GaneshError::ConfigError(
                "move weight must be finite and non-negative".into(),
            ));
        }
        match self.kind.as_str() {
            "differential" => Ok(ESSMove::differential(self.weight)),
            "gaussian" => Ok(ESSMove::gaussian(self.weight)),
            "global" => ESSMove::custom_global(
                self.weight,
                Some(self.scale),
                Some(self.rescale_covariance),
                Some(self.components),
            ),
            _ => unreachable!(),
        }
    }
}

/// Chain retention settings shared by Ganesh's MCMC algorithms.
#[pyclass(name = "ChainStorage", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyChainStorage {
    kind: String,
    value: Option<usize>,
    max_samples: Option<usize>,
}

#[pymethods]
impl PyChainStorage {
    #[staticmethod]
    fn full() -> Self {
        Self {
            kind: "full".into(),
            value: None,
            max_samples: None,
        }
    }
    #[staticmethod]
    fn rolling(window: usize) -> PyResult<Self> {
        let value = Self {
            kind: "rolling".into(),
            value: Some(window),
            max_samples: None,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
    #[staticmethod]
    #[pyo3(signature = (keep_every, *, max_samples=None))]
    fn sampled(keep_every: usize, max_samples: Option<usize>) -> PyResult<Self> {
        let value = Self {
            kind: "sampled".into(),
            value: Some(keep_every),
            max_samples,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}

impl PyChainStorage {
    /// Convert to Rust chain-storage settings.
    pub fn to_rust(&self) -> crate::error::GaneshResult<ChainStorageMode> {
        let positive = |name: &str, value: Option<usize>| {
            value.filter(|value| *value > 0).ok_or_else(|| {
                crate::error::GaneshError::ConfigError(format!("{name} must be positive"))
            })
        };
        match self.kind.as_str() {
            "full" => Ok(ChainStorageMode::Full),
            "rolling" => Ok(ChainStorageMode::Rolling {
                window: positive("window", self.value)?,
            }),
            "sampled" => Ok(ChainStorageMode::Sampled {
                keep_every: positive("keep_every", self.value)?,
                max_samples: self
                    .max_samples
                    .map(|value| positive("max_samples", Some(value)))
                    .transpose()?,
            }),
            _ => unreachable!(),
        }
    }
}

/// Affine-invariant ensemble-sampler configuration.
#[pyclass(name = "AIESConfig", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyAIESConfig {
    stretch_scale: f64,
    moves: Option<Vec<PyAIESMove>>,
    chain_storage: PyChainStorage,
    parameter_names: Option<Vec<String>>,
    transform: Option<TransformSpec>,
}
#[pymethods]
impl PyAIESConfig {
    #[new]
    #[pyo3(signature = (*, stretch_scale=2.0, moves=None, chain_storage=None, parameter_names=None, transform=None))]
    fn new(
        stretch_scale: f64,
        moves: Option<Vec<PyAIESMove>>,
        chain_storage: Option<PyChainStorage>,
        parameter_names: Option<Vec<String>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        finite_positive(stretch_scale, "stretch_scale")?;
        let value = Self {
            stretch_scale,
            moves,
            chain_storage: chain_storage.unwrap_or_else(PyChainStorage::full),
            parameter_names,
            transform: parse_transform(transform)?,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}
impl PyAIESConfig {
    /// Convert to Rust.
    pub fn to_rust(&self) -> crate::error::GaneshResult<AIESConfig> {
        let mut config = AIESConfig::default()
            .with_stretch_scale(self.stretch_scale)?
            .with_chain_storage(self.chain_storage.to_rust()?);
        if let Some(moves) = &self.moves {
            config = config.with_moves(
                moves
                    .iter()
                    .map(PyAIESMove::to_rust)
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
        }
        Ok(apply_common!(config, self.parameter_names, self.transform))
    }
}

/// Ensemble-slice-sampler configuration.
#[pyclass(name = "ESSConfig", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyESSConfig {
    bracket_width: f64,
    max_shrink_steps: usize,
    adaptive_steps: usize,
    direction_scale: f64,
    moves: Option<Vec<PyESSMove>>,
    chain_storage: PyChainStorage,
    parameter_names: Option<Vec<String>>,
    transform: Option<TransformSpec>,
}
#[pymethods]
impl PyESSConfig {
    #[new]
    #[pyo3(signature = (*, bracket_width=1.0, max_shrink_steps=1000, adaptive_steps=0, direction_scale=1.0, moves=None, chain_storage=None, parameter_names=None, transform=None))]
    fn new(
        bracket_width: f64,
        max_shrink_steps: usize,
        adaptive_steps: usize,
        direction_scale: f64,
        moves: Option<Vec<PyESSMove>>,
        chain_storage: Option<PyChainStorage>,
        parameter_names: Option<Vec<String>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let value = Self {
            bracket_width,
            max_shrink_steps,
            adaptive_steps,
            direction_scale,
            moves,
            chain_storage: chain_storage.unwrap_or_else(PyChainStorage::full),
            parameter_names,
            transform: parse_transform(transform)?,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}
impl PyESSConfig {
    /// Convert to Rust.
    pub fn to_rust(&self) -> crate::error::GaneshResult<ESSConfig> {
        let mut config = ESSConfig::default()
            .with_bracket_width(self.bracket_width)?
            .with_max_shrink_steps(self.max_shrink_steps)?
            .with_adaptive_steps(self.adaptive_steps)
            .with_direction_scale(self.direction_scale)?
            .with_chain_storage(self.chain_storage.to_rust()?);
        if let Some(moves) = &self.moves {
            config = config.with_moves(
                moves
                    .iter()
                    .map(PyESSMove::to_rust)
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
        }
        Ok(apply_common!(config, self.parameter_names, self.transform))
    }
}

/// Particle-swarm configuration.
#[pyclass(name = "PSOConfig", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyPSOConfig {
    particles: usize,
    inertia: f64,
    cognitive: f64,
    social: f64,
    initial_scale: f64,
    position_bounds: Option<Vec<(f64, f64)>>,
    topology: String,
    ring_radius: usize,
    update_method: String,
    velocity_scale: Option<f64>,
    parameter_names: Option<Vec<String>>,
    transform: Option<TransformSpec>,
}
#[pymethods]
impl PyPSOConfig {
    #[new]
    #[pyo3(signature = (*, particles=0, inertia=0.7298, cognitive=1.49618, social=1.49618, initial_scale=1.0, position_bounds=None, topology="global", ring_radius=1, update_method="synchronous", velocity_scale=None, parameter_names=None, transform=None))]
    fn new(
        particles: usize,
        inertia: f64,
        cognitive: f64,
        social: f64,
        initial_scale: f64,
        position_bounds: Option<Vec<(f64, f64)>>,
        topology: &str,
        ring_radius: usize,
        update_method: &str,
        velocity_scale: Option<f64>,
        parameter_names: Option<Vec<String>>,
        transform: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let value = Self {
            particles,
            inertia,
            cognitive,
            social,
            initial_scale,
            position_bounds,
            topology: topology.to_owned(),
            ring_radius,
            update_method: update_method.to_owned(),
            velocity_scale,
            parameter_names,
            transform: parse_transform(transform)?,
        };
        value.to_rust().map_err(super::ganesh_error)?;
        Ok(value)
    }
}

impl PyPSOConfig {
    /// Convert to Rust.
    pub fn to_rust(&self) -> crate::error::GaneshResult<PSOConfig> {
        let topology = match self.topology.as_str() {
            "global" => SwarmTopology::Global,
            "ring" => SwarmTopology::Ring {
                radius: self.ring_radius,
            },
            other => {
                return Err(crate::error::GaneshError::ConfigError(format!(
                    "unknown topology `{other}`"
                )))
            }
        };
        let update = match self.update_method.as_str() {
            "synchronous" => SwarmUpdateMethod::Synchronous,
            "asynchronous" => SwarmUpdateMethod::Asynchronous,
            other => {
                return Err(crate::error::GaneshError::ConfigError(format!(
                    "unknown update method `{other}`"
                )))
            }
        };
        let velocity = self
            .velocity_scale
            .map_or(SwarmVelocityInitializer::Zero, |scale| {
                SwarmVelocityInitializer::Uniform { scale }
            });
        let mut config = PSOConfig::default()
            .with_omega(self.inertia)?
            .with_c1(self.cognitive)?
            .with_c2(self.social)?
            .with_initial_scale(self.initial_scale)?
            .with_topology(topology)
            .with_update_method(update)
            .with_velocity_initializer(velocity);
        if let Some(bounds) = &self.position_bounds {
            config = config.with_uniform_initialization(bounds.clone())?;
        }
        if self.particles != 0 {
            config = config.with_particles(self.particles)?;
        }
        Ok(apply_common!(config, self.parameter_names, self.transform))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_search_specs_validate_and_convert() {
        PyMoreThuenteLineSearch::new(1e-4, 0.8, 25, 20).unwrap();
        assert!(PyMoreThuenteLineSearch::new(0.9, 0.1, 25, 20).is_err());
        PyHagerZhangLineSearch::new(0.1, 0.9, 1e-6, 0.5, 0.66, 100, 50).unwrap();
    }

    #[test]
    fn lbfgsb_config_accepts_transforms_with_native_bounds() {
        PyLBFGSBConfig {
            history_size: 10,
            max_step: 1e8,
            error_mode: "exact_hessian".into(),
            line_search: None,
            bounds: Some(vec![(0.0, 2.0), (-1.0, 1.0)]),
            parameter_names: None,
            transform: Some(TransformSpec::Scale(vec![2.0, 4.0])),
        }
        .to_rust()
        .unwrap();
    }

    #[test]
    fn typed_mcmc_components_build_rust_configs() {
        let moves = vec![
            PyAIESMove::stretch(2.5, 0.8).unwrap(),
            PyAIESMove::walk(0.2).unwrap(),
        ];
        let storage = PyChainStorage::sampled(10, Some(500)).unwrap();
        PyAIESConfig::new(2.0, Some(moves), Some(storage), None, None)
            .unwrap()
            .to_rust()
            .unwrap();

        let moves = vec![
            PyESSMove::differential(0.5).unwrap(),
            PyESSMove::global(1.0, 0.001, 5, 0.5).unwrap(),
        ];
        PyESSConfig::new(1.0, 1000, 0, 1.0, Some(moves), None, None, None)
            .unwrap()
            .to_rust()
            .unwrap();
    }

    #[test]
    fn invalid_component_settings_fail_eagerly() {
        assert!(PyAIESMove::stretch(1.0, 1.0).is_err());
        assert!(PyESSMove::global(1.0, 0.001, 1, 1.0).is_err());
        assert!(PyChainStorage::rolling(0).is_err());
    }
}
