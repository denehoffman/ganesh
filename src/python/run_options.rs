//! Python-facing run-option extraction support for built-in terminators and observers.
#![allow(missing_docs)]

use pyo3::{Borrowed, FromPyObject, PyAny};

use crate::{
    algorithms::{
        gradient::{
            adam::AdamEMATerminator,
            lbfgsb::{LBFGSBFTerminator, LBFGSBGTerminator, LBFGSBInfNormGTerminator},
            Adam, AdamConfig, ConjugateGradient, ConjugateGradientConfig,
            ConjugateGradientGTerminator, GradientStatus, LBFGSBConfig, TrustRegion,
            TrustRegionConfig, TrustRegionGTerminator, LBFGSB,
        },
        gradient_free::{
            nelder_mead::{NelderMeadFTerminator, NelderMeadXTerminator},
            simulated_annealing::SimulatedAnnealingTerminator,
            CMAESConditionCovTerminator, CMAESConfig, CMAESEqualFunValuesTerminator,
            CMAESNoEffectAxisTerminator, CMAESNoEffectCoordTerminator, CMAESSigmaTerminator,
            CMAESStagnationTerminator, CMAESTolFunTerminator, CMAESTolXTerminator,
            CMAESTolXUpTerminator, DifferentialEvolution, DifferentialEvolutionConfig,
            GradientFreeStatus, NelderMead, NelderMeadConfig, SimulatedAnnealing,
            SimulatedAnnealingConfig, SimulatedAnnealingGenerator, SimulatedAnnealingStatus, CMAES,
        },
        mcmc::{AIESConfig, AutocorrelationTerminator, ESSConfig, EnsembleStatus, AIES, ESS},
        particles::{PSOConfig, SwarmStatus, PSO},
    },
    core::{Callbacks, DebugObserver, MaxSteps, ProgressObserver},
    error::GaneshError,
    python::extract::{
        extract_optional_field, extract_optional_one_or_many_field, resolve_protocol,
    },
    traits::{Algorithm, CostFunction, Gradient, LogDensity, Status},
    Float,
};
use serde::{Deserialize, Serialize};

fn apply_common_callbacks<A, P, S, U, E, C>(
    mut callbacks: Callbacks<A, P, S, U, E, C>,
    max_steps: Option<usize>,
    debug: bool,
    progress_every: Option<usize>,
) -> Callbacks<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status + std::fmt::Debug,
    U: std::fmt::Debug,
{
    if let Some(max_steps) = max_steps {
        callbacks = callbacks.with_terminator(MaxSteps(max_steps));
    }
    if debug {
        callbacks = callbacks.with_observer(DebugObserver);
    }
    if let Some(progress_every) = progress_every {
        callbacks = callbacks.with_observer(ProgressObserver::new(progress_every));
    }
    callbacks
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyAutocorrelationTerminator {
    pub n_check: usize,
    pub n_taus_threshold: usize,
    pub dtau_threshold: Float,
    pub discard: Float,
    pub terminate: bool,
    pub sokal_window: Option<Float>,
    pub verbose: bool,
}

impl Default for PyAutocorrelationTerminator {
    fn default() -> Self {
        Self {
            n_check: 50,
            n_taus_threshold: 50,
            dtau_threshold: 0.01,
            discard: 0.5,
            terminate: true,
            sokal_window: None,
            verbose: false,
        }
    }
}

impl PyAutocorrelationTerminator {
    fn to_terminator(&self) -> AutocorrelationTerminator {
        let mut terminator = AutocorrelationTerminator::default()
            .with_n_check(self.n_check)
            .with_n_taus_threshold(self.n_taus_threshold)
            .with_dtau_threshold(self.dtau_threshold)
            .with_discard(self.discard)
            .with_terminate(self.terminate)
            .with_verbose(self.verbose);
        if let Some(sokal_window) = self.sokal_window {
            terminator = terminator.with_sokal_window(sokal_window);
        }
        terminator
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyAutocorrelationTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "n_check")? {
            config.n_check = value;
        }
        if let Some(value) = extract_optional_field(&obj, "n_taus_threshold")? {
            config.n_taus_threshold = value;
        }
        if let Some(value) = extract_optional_field(&obj, "dtau_threshold")? {
            config.dtau_threshold = value;
        }
        if let Some(value) = extract_optional_field(&obj, "discard")? {
            config.discard = value;
        }
        if let Some(value) = extract_optional_field(&obj, "terminate")? {
            config.terminate = value;
        }
        if let Some(value) = extract_optional_field(&obj, "sokal_window")? {
            config.sokal_window = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "verbose")? {
            config.verbose = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyLBFGSBFTerminator {
    pub eps_abs: Float,
}

impl Default for PyLBFGSBFTerminator {
    fn default() -> Self {
        Self {
            eps_abs: Float::sqrt(Float::EPSILON),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyLBFGSBFTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "eps_abs")? {
            config.eps_abs = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyLBFGSBGTerminator {
    pub eps_abs: Float,
}

impl Default for PyLBFGSBGTerminator {
    fn default() -> Self {
        Self {
            eps_abs: Float::cbrt(Float::EPSILON),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyLBFGSBGTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "eps_abs")? {
            config.eps_abs = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyLBFGSBInfNormGTerminator {
    pub eps_abs: Float,
}

impl Default for PyLBFGSBInfNormGTerminator {
    fn default() -> Self {
        Self {
            eps_abs: Float::cbrt(Float::EPSILON),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyLBFGSBInfNormGTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "eps_abs")? {
            config.eps_abs = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PyNelderMeadFTerminator {
    Amoeba { eps_rel: Float },
    Absolute { eps_abs: Float },
    StdDev { eps_abs: Float },
}

impl Default for PyNelderMeadFTerminator {
    fn default() -> Self {
        Self::StdDev {
            eps_abs: Float::EPSILON.powf(0.25),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyNelderMeadFTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let kind: Option<String> = extract_optional_field(&obj, "kind")?;
        match kind
            .unwrap_or_else(|| "stddev".to_string())
            .trim()
            .to_ascii_lowercase()
            .replace(['-', ' '], "_")
            .as_str()
        {
            "amoeba" => Ok(Self::Amoeba {
                eps_rel: extract_optional_field(&obj, "eps_rel")?
                    .unwrap_or(Float::EPSILON.powf(0.25)),
            }),
            "absolute" => Ok(Self::Absolute {
                eps_abs: extract_optional_field(&obj, "eps_abs")?
                    .unwrap_or(Float::EPSILON.powf(0.25)),
            }),
            "stddev" => Ok(Self::StdDev {
                eps_abs: extract_optional_field(&obj, "eps_abs")?
                    .unwrap_or(Float::EPSILON.powf(0.25)),
            }),
            other => Err(GaneshError::ConfigError(format!(
                "unknown Nelder-Mead f terminator kind `{other}`"
            ))
            .into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PyNelderMeadXTerminator {
    Diameter { eps_abs: Float },
    Higham { eps_rel: Float },
    Rowan { eps_rel: Float },
    Singer { eps_rel: Float },
}

impl Default for PyNelderMeadXTerminator {
    fn default() -> Self {
        Self::Singer {
            eps_rel: Float::EPSILON.powf(0.25),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyNelderMeadXTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let kind: Option<String> = extract_optional_field(&obj, "kind")?;
        match kind
            .unwrap_or_else(|| "singer".to_string())
            .trim()
            .to_ascii_lowercase()
            .replace(['-', ' '], "_")
            .as_str()
        {
            "diameter" => Ok(Self::Diameter {
                eps_abs: extract_optional_field(&obj, "eps_abs")?
                    .unwrap_or(Float::EPSILON.powf(0.25)),
            }),
            "higham" => Ok(Self::Higham {
                eps_rel: extract_optional_field(&obj, "eps_rel")?
                    .unwrap_or(Float::EPSILON.powf(0.25)),
            }),
            "rowan" => Ok(Self::Rowan {
                eps_rel: extract_optional_field(&obj, "eps_rel")?
                    .unwrap_or(Float::EPSILON.powf(0.25)),
            }),
            "singer" => Ok(Self::Singer {
                eps_rel: extract_optional_field(&obj, "eps_rel")?
                    .unwrap_or(Float::EPSILON.powf(0.25)),
            }),
            other => Err(GaneshError::ConfigError(format!(
                "unknown Nelder-Mead x terminator kind `{other}`"
            ))
            .into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyAdamEMATerminator {
    pub beta_c: Float,
    pub eps_loss: Float,
    pub patience: usize,
}

impl Default for PyAdamEMATerminator {
    fn default() -> Self {
        Self {
            beta_c: 0.9,
            eps_loss: Float::EPSILON.sqrt(),
            patience: 1,
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyAdamEMATerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "beta_c")? {
            config.beta_c = value;
        }
        if let Some(value) = extract_optional_field(&obj, "eps_loss")? {
            config.eps_loss = value;
        }
        if let Some(value) = extract_optional_field(&obj, "patience")? {
            config.patience = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyConjugateGradientGTerminator {
    pub eps_abs: Float,
}

impl Default for PyConjugateGradientGTerminator {
    fn default() -> Self {
        Self {
            eps_abs: Float::cbrt(Float::EPSILON),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyConjugateGradientGTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "eps_abs")? {
            config.eps_abs = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyTrustRegionGTerminator {
    pub eps_abs: Float,
}

impl Default for PyTrustRegionGTerminator {
    fn default() -> Self {
        Self {
            eps_abs: Float::cbrt(Float::EPSILON),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyTrustRegionGTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "eps_abs")? {
            config.eps_abs = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PySimulatedAnnealingTemperatureTerminator {
    pub min_temperature: Float,
}

impl Default for PySimulatedAnnealingTemperatureTerminator {
    fn default() -> Self {
        Self {
            min_temperature: 1e-3,
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PySimulatedAnnealingTemperatureTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "min_temperature")? {
            config.min_temperature = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyCMAESSigmaTerminator {
    pub eps_abs: Float,
}

impl Default for PyCMAESSigmaTerminator {
    fn default() -> Self {
        Self { eps_abs: 1e-10 }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyCMAESSigmaTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "eps_abs")? {
            config.eps_abs = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PyCMAESNoEffectAxisTerminator;

impl<'a, 'py> FromPyObject<'a, 'py> for PyCMAESNoEffectAxisTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let _ = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        Ok(Self)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PyCMAESNoEffectCoordTerminator;

impl<'a, 'py> FromPyObject<'a, 'py> for PyCMAESNoEffectCoordTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let _ = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        Ok(Self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyCMAESConditionCovTerminator {
    pub max_condition: Float,
}

impl Default for PyCMAESConditionCovTerminator {
    fn default() -> Self {
        Self {
            max_condition: 1e14,
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyCMAESConditionCovTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_condition")? {
            config.max_condition = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PyCMAESEqualFunValuesTerminator;

impl<'a, 'py> FromPyObject<'a, 'py> for PyCMAESEqualFunValuesTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let _ = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        Ok(Self)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PyCMAESStagnationTerminator;

impl<'a, 'py> FromPyObject<'a, 'py> for PyCMAESStagnationTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let _ = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        Ok(Self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyCMAESTolXUpTerminator {
    pub max_growth: Float,
}

impl Default for PyCMAESTolXUpTerminator {
    fn default() -> Self {
        Self { max_growth: 1e4 }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyCMAESTolXUpTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_growth")? {
            config.max_growth = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyCMAESTolFunTerminator {
    pub eps_abs: Float,
}

impl Default for PyCMAESTolFunTerminator {
    fn default() -> Self {
        Self { eps_abs: 1e-12 }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyCMAESTolFunTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "eps_abs")? {
            config.eps_abs = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyCMAESTolXTerminator {
    pub eps_abs: Float,
}

impl Default for PyCMAESTolXTerminator {
    fn default() -> Self {
        Self { eps_abs: 0.0 }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyCMAESTolXTerminator {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_terminator__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "eps_abs")? {
            config.eps_abs = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyLBFGSBOptions {
    pub max_steps: Option<usize>,
    pub debug: bool,
    pub progress_every: Option<usize>,
    pub f_tolerance: Option<PyLBFGSBFTerminator>,
    pub g_tolerance: Option<PyLBFGSBGTerminator>,
    pub projected_gradient_tolerance: Option<PyLBFGSBInfNormGTerminator>,
}

impl PyLBFGSBOptions {
    pub fn build_callbacks<P, U, E>(
        &self,
    ) -> Callbacks<LBFGSB, P, GradientStatus, U, E, LBFGSBConfig>
    where
        P: Gradient<U, E>,
        U: std::fmt::Debug,
    {
        let mut callbacks = apply_common_callbacks(
            Callbacks::empty(),
            self.max_steps,
            self.debug,
            self.progress_every,
        );
        if let Some(f_tolerance) = &self.f_tolerance {
            callbacks = callbacks.with_terminator(
                LBFGSBFTerminator::new(f_tolerance.eps_abs)
                    .expect("PyLBFGSBFTerminator should be validated by Python wrapper"),
            );
        }
        if let Some(g_tolerance) = &self.g_tolerance {
            callbacks = callbacks.with_terminator(
                LBFGSBGTerminator::new(g_tolerance.eps_abs)
                    .expect("PyLBFGSBGTerminator should be validated by Python wrapper"),
            );
        }
        if let Some(projected_gradient_tolerance) = &self.projected_gradient_tolerance {
            callbacks = callbacks.with_terminator(
                LBFGSBInfNormGTerminator::new(projected_gradient_tolerance.eps_abs)
                    .expect("PyLBFGSBInfNormGTerminator should be validated by Python wrapper"),
            );
        }
        callbacks
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyLBFGSBOptions {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_run_options__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_steps")? {
            config.max_steps = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "debug")? {
            config.debug = value;
        }
        if let Some(value) = extract_optional_field(&obj, "progress_every")? {
            config.progress_every = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "f_tolerance")? {
            config.f_tolerance = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "g_tolerance")? {
            config.g_tolerance = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "projected_gradient_tolerance")? {
            config.projected_gradient_tolerance = Some(value);
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PyNelderMeadOptions {
    pub max_steps: Option<usize>,
    pub debug: bool,
    pub progress_every: Option<usize>,
    pub f_terminators: Vec<PyNelderMeadFTerminator>,
    pub x_terminators: Vec<PyNelderMeadXTerminator>,
}

impl Default for PyNelderMeadOptions {
    fn default() -> Self {
        Self {
            max_steps: None,
            debug: false,
            progress_every: None,
            f_terminators: vec![PyNelderMeadFTerminator::default()],
            x_terminators: vec![PyNelderMeadXTerminator::default()],
        }
    }
}

impl PyNelderMeadOptions {
    pub fn build_callbacks<P, U, E>(
        &self,
    ) -> Callbacks<NelderMead, P, GradientFreeStatus, U, E, NelderMeadConfig>
    where
        P: CostFunction<U, E>,
        U: std::fmt::Debug,
    {
        let mut callbacks = apply_common_callbacks(
            Callbacks::empty(),
            self.max_steps,
            self.debug,
            self.progress_every,
        );
        for f_terminator in &self.f_terminators {
            callbacks = callbacks.with_terminator(match f_terminator {
                PyNelderMeadFTerminator::Amoeba { eps_rel } => {
                    NelderMeadFTerminator::Amoeba { eps_rel: *eps_rel }
                }
                PyNelderMeadFTerminator::Absolute { eps_abs } => {
                    NelderMeadFTerminator::Absolute { eps_abs: *eps_abs }
                }
                PyNelderMeadFTerminator::StdDev { eps_abs } => {
                    NelderMeadFTerminator::StdDev { eps_abs: *eps_abs }
                }
            });
        }
        for x_terminator in &self.x_terminators {
            callbacks = callbacks.with_terminator(match x_terminator {
                PyNelderMeadXTerminator::Diameter { eps_abs } => {
                    NelderMeadXTerminator::Diameter { eps_abs: *eps_abs }
                }
                PyNelderMeadXTerminator::Higham { eps_rel } => {
                    NelderMeadXTerminator::Higham { eps_rel: *eps_rel }
                }
                PyNelderMeadXTerminator::Rowan { eps_rel } => {
                    NelderMeadXTerminator::Rowan { eps_rel: *eps_rel }
                }
                PyNelderMeadXTerminator::Singer { eps_rel } => {
                    NelderMeadXTerminator::Singer { eps_rel: *eps_rel }
                }
            });
        }
        callbacks
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyNelderMeadOptions {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_run_options__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_steps")? {
            config.max_steps = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "debug")? {
            config.debug = value;
        }
        if let Some(value) = extract_optional_field(&obj, "progress_every")? {
            config.progress_every = Some(value);
        }
        if let Some(value) = extract_optional_one_or_many_field(&obj, "f_terminators")? {
            config.f_terminators = value;
        }
        if let Some(value) = extract_optional_one_or_many_field(&obj, "x_terminators")? {
            config.x_terminators = value;
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyPSOOptions {
    pub max_steps: Option<usize>,
    pub debug: bool,
    pub progress_every: Option<usize>,
}

impl PyPSOOptions {
    pub fn build_callbacks<P, U, E>(&self) -> Callbacks<PSO, P, SwarmStatus, U, E, PSOConfig>
    where
        P: CostFunction<U, E>,
        U: std::fmt::Debug,
    {
        apply_common_callbacks(
            Callbacks::empty(),
            self.max_steps,
            self.debug,
            self.progress_every,
        )
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyPSOOptions {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_run_options__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_steps")? {
            config.max_steps = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "debug")? {
            config.debug = value;
        }
        if let Some(value) = extract_optional_field(&obj, "progress_every")? {
            config.progress_every = Some(value);
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyDifferentialEvolutionOptions {
    pub max_steps: Option<usize>,
    pub debug: bool,
    pub progress_every: Option<usize>,
}

impl PyDifferentialEvolutionOptions {
    pub fn build_callbacks<P, U, E>(
        &self,
    ) -> Callbacks<DifferentialEvolution, P, GradientFreeStatus, U, E, DifferentialEvolutionConfig>
    where
        P: CostFunction<U, E>,
        U: std::fmt::Debug,
    {
        apply_common_callbacks(
            Callbacks::empty(),
            self.max_steps,
            self.debug,
            self.progress_every,
        )
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyDifferentialEvolutionOptions {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_run_options__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_steps")? {
            config.max_steps = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "debug")? {
            config.debug = value;
        }
        if let Some(value) = extract_optional_field(&obj, "progress_every")? {
            config.progress_every = Some(value);
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyAIESOptions {
    pub max_steps: Option<usize>,
    pub debug: bool,
    pub progress_every: Option<usize>,
    pub autocorrelation: Option<PyAutocorrelationTerminator>,
}

impl PyAIESOptions {
    pub fn build_callbacks<P, U, E>(&self) -> Callbacks<AIES, P, EnsembleStatus, U, E, AIESConfig>
    where
        P: LogDensity<U, E>,
        U: std::fmt::Debug,
    {
        let mut callbacks = apply_common_callbacks(
            Callbacks::empty(),
            self.max_steps,
            self.debug,
            self.progress_every,
        );
        if let Some(autocorrelation) = &self.autocorrelation {
            callbacks = callbacks.with_terminator(autocorrelation.to_terminator());
        }
        callbacks
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyAIESOptions {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_run_options__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_steps")? {
            config.max_steps = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "debug")? {
            config.debug = value;
        }
        if let Some(value) = extract_optional_field(&obj, "progress_every")? {
            config.progress_every = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "autocorrelation")? {
            config.autocorrelation = Some(value);
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyESSOptions {
    pub max_steps: Option<usize>,
    pub debug: bool,
    pub progress_every: Option<usize>,
    pub autocorrelation: Option<PyAutocorrelationTerminator>,
}

impl PyESSOptions {
    pub fn build_callbacks<P, U, E>(&self) -> Callbacks<ESS, P, EnsembleStatus, U, E, ESSConfig>
    where
        P: LogDensity<U, E>,
        U: std::fmt::Debug,
    {
        let mut callbacks = apply_common_callbacks(
            Callbacks::empty(),
            self.max_steps,
            self.debug,
            self.progress_every,
        );
        if let Some(autocorrelation) = &self.autocorrelation {
            callbacks = callbacks.with_terminator(autocorrelation.to_terminator());
        }
        callbacks
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyESSOptions {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_run_options__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_steps")? {
            config.max_steps = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "debug")? {
            config.debug = value;
        }
        if let Some(value) = extract_optional_field(&obj, "progress_every")? {
            config.progress_every = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "autocorrelation")? {
            config.autocorrelation = Some(value);
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyCMAESOptions {
    pub max_steps: Option<usize>,
    pub debug: bool,
    pub progress_every: Option<usize>,
    pub sigma: Option<PyCMAESSigmaTerminator>,
    pub no_effect_axis: Option<PyCMAESNoEffectAxisTerminator>,
    pub no_effect_coord: Option<PyCMAESNoEffectCoordTerminator>,
    pub condition_cov: Option<PyCMAESConditionCovTerminator>,
    pub equal_fun_values: Option<PyCMAESEqualFunValuesTerminator>,
    pub stagnation: Option<PyCMAESStagnationTerminator>,
    pub tol_x_up: Option<PyCMAESTolXUpTerminator>,
    pub tol_fun: Option<PyCMAESTolFunTerminator>,
    pub tol_x: Option<PyCMAESTolXTerminator>,
}

impl PyCMAESOptions {
    pub fn build_callbacks<P, U, E>(
        &self,
    ) -> Callbacks<CMAES, P, GradientFreeStatus, U, E, CMAESConfig>
    where
        P: CostFunction<U, E>,
        U: std::fmt::Debug,
    {
        let mut callbacks = apply_common_callbacks(
            Callbacks::empty(),
            self.max_steps,
            self.debug,
            self.progress_every,
        );
        if let Some(sigma) = &self.sigma {
            callbacks = callbacks.with_terminator(CMAESSigmaTerminator {
                eps_abs: sigma.eps_abs,
            });
        }
        if self.no_effect_axis.is_some() {
            callbacks = callbacks.with_terminator(CMAESNoEffectAxisTerminator);
        }
        if self.no_effect_coord.is_some() {
            callbacks = callbacks.with_terminator(CMAESNoEffectCoordTerminator);
        }
        if let Some(condition_cov) = &self.condition_cov {
            callbacks = callbacks.with_terminator(CMAESConditionCovTerminator {
                max_condition: condition_cov.max_condition,
            });
        }
        if self.equal_fun_values.is_some() {
            callbacks = callbacks.with_terminator(CMAESEqualFunValuesTerminator);
        }
        if self.stagnation.is_some() {
            callbacks = callbacks.with_terminator(CMAESStagnationTerminator);
        }
        if let Some(tol_x_up) = &self.tol_x_up {
            callbacks = callbacks.with_terminator(CMAESTolXUpTerminator {
                max_growth: tol_x_up.max_growth,
            });
        }
        if let Some(tol_fun) = &self.tol_fun {
            callbacks = callbacks.with_terminator(CMAESTolFunTerminator {
                eps_abs: tol_fun.eps_abs,
            });
        }
        if let Some(tol_x) = &self.tol_x {
            callbacks = callbacks.with_terminator(CMAESTolXTerminator {
                eps_abs: tol_x.eps_abs,
            });
        }
        callbacks
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyCMAESOptions {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_run_options__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_steps")? {
            config.max_steps = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "debug")? {
            config.debug = value;
        }
        if let Some(value) = extract_optional_field(&obj, "progress_every")? {
            config.progress_every = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "sigma")? {
            config.sigma = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "no_effect_axis")? {
            config.no_effect_axis = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "no_effect_coord")? {
            config.no_effect_coord = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "condition_cov")? {
            config.condition_cov = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "equal_fun_values")? {
            config.equal_fun_values = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "stagnation")? {
            config.stagnation = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "tol_x_up")? {
            config.tol_x_up = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "tol_fun")? {
            config.tol_fun = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "tol_x")? {
            config.tol_x = Some(value);
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyAdamOptions {
    pub max_steps: Option<usize>,
    pub debug: bool,
    pub progress_every: Option<usize>,
    pub ema: Option<PyAdamEMATerminator>,
}

impl PyAdamOptions {
    pub fn build_callbacks<P, U, E>(&self) -> Callbacks<Adam, P, GradientStatus, U, E, AdamConfig>
    where
        P: Gradient<U, E>,
        U: std::fmt::Debug,
    {
        let mut callbacks = apply_common_callbacks(
            Callbacks::empty(),
            self.max_steps,
            self.debug,
            self.progress_every,
        );
        if let Some(ema) = &self.ema {
            callbacks = callbacks.with_terminator(AdamEMATerminator {
                beta_c: ema.beta_c,
                eps_loss: ema.eps_loss,
                patience: ema.patience,
            });
        }
        callbacks
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyAdamOptions {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_run_options__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_steps")? {
            config.max_steps = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "debug")? {
            config.debug = value;
        }
        if let Some(value) = extract_optional_field(&obj, "progress_every")? {
            config.progress_every = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "ema")? {
            config.ema = Some(value);
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyConjugateGradientOptions {
    pub max_steps: Option<usize>,
    pub debug: bool,
    pub progress_every: Option<usize>,
    pub g_tolerance: Option<PyConjugateGradientGTerminator>,
}

impl PyConjugateGradientOptions {
    pub fn build_callbacks<P, U, E>(
        &self,
    ) -> Callbacks<ConjugateGradient, P, GradientStatus, U, E, ConjugateGradientConfig>
    where
        P: Gradient<U, E>,
        U: std::fmt::Debug,
    {
        let mut callbacks = apply_common_callbacks(
            Callbacks::empty(),
            self.max_steps,
            self.debug,
            self.progress_every,
        );
        if let Some(g_tolerance) = &self.g_tolerance {
            callbacks = callbacks.with_terminator(
                ConjugateGradientGTerminator::new(g_tolerance.eps_abs)
                    .expect("PyConjugateGradientGTerminator should be validated by Python wrapper"),
            );
        }
        callbacks
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyConjugateGradientOptions {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_run_options__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_steps")? {
            config.max_steps = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "debug")? {
            config.debug = value;
        }
        if let Some(value) = extract_optional_field(&obj, "progress_every")? {
            config.progress_every = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "g_tolerance")? {
            config.g_tolerance = Some(value);
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PyTrustRegionOptions {
    pub max_steps: Option<usize>,
    pub debug: bool,
    pub progress_every: Option<usize>,
    pub g_tolerance: Option<PyTrustRegionGTerminator>,
}

impl PyTrustRegionOptions {
    pub fn build_callbacks<P, U, E>(
        &self,
    ) -> Callbacks<TrustRegion, P, GradientStatus, U, E, TrustRegionConfig>
    where
        P: Gradient<U, E>,
        U: std::fmt::Debug,
    {
        let mut callbacks = apply_common_callbacks(
            Callbacks::empty(),
            self.max_steps,
            self.debug,
            self.progress_every,
        );
        if let Some(g_tolerance) = &self.g_tolerance {
            callbacks = callbacks.with_terminator(
                TrustRegionGTerminator::new(g_tolerance.eps_abs)
                    .expect("PyTrustRegionGTerminator should be validated by Python wrapper"),
            );
        }
        callbacks
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyTrustRegionOptions {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_run_options__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_steps")? {
            config.max_steps = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "debug")? {
            config.debug = value;
        }
        if let Some(value) = extract_optional_field(&obj, "progress_every")? {
            config.progress_every = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "g_tolerance")? {
            config.g_tolerance = Some(value);
        }
        Ok(config)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PySimulatedAnnealingOptions {
    pub max_steps: Option<usize>,
    pub debug: bool,
    pub progress_every: Option<usize>,
    pub temperature: Option<PySimulatedAnnealingTemperatureTerminator>,
}

impl PySimulatedAnnealingOptions {
    pub fn build_callbacks<P, U, E, I>(
        &self,
    ) -> Callbacks<SimulatedAnnealing, P, SimulatedAnnealingStatus<I>, U, E, SimulatedAnnealingConfig>
    where
        P: SimulatedAnnealingGenerator<U, E, Input = I>,
        I: Serialize + for<'de> Deserialize<'de> + Clone + Default + std::fmt::Debug,
        U: std::fmt::Debug,
    {
        let mut callbacks = apply_common_callbacks(
            Callbacks::empty(),
            self.max_steps,
            self.debug,
            self.progress_every,
        );
        if let Some(temperature) = &self.temperature {
            callbacks = callbacks.with_terminator(SimulatedAnnealingTerminator {
                min_temperature: temperature.min_temperature,
            });
        }
        callbacks
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PySimulatedAnnealingOptions {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_run_options__")?;
        let mut config = Self::default();
        if let Some(value) = extract_optional_field(&obj, "max_steps")? {
            config.max_steps = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "debug")? {
            config.debug = value;
        }
        if let Some(value) = extract_optional_field(&obj, "progress_every")? {
            config.progress_every = Some(value);
        }
        if let Some(value) = extract_optional_field(&obj, "temperature")? {
            config.temperature = Some(value);
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{
        types::{PyAnyMethods, PyDict, PyDictMethods},
        Bound, PyAny, Python,
    };

    use super::*;
    use crate::{
        algorithms::{
            gradient::{
                Adam, AdamConfig, ConjugateGradient, ConjugateGradientConfig, LBFGSBConfig,
                TrustRegion, TrustRegionConfig, LBFGSB,
            },
            gradient_free::{
                nelder_mead::NelderMeadInit, CMAESConfig, CMAESInit, NelderMead, NelderMeadConfig,
                CMAES,
            },
            mcmc::{aies::AIESInit, ess::ESSInit, AIESConfig, ESSConfig, AIES, ESS},
        },
        traits::{Algorithm, CostFunction, Gradient, LogDensity},
        DVector,
    };
    use std::convert::Infallible;

    fn package_root() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/python")
    }

    fn import_ganesh<'py>(py: Python<'py>) -> Bound<'py, PyAny> {
        let sys = py.import("sys").unwrap();
        sys.getattr("path")
            .unwrap()
            .call_method1("insert", (0, package_root()))
            .unwrap();
        py.import("ganesh").unwrap().into_any()
    }

    struct Quadratic;
    struct GaussianLogDensity;

    impl CostFunction for Quadratic {
        fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(x.dot(x))
        }
    }

    impl Gradient for Quadratic {
        fn gradient(&self, x: &DVector<Float>, _args: &()) -> Result<DVector<Float>, Infallible> {
            Ok(x * 2.0)
        }
    }

    impl LogDensity for GaussianLogDensity {
        fn log_density(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(-0.5 * x.dot(x))
        }
    }

    #[test]
    fn pure_python_aies_run_options_build_callbacks() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "autocorrelation",
                    ganesh
                        .getattr("AutocorrelationTerminator")
                        .unwrap()
                        .call0()
                        .unwrap(),
                )
                .unwrap();
            kwargs.set_item("max_steps", 2).unwrap();
            kwargs.set_item("progress_every", 1).unwrap();
            let obj = ganesh
                .getattr("AIESOptions")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap();
            let options: PyAIESOptions = obj.extract().unwrap();
            let callbacks = options.build_callbacks::<GaussianLogDensity, (), Infallible>();
            let init = AIESInit::new(vec![
                DVector::from_vec(vec![0.0, 0.0]),
                DVector::from_vec(vec![0.1, 0.0]),
                DVector::from_vec(vec![0.0, 0.1]),
                DVector::from_vec(vec![0.1, 0.1]),
            ])
            .unwrap();
            let config = AIESConfig::default();
            let _summary = AIES::default()
                .process(&GaussianLogDensity, &(), init, config, callbacks)
                .unwrap();
        });
    }

    #[test]
    fn pure_python_ess_run_options_build_callbacks() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "autocorrelation",
                    ganesh
                        .getattr("AutocorrelationTerminator")
                        .unwrap()
                        .call0()
                        .unwrap(),
                )
                .unwrap();
            kwargs.set_item("max_steps", 2).unwrap();
            let obj = ganesh
                .getattr("ESSOptions")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap();
            let options: PyESSOptions = obj.extract().unwrap();
            let callbacks = options.build_callbacks::<GaussianLogDensity, (), Infallible>();
            let init = ESSInit::new(vec![
                DVector::from_vec(vec![0.0, 0.0]),
                DVector::from_vec(vec![0.1, 0.0]),
                DVector::from_vec(vec![0.0, 0.1]),
                DVector::from_vec(vec![0.1, 0.1]),
            ])
            .unwrap();
            let config = ESSConfig::default();
            let _summary = ESS::default()
                .process(&GaussianLogDensity, &(), init, config, callbacks)
                .unwrap();
        });
    }

    #[test]
    fn pure_python_cmaes_run_options_build_callbacks() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "sigma",
                    ganesh
                        .getattr("CMAESSigmaTerminator")
                        .unwrap()
                        .call0()
                        .unwrap(),
                )
                .unwrap();
            kwargs
                .set_item(
                    "no_effect_axis",
                    ganesh
                        .getattr("CMAESNoEffectAxisTerminator")
                        .unwrap()
                        .call0()
                        .unwrap(),
                )
                .unwrap();
            kwargs.set_item("max_steps", 1).unwrap();
            let obj = ganesh
                .getattr("CMAESOptions")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap();
            let options: PyCMAESOptions = obj.extract().unwrap();
            let callbacks = options.build_callbacks::<Quadratic, (), Infallible>();
            let init = CMAESInit::new([0.5, -0.5], 0.3).unwrap();
            let config = CMAESConfig::default();
            let _summary = CMAES::default()
                .process(&Quadratic, &(), init, config, callbacks)
                .unwrap();
        });
    }

    #[test]
    fn pure_python_lbfgsb_run_options_build_callbacks() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "f_tolerance",
                    ganesh
                        .getattr("LBFGSBFTerminator")
                        .unwrap()
                        .call0()
                        .unwrap(),
                )
                .unwrap();
            kwargs
                .set_item(
                    "projected_gradient_tolerance",
                    ganesh
                        .getattr("LBFGSBInfNormGTerminator")
                        .unwrap()
                        .call0()
                        .unwrap(),
                )
                .unwrap();
            kwargs.set_item("max_steps", 2).unwrap();
            let obj = ganesh
                .getattr("LBFGSBOptions")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap();
            let options: PyLBFGSBOptions = obj.extract().unwrap();
            let callbacks = options.build_callbacks::<Quadratic, (), Infallible>();
            let config = LBFGSBConfig::default();
            let _summary = LBFGSB::default()
                .process(
                    &Quadratic,
                    &(),
                    DVector::from_row_slice(&[1.0, -1.0]),
                    config,
                    callbacks,
                )
                .unwrap();
        });
    }

    #[test]
    fn pure_python_nelder_mead_run_options_build_callbacks() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "f_terminators",
                    vec![ganesh
                        .getattr("NelderMeadStdDevFTerminator")
                        .unwrap()
                        .call0()
                        .unwrap()],
                )
                .unwrap();
            kwargs
                .set_item(
                    "x_terminators",
                    vec![ganesh
                        .getattr("NelderMeadDiameterXTerminator")
                        .unwrap()
                        .call0()
                        .unwrap()],
                )
                .unwrap();
            kwargs.set_item("max_steps", 2).unwrap();
            let obj = ganesh
                .getattr("NelderMeadOptions")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap();
            let options: PyNelderMeadOptions = obj.extract().unwrap();
            let callbacks = options.build_callbacks::<Quadratic, (), Infallible>();
            let init = NelderMeadInit::new([1.0, -1.0]);
            let config = NelderMeadConfig::default();
            let _summary = NelderMead::default()
                .process(&Quadratic, &(), init, config, callbacks)
                .unwrap();
        });
    }

    #[test]
    fn pure_python_adam_run_options_build_callbacks() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "ema",
                    ganesh
                        .getattr("AdamEMATerminator")
                        .unwrap()
                        .call0()
                        .unwrap(),
                )
                .unwrap();
            kwargs.set_item("max_steps", 2).unwrap();
            let obj = ganesh
                .getattr("AdamOptions")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap();
            let options: PyAdamOptions = obj.extract().unwrap();
            let callbacks = options.build_callbacks::<Quadratic, (), Infallible>();
            let config = AdamConfig::default();
            let _summary = Adam::default()
                .process(
                    &Quadratic,
                    &(),
                    DVector::from_row_slice(&[1.0, -1.0]),
                    config,
                    callbacks,
                )
                .unwrap();
        });
    }

    #[test]
    fn pure_python_conjugate_gradient_run_options_build_callbacks() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "g_tolerance",
                    ganesh
                        .getattr("ConjugateGradientGTerminator")
                        .unwrap()
                        .call0()
                        .unwrap(),
                )
                .unwrap();
            kwargs.set_item("max_steps", 2).unwrap();
            let obj = ganesh
                .getattr("ConjugateGradientOptions")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap();
            let options: PyConjugateGradientOptions = obj.extract().unwrap();
            let callbacks = options.build_callbacks::<Quadratic, (), Infallible>();
            let config = ConjugateGradientConfig::default();
            let _summary = ConjugateGradient::default()
                .process(
                    &Quadratic,
                    &(),
                    DVector::from_row_slice(&[1.0, -1.0]),
                    config,
                    callbacks,
                )
                .unwrap();
        });
    }

    #[test]
    fn pure_python_trust_region_run_options_build_callbacks() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "g_tolerance",
                    ganesh
                        .getattr("TrustRegionGTerminator")
                        .unwrap()
                        .call0()
                        .unwrap(),
                )
                .unwrap();
            kwargs.set_item("max_steps", 2).unwrap();
            let obj = ganesh
                .getattr("TrustRegionOptions")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap();
            let options: PyTrustRegionOptions = obj.extract().unwrap();
            let callbacks = options.build_callbacks::<Quadratic, (), Infallible>();
            let config = TrustRegionConfig::default();
            let _summary = TrustRegion::default()
                .process(
                    &Quadratic,
                    &(),
                    DVector::from_row_slice(&[1.0, -1.0]),
                    config,
                    callbacks,
                )
                .unwrap();
        });
    }
}
