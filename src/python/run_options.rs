//! Python-facing run-options wrappers for built-in terminators and observers.
#![allow(missing_docs)]

use pyo3::{pyclass, pymethods};

use crate::{
    Float,
    algorithms::{
        gradient_free::{
            CMAES, CMAESConditionCovTerminator, CMAESEqualFunValuesTerminator,
            CMAESNoEffectAxisTerminator, CMAESNoEffectCoordTerminator, CMAESSigmaTerminator,
            CMAESStagnationTerminator, CMAESTolFunTerminator, CMAESTolXTerminator,
            CMAESTolXUpTerminator, CMAESConfig, GradientFreeStatus,
        },
        mcmc::{AIES, AIESConfig, AutocorrelationTerminator, ESS, ESSConfig, EnsembleStatus},
    },
    core::{Callbacks, DebugObserver, MaxSteps, ProgressObserver},
    traits::{Algorithm, CostFunction, LogDensity, Status},
};

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

/// Python-facing config for the built-in autocorrelation terminator.
#[pyclass(module = "ganesh", name = "AutocorrelationTerminatorConfig")]
#[derive(Clone)]
pub struct PyAutocorrelationTerminatorConfig {
    n_check: usize,
    n_taus_threshold: usize,
    dtau_threshold: Float,
    discard: Float,
    terminate: bool,
    sokal_window: Option<Float>,
    verbose: bool,
}

#[allow(missing_docs)]
#[pymethods]
impl PyAutocorrelationTerminatorConfig {
    #[new]
    #[pyo3(signature = (
        n_check=50,
        n_taus_threshold=50,
        dtau_threshold=0.01,
        discard=0.5,
        terminate=true,
        sokal_window=None,
        verbose=false
    ))]
    pub const fn new(
        n_check: usize,
        n_taus_threshold: usize,
        dtau_threshold: Float,
        discard: Float,
        terminate: bool,
        sokal_window: Option<Float>,
        verbose: bool,
    ) -> Self {
        Self {
            n_check,
            n_taus_threshold,
            dtau_threshold,
            discard,
            terminate,
            sokal_window,
            verbose,
        }
    }
}

impl Default for PyAutocorrelationTerminatorConfig {
    fn default() -> Self {
        Self::new(50, 50, 0.01, 0.5, true, None, false)
    }
}

impl PyAutocorrelationTerminatorConfig {
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

/// Python-facing config for the built-in CMA-ES sigma terminator.
#[pyclass(module = "ganesh", name = "CMAESSigmaTerminatorConfig")]
#[derive(Clone)]
pub struct PyCMAESSigmaTerminatorConfig {
    eps_abs: Float,
}

#[allow(missing_docs)]
#[pymethods]
impl PyCMAESSigmaTerminatorConfig {
    #[new]
    #[pyo3(signature = (eps_abs=1e-10))]
    pub const fn new(eps_abs: Float) -> Self {
        Self { eps_abs }
    }
}

impl Default for PyCMAESSigmaTerminatorConfig {
    fn default() -> Self {
        Self::new(1e-10)
    }
}

/// Python-facing config for the built-in CMA-ES no-effect-axis terminator.
#[pyclass(module = "ganesh", name = "CMAESNoEffectAxisTerminatorConfig")]
#[derive(Clone, Default)]
pub struct PyCMAESNoEffectAxisTerminatorConfig;

#[pymethods]
impl PyCMAESNoEffectAxisTerminatorConfig {
    #[new]
    pub const fn new() -> Self {
        Self
    }
}

/// Python-facing config for the built-in CMA-ES no-effect-coordinate terminator.
#[pyclass(module = "ganesh", name = "CMAESNoEffectCoordTerminatorConfig")]
#[derive(Clone, Default)]
pub struct PyCMAESNoEffectCoordTerminatorConfig;

#[pymethods]
impl PyCMAESNoEffectCoordTerminatorConfig {
    #[new]
    pub const fn new() -> Self {
        Self
    }
}

/// Python-facing config for the built-in CMA-ES covariance-condition terminator.
#[pyclass(module = "ganesh", name = "CMAESConditionCovTerminatorConfig")]
#[derive(Clone)]
pub struct PyCMAESConditionCovTerminatorConfig {
    max_condition: Float,
}

#[allow(missing_docs)]
#[pymethods]
impl PyCMAESConditionCovTerminatorConfig {
    #[new]
    #[pyo3(signature = (max_condition=1e14))]
    pub const fn new(max_condition: Float) -> Self {
        Self { max_condition }
    }
}

impl Default for PyCMAESConditionCovTerminatorConfig {
    fn default() -> Self {
        Self::new(1e14)
    }
}

/// Python-facing config for the built-in CMA-ES equal-function-values terminator.
#[pyclass(module = "ganesh", name = "CMAESEqualFunValuesTerminatorConfig")]
#[derive(Clone, Default)]
pub struct PyCMAESEqualFunValuesTerminatorConfig;

#[pymethods]
impl PyCMAESEqualFunValuesTerminatorConfig {
    #[new]
    pub const fn new() -> Self {
        Self
    }
}

/// Python-facing config for the built-in CMA-ES stagnation terminator.
#[pyclass(module = "ganesh", name = "CMAESStagnationTerminatorConfig")]
#[derive(Clone, Default)]
pub struct PyCMAESStagnationTerminatorConfig;

#[pymethods]
impl PyCMAESStagnationTerminatorConfig {
    #[new]
    pub const fn new() -> Self {
        Self
    }
}

/// Python-facing config for the built-in CMA-ES TolXUp terminator.
#[pyclass(module = "ganesh", name = "CMAESTolXUpTerminatorConfig")]
#[derive(Clone)]
pub struct PyCMAESTolXUpTerminatorConfig {
    max_growth: Float,
}

#[allow(missing_docs)]
#[pymethods]
impl PyCMAESTolXUpTerminatorConfig {
    #[new]
    #[pyo3(signature = (max_growth=1e4))]
    pub const fn new(max_growth: Float) -> Self {
        Self { max_growth }
    }
}

impl Default for PyCMAESTolXUpTerminatorConfig {
    fn default() -> Self {
        Self::new(1e4)
    }
}

/// Python-facing config for the built-in CMA-ES TolFun terminator.
#[pyclass(module = "ganesh", name = "CMAESTolFunTerminatorConfig")]
#[derive(Clone)]
pub struct PyCMAESTolFunTerminatorConfig {
    eps_abs: Float,
}

#[allow(missing_docs)]
#[pymethods]
impl PyCMAESTolFunTerminatorConfig {
    #[new]
    #[pyo3(signature = (eps_abs=1e-12))]
    pub const fn new(eps_abs: Float) -> Self {
        Self { eps_abs }
    }
}

impl Default for PyCMAESTolFunTerminatorConfig {
    fn default() -> Self {
        Self::new(1e-12)
    }
}

/// Python-facing config for the built-in CMA-ES TolX terminator.
#[pyclass(module = "ganesh", name = "CMAESTolXTerminatorConfig")]
#[derive(Clone)]
pub struct PyCMAESTolXTerminatorConfig {
    eps_abs: Float,
}

#[allow(missing_docs)]
#[pymethods]
impl PyCMAESTolXTerminatorConfig {
    #[new]
    #[pyo3(signature = (eps_abs=0.0))]
    pub const fn new(eps_abs: Float) -> Self {
        Self { eps_abs }
    }
}

impl Default for PyCMAESTolXTerminatorConfig {
    fn default() -> Self {
        Self::new(0.0)
    }
}

/// Python-facing run options for the AIES sampler.
#[pyclass(module = "ganesh", name = "AIESRunOptions")]
#[derive(Clone)]
pub struct PyAIESRunOptions {
    max_steps: Option<usize>,
    debug: bool,
    progress_every: Option<usize>,
    autocorrelation: Option<PyAutocorrelationTerminatorConfig>,
}

#[allow(missing_docs)]
#[pymethods]
impl PyAIESRunOptions {
    #[new]
    #[pyo3(signature = (
        max_steps=None,
        debug=false,
        progress_every=None,
        autocorrelation=None
    ))]
    pub fn new(
        max_steps: Option<usize>,
        debug: bool,
        progress_every: Option<usize>,
        autocorrelation: Option<PyAutocorrelationTerminatorConfig>,
    ) -> Self {
        Self {
            max_steps,
            debug,
            progress_every,
            autocorrelation,
        }
    }
}

impl Default for PyAIESRunOptions {
    fn default() -> Self {
        Self::new(None, false, None, None)
    }
}

impl PyAIESRunOptions {
    pub fn build_callbacks<P, U, E>(&self) -> Callbacks<AIES, P, EnsembleStatus, U, E, AIESConfig>
    where
        P: LogDensity<U, E>,
        U: std::fmt::Debug,
    {
        let mut callbacks =
            apply_common_callbacks(Callbacks::empty(), self.max_steps, self.debug, self.progress_every);
        if let Some(autocorrelation) = &self.autocorrelation {
            callbacks = callbacks.with_terminator(autocorrelation.to_terminator());
        }
        callbacks
    }
}

/// Python-facing run options for the ESS sampler.
#[pyclass(module = "ganesh", name = "ESSRunOptions")]
#[derive(Clone)]
pub struct PyESSRunOptions {
    max_steps: Option<usize>,
    debug: bool,
    progress_every: Option<usize>,
    autocorrelation: Option<PyAutocorrelationTerminatorConfig>,
}

#[allow(missing_docs)]
#[pymethods]
impl PyESSRunOptions {
    #[new]
    #[pyo3(signature = (
        max_steps=None,
        debug=false,
        progress_every=None,
        autocorrelation=None
    ))]
    pub fn new(
        max_steps: Option<usize>,
        debug: bool,
        progress_every: Option<usize>,
        autocorrelation: Option<PyAutocorrelationTerminatorConfig>,
    ) -> Self {
        Self {
            max_steps,
            debug,
            progress_every,
            autocorrelation,
        }
    }
}

impl Default for PyESSRunOptions {
    fn default() -> Self {
        Self::new(None, false, None, None)
    }
}

impl PyESSRunOptions {
    pub fn build_callbacks<P, U, E>(&self) -> Callbacks<ESS, P, EnsembleStatus, U, E, ESSConfig>
    where
        P: LogDensity<U, E>,
        U: std::fmt::Debug,
    {
        let mut callbacks =
            apply_common_callbacks(Callbacks::empty(), self.max_steps, self.debug, self.progress_every);
        if let Some(autocorrelation) = &self.autocorrelation {
            callbacks = callbacks.with_terminator(autocorrelation.to_terminator());
        }
        callbacks
    }
}

/// Python-facing run options for the CMA-ES optimizer.
#[pyclass(module = "ganesh", name = "CMAESRunOptions")]
#[derive(Clone)]
pub struct PyCMAESRunOptions {
    max_steps: Option<usize>,
    debug: bool,
    progress_every: Option<usize>,
    sigma: Option<PyCMAESSigmaTerminatorConfig>,
    no_effect_axis: Option<PyCMAESNoEffectAxisTerminatorConfig>,
    no_effect_coord: Option<PyCMAESNoEffectCoordTerminatorConfig>,
    condition_cov: Option<PyCMAESConditionCovTerminatorConfig>,
    equal_fun_values: Option<PyCMAESEqualFunValuesTerminatorConfig>,
    stagnation: Option<PyCMAESStagnationTerminatorConfig>,
    tol_x_up: Option<PyCMAESTolXUpTerminatorConfig>,
    tol_fun: Option<PyCMAESTolFunTerminatorConfig>,
    tol_x: Option<PyCMAESTolXTerminatorConfig>,
}

#[allow(missing_docs)]
#[pymethods]
impl PyCMAESRunOptions {
    #[new]
    #[pyo3(signature = (
        max_steps=None,
        debug=false,
        progress_every=None,
        sigma=None,
        no_effect_axis=None,
        no_effect_coord=None,
        condition_cov=None,
        equal_fun_values=None,
        stagnation=None,
        tol_x_up=None,
        tol_fun=None,
        tol_x=None
    ))]
    pub fn new(
        max_steps: Option<usize>,
        debug: bool,
        progress_every: Option<usize>,
        sigma: Option<PyCMAESSigmaTerminatorConfig>,
        no_effect_axis: Option<PyCMAESNoEffectAxisTerminatorConfig>,
        no_effect_coord: Option<PyCMAESNoEffectCoordTerminatorConfig>,
        condition_cov: Option<PyCMAESConditionCovTerminatorConfig>,
        equal_fun_values: Option<PyCMAESEqualFunValuesTerminatorConfig>,
        stagnation: Option<PyCMAESStagnationTerminatorConfig>,
        tol_x_up: Option<PyCMAESTolXUpTerminatorConfig>,
        tol_fun: Option<PyCMAESTolFunTerminatorConfig>,
        tol_x: Option<PyCMAESTolXTerminatorConfig>,
    ) -> Self {
        Self {
            max_steps,
            debug,
            progress_every,
            sigma,
            no_effect_axis,
            no_effect_coord,
            condition_cov,
            equal_fun_values,
            stagnation,
            tol_x_up,
            tol_fun,
            tol_x,
        }
    }
}

impl Default for PyCMAESRunOptions {
    fn default() -> Self {
        Self::new(
            None, false, None, None, None, None, None, None, None, None, None, None,
        )
    }
}

impl PyCMAESRunOptions {
    pub fn build_callbacks<P, U, E>(
        &self,
    ) -> Callbacks<CMAES, P, GradientFreeStatus, U, E, CMAESConfig>
    where
        P: CostFunction<U, E>,
        U: std::fmt::Debug,
    {
        let mut callbacks =
            apply_common_callbacks(Callbacks::empty(), self.max_steps, self.debug, self.progress_every);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DVector,
        algorithms::{
            gradient_free::CMAES,
            mcmc::{AIES, AIESConfig, ESS, ESSConfig},
        },
        traits::{CostFunction, LogDensity},
    };
    use std::convert::Infallible;

    struct Quadratic;
    struct GaussianLogDensity;

    impl CostFunction for Quadratic {
        fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(x.dot(x))
        }
    }

    impl LogDensity for GaussianLogDensity {
        fn log_density(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(-0.5 * x.dot(x))
        }
    }

    #[test]
    fn aies_run_options_build_callbacks() {
        let options = PyAIESRunOptions::new(
            Some(2),
            false,
            Some(1),
            Some(PyAutocorrelationTerminatorConfig::default()),
        );
        let callbacks = options.build_callbacks::<GaussianLogDensity, (), Infallible>();
        let config = AIESConfig::new(vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![0.1, 0.0]),
            DVector::from_vec(vec![0.0, 0.1]),
            DVector::from_vec(vec![0.1, 0.1]),
        ])
        .unwrap();
        let mut solver = AIES::default();
        let summary = solver.process(&GaussianLogDensity, &(), config, callbacks).unwrap();
        assert_eq!(summary.dimension.0, 4);
    }

    #[test]
    fn ess_run_options_build_callbacks() {
        let options = PyESSRunOptions::new(
            Some(2),
            false,
            Some(1),
            Some(PyAutocorrelationTerminatorConfig::default()),
        );
        let callbacks = options.build_callbacks::<GaussianLogDensity, (), Infallible>();
        let config = ESSConfig::new(vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![0.1, 0.0]),
            DVector::from_vec(vec![0.0, 0.1]),
        ])
        .unwrap();
        let mut solver = ESS::default();
        let summary = solver.process(&GaussianLogDensity, &(), config, callbacks).unwrap();
        assert_eq!(summary.dimension.0, 3);
    }

    #[test]
    fn cmaes_run_options_build_callbacks() {
        let options = PyCMAESRunOptions::new(
            Some(4),
            false,
            Some(1),
            Some(PyCMAESSigmaTerminatorConfig::default()),
            Some(PyCMAESNoEffectAxisTerminatorConfig::new()),
            None,
            Some(PyCMAESConditionCovTerminatorConfig::default()),
            None,
            None,
            None,
            Some(PyCMAESTolFunTerminatorConfig::default()),
            None,
        );
        let callbacks = options.build_callbacks::<Quadratic, (), Infallible>();
        let config = CMAESConfig::new([1.0, -1.0], 0.5).unwrap();
        let mut solver = CMAES::default();
        let summary = solver.process(&Quadratic, &(), config, callbacks).unwrap();
        assert!(summary.cost_evals > 0);
    }
}
