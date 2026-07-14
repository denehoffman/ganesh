use std::{marker::PhantomData, ops::ControlFlow, sync::Arc};

use parking_lot::Mutex;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::{
    algorithms::{
        gradient::{
            AdamEMATerminator, ConjugateGradientGTerminator, LBFGSBFTerminator, LBFGSBGTerminator,
            LBFGSBInfNormGTerminator, TrustRegionGTerminator,
        },
        gradient_free::{
            CMAESConditionCovTerminator, CMAESEqualFunValuesTerminator,
            CMAESNoEffectAxisTerminator, CMAESNoEffectCoordTerminator, CMAESSigmaTerminator,
            CMAESStagnationTerminator, CMAESTolFunTerminator, CMAESTolXTerminator,
            CMAESTolXUpTerminator, NelderMeadFTerminator, NelderMeadXTerminator,
            SimulatedAnnealingTerminator,
        },
        mcmc::AutocorrelationTerminator,
    },
    core::{Callbacks, DebugObserver, MaxSteps, ProgressObserver},
    traits::{Algorithm, Observer, Status, Terminator},
};

type AlgorithmTypes<A, P, S, U, E, C> = fn() -> (A, P, S, U, E, C);

fn positive(value: f64, name: &str) -> PyResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must be finite and positive"
        )))
    }
}

/// Python-visible settings for Ganesh's maximum-step terminator.
#[pyclass(name = "MaxSteps", frozen, from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyMaxSteps {
    steps: usize,
}

#[pymethods]
impl PyMaxSteps {
    #[new]
    fn new(steps: usize) -> PyResult<Self> {
        if steps == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "steps must be greater than zero",
            ));
        }
        Ok(Self { steps })
    }

    #[getter]
    const fn steps(&self) -> usize {
        self.steps
    }
}

impl From<PyMaxSteps> for MaxSteps {
    fn from(value: PyMaxSteps) -> Self {
        Self(value.steps)
    }
}

/// Python-visible settings for Ganesh's concise progress observer.
#[pyclass(name = "ProgressObserver", frozen, from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyProgressObserver {
    interval: usize,
}

#[pymethods]
impl PyProgressObserver {
    #[new]
    #[pyo3(signature = (*, interval=1))]
    fn new(interval: usize) -> PyResult<Self> {
        if interval == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "interval must be greater than zero",
            ));
        }
        Ok(Self { interval })
    }

    #[getter]
    const fn interval(&self) -> usize {
        self.interval
    }
}

impl From<PyProgressObserver> for ProgressObserver {
    fn from(value: PyProgressObserver) -> Self {
        Self::new(value.interval)
    }
}

/// Python-visible settings for Ganesh's verbose debugging observer.
#[pyclass(name = "DebugObserver", frozen, from_py_object)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PyDebugObserver;

#[pymethods]
impl PyDebugObserver {
    #[new]
    const fn new() -> Self {
        Self
    }
}

impl From<PyDebugObserver> for DebugObserver {
    fn from(_: PyDebugObserver) -> Self {
        Self
    }
}

/// Python-visible settings for Adam's EMA terminator.
#[pyclass(name = "AdamEMATerminator", frozen, from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyAdamEMATerminator {
    beta_c: f64,
    eps_loss: f64,
    patience: usize,
}
#[pymethods]
impl PyAdamEMATerminator {
    #[new]
    #[pyo3(signature = (*, beta_c=0.9, eps_loss=f64::EPSILON.sqrt(), patience=1))]
    fn new(beta_c: f64, eps_loss: f64, patience: usize) -> PyResult<Self> {
        if !(0.0..1.0).contains(&beta_c) || patience == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "beta_c must be in [0, 1) and patience must be positive",
            ));
        }
        positive(eps_loss, "eps_loss")?;
        Ok(Self {
            beta_c,
            eps_loss,
            patience,
        })
    }
}
impl From<PyAdamEMATerminator> for AdamEMATerminator {
    fn from(v: PyAdamEMATerminator) -> Self {
        Self {
            beta_c: v.beta_c,
            eps_loss: v.eps_loss,
            patience: v.patience,
        }
    }
}

macro_rules! tolerance_terminator {
    ($py:ident, $python:literal, $rust:ty, $default:expr) => {
        #[doc = concat!("Python-visible settings for `", $python, "`.")]
        #[pyclass(name = $python, frozen, from_py_object)]
        #[derive(Clone, Copy, Debug)]
        pub struct $py {
            eps_abs: f64,
        }
        #[pymethods]
        impl $py {
            #[new]
            #[pyo3(signature = (*, eps_abs=$default))]
            fn new(eps_abs: f64) -> PyResult<Self> {
                positive(eps_abs, "eps_abs")?;
                Ok(Self { eps_abs })
            }
        }
        impl TryFrom<$py> for $rust {
            type Error = crate::error::GaneshError;
            fn try_from(value: $py) -> Result<Self, Self::Error> {
                <$rust>::new(value.eps_abs)
            }
        }
    };
}

tolerance_terminator!(
    PyConjugateGradientGTerminator,
    "ConjugateGradientGTerminator",
    ConjugateGradientGTerminator,
    f64::EPSILON.cbrt()
);
tolerance_terminator!(
    PyLBFGSBFTerminator,
    "LBFGSBFTerminator",
    LBFGSBFTerminator,
    f64::EPSILON.sqrt()
);
tolerance_terminator!(
    PyLBFGSBGTerminator,
    "LBFGSBGTerminator",
    LBFGSBGTerminator,
    f64::EPSILON.cbrt()
);
tolerance_terminator!(
    PyLBFGSBInfNormGTerminator,
    "LBFGSBInfNormGTerminator",
    LBFGSBInfNormGTerminator,
    f64::EPSILON.cbrt()
);
tolerance_terminator!(
    PyTrustRegionGTerminator,
    "TrustRegionGTerminator",
    TrustRegionGTerminator,
    f64::EPSILON.cbrt()
);

/// Python-visible Nelder-Mead objective terminator.
#[pyclass(name = "NelderMeadFTerminator", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyNelderMeadFTerminator {
    kind: String,
    tolerance: f64,
}
#[pymethods]
impl PyNelderMeadFTerminator {
    #[new]
    #[pyo3(signature = (*, kind="stddev", tolerance=f64::EPSILON.sqrt().sqrt()))]
    fn new(kind: &str, tolerance: f64) -> PyResult<Self> {
        positive(tolerance, "tolerance")?;
        if !matches!(kind, "amoeba" | "absolute" | "stddev") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "kind must be amoeba, absolute, or stddev",
            ));
        }
        Ok(Self {
            kind: kind.into(),
            tolerance,
        })
    }
}
impl From<PyNelderMeadFTerminator> for NelderMeadFTerminator {
    fn from(v: PyNelderMeadFTerminator) -> Self {
        match v.kind.as_str() {
            "amoeba" => Self::Amoeba {
                eps_rel: v.tolerance,
            },
            "absolute" => Self::Absolute {
                eps_abs: v.tolerance,
            },
            _ => Self::StdDev {
                eps_abs: v.tolerance,
            },
        }
    }
}

/// Python-visible Nelder-Mead position terminator.
#[pyclass(name = "NelderMeadXTerminator", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyNelderMeadXTerminator {
    kind: String,
    tolerance: f64,
}
#[pymethods]
impl PyNelderMeadXTerminator {
    #[new]
    #[pyo3(signature = (*, kind="singer", tolerance=f64::EPSILON.sqrt().sqrt()))]
    fn new(kind: &str, tolerance: f64) -> PyResult<Self> {
        positive(tolerance, "tolerance")?;
        if !matches!(kind, "diameter" | "higham" | "rowan" | "singer") {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "kind must be diameter, higham, rowan, or singer",
            ));
        }
        Ok(Self {
            kind: kind.into(),
            tolerance,
        })
    }
}
impl From<PyNelderMeadXTerminator> for NelderMeadXTerminator {
    fn from(v: PyNelderMeadXTerminator) -> Self {
        match v.kind.as_str() {
            "diameter" => Self::Diameter {
                eps_abs: v.tolerance,
            },
            "higham" => Self::Higham {
                eps_rel: v.tolerance,
            },
            "rowan" => Self::Rowan {
                eps_rel: v.tolerance,
            },
            _ => Self::Singer {
                eps_rel: v.tolerance,
            },
        }
    }
}

/// Python-visible simulated-annealing temperature terminator.
#[pyclass(name = "SimulatedAnnealingTerminator", frozen, from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PySimulatedAnnealingTerminator {
    min_temperature: f64,
}
#[pymethods]
impl PySimulatedAnnealingTerminator {
    #[new]
    #[pyo3(signature = (*, min_temperature=1e-3))]
    fn new(min_temperature: f64) -> PyResult<Self> {
        positive(min_temperature, "min_temperature")?;
        Ok(Self { min_temperature })
    }
}
impl From<PySimulatedAnnealingTerminator> for SimulatedAnnealingTerminator {
    fn from(v: PySimulatedAnnealingTerminator) -> Self {
        Self {
            min_temperature: v.min_temperature,
        }
    }
}

macro_rules! cmaes_scalar {
    ($py:ident, $python:literal, $rust:ty, $field:ident, $default:expr) => {
        #[doc = concat!("Python-visible settings for `", $python, "`.")]
        #[pyclass(name = $python, frozen, from_py_object)]
        #[derive(Clone, Copy, Debug)]
        pub struct $py {
            value: f64,
        }
        #[pymethods]
        impl $py {
            #[new]
            #[pyo3(signature = (*, value=$default))]
            fn new(value: f64) -> PyResult<Self> {
                if value < 0.0 || !value.is_finite() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "value must be finite and non-negative",
                    ));
                }
                Ok(Self { value })
            }
        }
        impl From<$py> for $rust {
            fn from(v: $py) -> Self {
                Self { $field: v.value }
            }
        }
    };
}
cmaes_scalar!(
    PyCMAESSigmaTerminator,
    "CMAESSigmaTerminator",
    CMAESSigmaTerminator,
    eps_abs,
    1e-10
);
cmaes_scalar!(
    PyCMAESConditionCovTerminator,
    "CMAESConditionCovTerminator",
    CMAESConditionCovTerminator,
    max_condition,
    1e14
);
cmaes_scalar!(
    PyCMAESTolXUpTerminator,
    "CMAESTolXUpTerminator",
    CMAESTolXUpTerminator,
    max_growth,
    1e4
);
cmaes_scalar!(
    PyCMAESTolFunTerminator,
    "CMAESTolFunTerminator",
    CMAESTolFunTerminator,
    eps_abs,
    1e-12
);
cmaes_scalar!(
    PyCMAESTolXTerminator,
    "CMAESTolXTerminator",
    CMAESTolXTerminator,
    eps_abs,
    0.0
);

macro_rules! cmaes_unit {
    ($py:ident, $python:literal, $rust:ty) => {
        #[doc = concat!("Python-visible `", $python, "`.")]
        #[pyclass(name = $python, frozen, from_py_object)]
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $py;
        #[pymethods]
        impl $py {
            #[new]
            const fn new() -> Self {
                Self
            }
        }
        impl From<$py> for $rust {
            fn from(_: $py) -> Self {
                Self
            }
        }
    };
}
cmaes_unit!(
    PyCMAESNoEffectAxisTerminator,
    "CMAESNoEffectAxisTerminator",
    CMAESNoEffectAxisTerminator
);
cmaes_unit!(
    PyCMAESNoEffectCoordTerminator,
    "CMAESNoEffectCoordTerminator",
    CMAESNoEffectCoordTerminator
);
cmaes_unit!(
    PyCMAESEqualFunValuesTerminator,
    "CMAESEqualFunValuesTerminator",
    CMAESEqualFunValuesTerminator
);
cmaes_unit!(
    PyCMAESStagnationTerminator,
    "CMAESStagnationTerminator",
    CMAESStagnationTerminator
);

/// Python-visible integrated-autocorrelation terminator settings.
#[pyclass(name = "AutocorrelationTerminator", frozen, from_py_object)]
#[derive(Clone, Copy, Debug)]
pub struct PyAutocorrelationTerminator {
    n_check: usize,
    n_taus_threshold: usize,
    dtau_threshold: f64,
    discard: f64,
    terminate: bool,
    sokal_window: Option<f64>,
    verbose: bool,
}
#[pymethods]
impl PyAutocorrelationTerminator {
    #[new]
    #[pyo3(signature = (*, n_check=50, n_taus_threshold=50, dtau_threshold=0.01, discard=0.5, terminate=true, sokal_window=None, verbose=false))]
    fn new(
        n_check: usize,
        n_taus_threshold: usize,
        dtau_threshold: f64,
        discard: f64,
        terminate: bool,
        sokal_window: Option<f64>,
        verbose: bool,
    ) -> PyResult<Self> {
        if n_check == 0 || n_taus_threshold == 0 || !(0.0..1.0).contains(&discard) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "check intervals must be positive and discard must be in [0, 1)",
            ));
        }
        positive(dtau_threshold, "dtau_threshold")?;
        if let Some(value) = sokal_window {
            positive(value, "sokal_window")?;
        }
        Ok(Self {
            n_check,
            n_taus_threshold,
            dtau_threshold,
            discard,
            terminate,
            sokal_window,
            verbose,
        })
    }
}
impl From<PyAutocorrelationTerminator> for AutocorrelationTerminator {
    fn from(v: PyAutocorrelationTerminator) -> Self {
        let mut out = Self::default()
            .with_n_check(v.n_check)
            .with_n_taus_threshold(v.n_taus_threshold)
            .with_dtau_threshold(v.dtau_threshold)
            .with_discard(v.discard)
            .with_terminate(v.terminate)
            .with_verbose(v.verbose);
        if let Some(value) = v.sokal_window {
            out = out.with_sokal_window(value);
        }
        out
    }
}

/// Immutable status information passed to Python callbacks.
#[pyclass(name = "StatusSnapshot", frozen, from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStatusSnapshot {
    step: usize,
    success: bool,
    message: String,
}

#[pymethods]
impl PyStatusSnapshot {
    #[getter]
    const fn step(&self) -> usize {
        self.step
    }

    #[getter]
    const fn success(&self) -> bool {
        self.success
    }

    #[getter]
    fn message(&self) -> &str {
        &self.message
    }
}

/// Build a Python callback snapshot for a Ganesh status.
pub trait ToPyStatusSnapshot: Status {
    /// Convert the status into an immutable callback snapshot.
    fn to_python_snapshot(&self, step: usize) -> PyStatusSnapshot {
        PyStatusSnapshot {
            step,
            success: self.success(),
            message: self.message().to_string(),
        }
    }
}

impl<S: Status> ToPyStatusSnapshot for S {}

#[derive(Clone)]
struct PythonCallable {
    callable: Arc<Py<PyAny>>,
    error: Arc<Mutex<Option<PyErr>>>,
}

impl PythonCallable {
    fn call(&self, step: usize, status: &impl ToPyStatusSnapshot) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let snapshot = Py::new(py, status.to_python_snapshot(step))?;
            self.callable.call1(py, (step, snapshot))
        })
    }

    fn record(&self, error: PyErr) {
        let mut slot = self.error.lock();
        if slot.is_none() {
            *slot = Some(error);
        }
    }
}

struct PythonObserver<A, P, S, U, E, C> {
    inner: PythonCallable,
    _types: PhantomData<AlgorithmTypes<A, P, S, U, E, C>>,
}

impl<A, P, S, U, E, C> Clone for PythonObserver<A, P, S, U, E, C> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _types: PhantomData,
        }
    }
}

impl<A, P, S, U, E, C> Observer<A, P, S, U, E, C> for PythonObserver<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C> + 'static,
    S: Status,
{
    fn observe(
        &mut self,
        current_step: usize,
        _algorithm: &A,
        _problem: &P,
        status: &S,
        _args: &U,
        _config: &C,
    ) {
        if let Err(error) = self.inner.call(current_step, status) {
            self.inner.record(error);
        }
    }
}

struct PythonTerminator<A, P, S, U, E, C> {
    inner: PythonCallable,
    _types: PhantomData<AlgorithmTypes<A, P, S, U, E, C>>,
}

impl<A, P, S, U, E, C> Clone for PythonTerminator<A, P, S, U, E, C> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _types: PhantomData,
        }
    }
}

impl<A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for PythonTerminator<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        status: &mut S,
        _args: &U,
        _config: &C,
    ) -> ControlFlow<()> {
        match self.inner.call(current_step, status) {
            Ok(value) => Python::attach(|py| match value.extract::<bool>(py) {
                Ok(true) => ControlFlow::Break(()),
                Ok(false) => ControlFlow::Continue(()),
                Err(error) => {
                    self.inner.record(PyTypeError::new_err(format!(
                        "Python terminator must return bool: {error}"
                    )));
                    ControlFlow::Break(())
                }
            }),
            Err(error) => {
                self.inner.record(error);
                ControlFlow::Break(())
            }
        }
    }
}

struct ErrorTerminator<A, P, S, U, E, C> {
    error: Arc<Mutex<Option<PyErr>>>,
    _types: PhantomData<AlgorithmTypes<A, P, S, U, E, C>>,
}

impl<A, P, S, U, E, C> Clone for ErrorTerminator<A, P, S, U, E, C> {
    fn clone(&self) -> Self {
        Self {
            error: self.error.clone(),
            _types: PhantomData,
        }
    }
}

impl<A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for ErrorTerminator<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C> + 'static,
    S: Status,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        _status: &mut S,
        _args: &U,
        _config: &C,
    ) -> ControlFlow<()> {
        if self.error.lock().is_some() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// A callback set built from Python settings and extensible with downstream Rust callbacks.
pub struct PythonCallbackBundle<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    callbacks: Callbacks<A, P, S, U, E, C>,
    error: Arc<Mutex<Option<PyErr>>>,
}

impl<A, P, S, U, E, C> PythonCallbackBundle<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C> + 'static,
    S: Status + 'static,
    P: 'static,
    U: 'static,
    E: 'static,
    C: 'static,
{
    /// Start with an existing Rust callback collection.
    #[must_use]
    pub fn new(callbacks: Callbacks<A, P, S, U, E, C>) -> Self {
        Self {
            callbacks,
            error: Arc::new(Mutex::new(None)),
        }
    }

    /// Append a downstream-defined Rust terminator.
    #[must_use]
    pub fn with_terminator<T>(mut self, terminator: T) -> Self
    where
        T: Terminator<A, P, S, U, E, C> + 'static,
    {
        self.callbacks = self.callbacks.with_terminator(terminator);
        self
    }

    /// Prepend a downstream-defined Rust terminator.
    #[must_use]
    pub fn with_terminator_first<T>(mut self, terminator: T) -> Self
    where
        T: Terminator<A, P, S, U, E, C> + 'static,
    {
        self.callbacks = self.callbacks.with_terminator_first(terminator);
        self
    }

    /// Append a downstream-defined Rust observer.
    #[must_use]
    pub fn with_observer<O>(mut self, observer: O) -> Self
    where
        O: Observer<A, P, S, U, E, C> + 'static,
    {
        self.callbacks = self.callbacks.with_observer(observer);
        self
    }

    /// Prepend a downstream-defined Rust observer.
    #[must_use]
    pub fn with_observer_first<O>(mut self, observer: O) -> Self
    where
        O: Observer<A, P, S, U, E, C> + 'static,
    {
        self.callbacks = self.callbacks.with_observer_first(observer);
        self
    }

    /// Append a Python observer with signature `(step, status) -> None`.
    #[must_use]
    pub fn with_python_observer(mut self, observer: Py<PyAny>) -> Self {
        let inner = PythonCallable {
            callable: Arc::new(observer),
            error: self.error.clone(),
        };
        self.callbacks = self.callbacks.with_observer(PythonObserver {
            inner,
            _types: PhantomData,
        });
        self.callbacks = self.callbacks.with_terminator(ErrorTerminator {
            error: self.error.clone(),
            _types: PhantomData,
        });
        self
    }

    /// Append a Python terminator with signature `(step, status) -> bool`.
    #[must_use]
    pub fn with_python_terminator(mut self, terminator: Py<PyAny>) -> Self {
        let inner = PythonCallable {
            callable: Arc::new(terminator),
            error: self.error.clone(),
        };
        self.callbacks = self.callbacks.with_terminator(PythonTerminator {
            inner,
            _types: PhantomData,
        });
        self.callbacks = self.callbacks.with_terminator(ErrorTerminator {
            error: self.error.clone(),
            _types: PhantomData,
        });
        self
    }

    /// Consume the bundle and return its callback collection and error state.
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn into_parts(self) -> (Callbacks<A, P, S, U, E, C>, Arc<Mutex<Option<PyErr>>>) {
        (self.callbacks, self.error)
    }
}

/// Run an algorithm with callback exception propagation.
pub fn process_with_python_callbacks<A, P, S, U, E, C, F>(
    algorithm: &mut A,
    problem: &P,
    args: &U,
    init: A::Init,
    config: C,
    bundle: PythonCallbackBundle<A, P, S, U, E, C>,
    map_algorithm_error: F,
) -> PyResult<A::Summary>
where
    A: Algorithm<P, S, U, E, Config = C> + 'static,
    S: Status + 'static,
    P: 'static,
    U: 'static,
    E: 'static,
    C: 'static,
    F: FnOnce(E) -> PyErr,
{
    let (callbacks, callback_error) = bundle.into_parts();
    let result = algorithm.process(problem, args, init, config, callbacks);
    let callback_error = callback_error.lock().take();
    if let Some(error) = callback_error {
        return Err(error);
    }
    result.map_err(map_algorithm_error)
}
