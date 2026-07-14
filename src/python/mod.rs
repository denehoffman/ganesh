//! Reusable `pyo3` types for downstream Python extension modules.
//!
//! Enable the `python` feature, then re-export the `ganesh` submodule from an inline downstream
//! `#[pymodule]`. This crate intentionally does not define a top-level extension module.
//!
//! ```
//! #[pyo3::pymodule]
//! mod my_extension {
//!     #[pymodule_export]
//!     use ganesh::python::ganesh;
//! }
//! ```

mod callbacks;
mod config;
mod numeric;
mod output;

pub use callbacks::*;
pub use config::*;
pub use numeric::*;
pub use output::*;

use pyo3::{create_exception, exceptions::PyException, PyErr};

#[allow(missing_docs)]
mod exceptions {
    use super::*;
    create_exception!(ganesh, GaneshError, PyException);
    create_exception!(ganesh, GaneshConfigError, GaneshError);
    create_exception!(ganesh, GaneshNumericalError, GaneshError);
    create_exception!(ganesh, GaneshCallbackError, GaneshError);
}
pub use exceptions::*;

pub(crate) fn ganesh_error(error: crate::error::GaneshError) -> PyErr {
    match error {
        crate::error::GaneshError::ConfigError(message) => GaneshConfigError::new_err(message),
        crate::error::GaneshError::NumericalError(message) => {
            GaneshNumericalError::new_err(message)
        }
    }
}

/// Declarative submodule for downstream `pyo3` modules.
#[pyo3::pymodule(submodule)]
pub mod ganesh {
    #[pymodule_export]
    use super::{
        GaneshCallbackError, GaneshConfigError, GaneshError, GaneshNumericalError,
        PyAIESConfig as AIESConfig, PyAIESInit as AIESInit, PyAIESMove as AIESMove,
        PyAdamConfig as AdamConfig, PyAdamEMATerminator as AdamEMATerminator,
        PyAutocorrelationTerminator as AutocorrelationTerminator, PyBounds as Bounds,
        PyCMAESConditionCovTerminator as CMAESConditionCovTerminator, PyCMAESConfig as CMAESConfig,
        PyCMAESEqualFunValuesTerminator as CMAESEqualFunValuesTerminator,
        PyCMAESNoEffectAxisTerminator as CMAESNoEffectAxisTerminator,
        PyCMAESNoEffectCoordTerminator as CMAESNoEffectCoordTerminator,
        PyCMAESSigmaTerminator as CMAESSigmaTerminator,
        PyCMAESStagnationTerminator as CMAESStagnationTerminator,
        PyCMAESTolFunTerminator as CMAESTolFunTerminator,
        PyCMAESTolXTerminator as CMAESTolXTerminator,
        PyCMAESTolXUpTerminator as CMAESTolXUpTerminator, PyChainStorage as ChainStorage,
        PyConjugateGradientConfig as ConjugateGradientConfig,
        PyConjugateGradientGTerminator as ConjugateGradientGTerminator,
        PyDebugObserver as DebugObserver,
        PyDifferentialEvolutionConfig as DifferentialEvolutionConfig, PyESSConfig as ESSConfig,
        PyESSInit as ESSInit, PyESSMove as ESSMove, PyEvalCounts as EvalCounts,
        PyHagerZhangLineSearch as HagerZhangLineSearch, PyLBFGSBConfig as LBFGSBConfig,
        PyLBFGSBFTerminator as LBFGSBFTerminator, PyLBFGSBGTerminator as LBFGSBGTerminator,
        PyLBFGSBInfNormGTerminator as LBFGSBInfNormGTerminator, PyMCMCSummary as MCMCSummary,
        PyMaxSteps as MaxSteps, PyMinimizationSummary as MinimizationSummary,
        PyMoreThuenteLineSearch as MoreThuenteLineSearch, PyMultiStartSummary as MultiStartSummary,
        PyNelderMeadConfig as NelderMeadConfig, PyNelderMeadFTerminator as NelderMeadFTerminator,
        PyNelderMeadXTerminator as NelderMeadXTerminator, PyPSOConfig as PSOConfig,
        PyPeriodicTransform as PeriodicTransform, PyProgressObserver as ProgressObserver,
        PyScaleTransform as ScaleTransform, PySimulatedAnnealingConfig as SimulatedAnnealingConfig,
        PySimulatedAnnealingTerminator as SimulatedAnnealingTerminator,
        PyStatusMessage as StatusMessage, PyStatusSnapshot as StatusSnapshot,
        PyTransformChain as TransformChain, PyTrustRegionConfig as TrustRegionConfig,
        PyTrustRegionGTerminator as TrustRegionGTerminator, PyVectorInit as VectorInit,
    };
}
