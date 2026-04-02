//! Native module entrypoint for the optional mixed Python package build.

use pyo3::{
    pyfunction, pymodule,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult,
};

use super::{errors::register_exceptions, summary::register_summary_types};
use crate::{
    algorithms::mcmc::ChainStorageMode,
    core::{transforms::Bounds, MCMCSummary, MinimizationSummary, SimulatedAnnealingSummary},
    traits::StatusMessage,
    DMatrix, DVector,
};

#[pyfunction]
fn _testing_sample_minimization_summary() -> MinimizationSummary {
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

#[pyfunction]
fn _testing_sample_mcmc_summary() -> MCMCSummary {
    MCMCSummary {
        bounds: Some(Bounds::new_default([(Some(-1.0), Some(1.0))])),
        parameter_names: Some(vec!["theta".into()]),
        message: StatusMessage::default().set_initialized_with_message("warmup"),
        chain: vec![vec![
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![0.5]),
        ]],
        chain_storage: ChainStorageMode::Rolling { window: 16 },
        cost_evals: 8,
        gradient_evals: 0,
        dimension: (1, 2, 1),
    }
}

#[pyfunction]
fn _testing_sample_simulated_annealing_summary() -> SimulatedAnnealingSummary<DVector<crate::Float>>
{
    SimulatedAnnealingSummary {
        bounds: Some(Bounds::new_default([
            (Some(-2.0), Some(2.0)),
            (None, Some(3.0)),
        ])),
        message: StatusMessage::default().set_success_with_message("cooled"),
        x0: DVector::from_vec(vec![1.5, -0.5]),
        x: DVector::from_vec(vec![0.25, 1.25]),
        fx: 0.125,
        cost_evals: 42,
    }
}

#[pymodule]
pub fn _ganesh(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_exceptions(module)?;
    register_summary_types(module)?;
    module.add_function(wrap_pyfunction!(
        _testing_sample_minimization_summary,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_testing_sample_mcmc_summary, module)?)?;
    module.add_function(wrap_pyfunction!(
        _testing_sample_simulated_annealing_summary,
        module
    )?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
