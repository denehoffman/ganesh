//! Native module entrypoint for the optional mixed Python package build.

use pyo3::{
    pyfunction, pymodule,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult,
};

use super::{
    errors::register_exceptions,
    status::{
        register_status_types, PyEnsembleStatus, PyGradientFreeStatus, PyGradientStatus,
        PySimulatedAnnealingStatus, PySwarmStatus,
    },
    summary::{
        register_summary_types, restore_mcmc_summary, restore_minimization_summary,
        restore_multistart_summary, restore_simulated_annealing_summary,
    },
};
use crate::{
    algorithms::{
        gradient::GradientStatus,
        gradient_free::{GradientFreeStatus, SimulatedAnnealingStatus},
        mcmc::{ChainStorageMode, EnsembleStatus, Walker},
        particles::{
            Swarm, SwarmBoundaryMethod, SwarmParticle, SwarmPositionInitializer, SwarmStatus,
            SwarmTopology, SwarmUpdateMethod, SwarmVelocityInitializer,
        },
    },
    core::{
        transforms::Bounds, MCMCSummary, MinimizationSummary, MultiStartSummary, Point,
        SimulatedAnnealingSummary,
    },
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
        n_f_evals: 10,
        n_g_evals: 4,
        n_h_evals: 0,
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
        n_f_evals: 8,
        n_g_evals: 0,
        n_h_evals: 0,
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
        n_f_evals: 42,
        n_g_evals: 0,
        n_h_evals: 0,
    }
}

#[pyfunction]
fn _testing_sample_multistart_summary() -> MultiStartSummary {
    let best = MinimizationSummary {
        bounds: Some(Bounds::new_default([
            (Some(-1.0), Some(1.0)),
            (None, Some(2.0)),
        ])),
        parameter_names: Some(vec!["alpha".into(), "beta".into()]),
        message: StatusMessage::default().set_success_with_message("best"),
        x0: DVector::from_vec(vec![1.0, 2.0]),
        x: DVector::from_vec(vec![0.5, 1.5]),
        std: DVector::from_vec(vec![0.1, 0.2]),
        fx: 1.25,
        n_f_evals: 10,
        n_g_evals: 4,
        n_h_evals: 0,
        covariance: DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
    };
    let other = MinimizationSummary {
        fx: 2.0,
        ..best.clone()
    };
    MultiStartSummary {
        runs: vec![other, best],
        best_run_index: Some(1),
        restart_count: 1,
    }
}

#[pyfunction]
fn _restore_minimization_summary(state: Vec<u8>) -> PyResult<crate::python::PyMinimizationSummary> {
    restore_minimization_summary(&state)
}

#[pyfunction]
fn _restore_mcmc_summary(state: Vec<u8>) -> PyResult<crate::python::PyMCMCSummary> {
    restore_mcmc_summary(&state)
}

#[pyfunction]
fn _restore_multistart_summary(state: Vec<u8>) -> PyResult<crate::python::PyMultiStartSummary> {
    restore_multistart_summary(&state)
}

#[pyfunction]
fn _restore_simulated_annealing_summary(
    state: Vec<u8>,
) -> PyResult<crate::python::PySimulatedAnnealingSummary> {
    restore_simulated_annealing_summary(&state)
}

#[pyfunction]
fn _testing_sample_gradient_status() -> PyGradientStatus {
    PyGradientStatus::from(GradientStatus {
        message: StatusMessage::default().set_step_with_message("iterating"),
        x: DVector::from_vec(vec![0.25, 0.75]),
        fx: 0.5,
        n_f_evals: 12,
        n_g_evals: 7,
        n_h_evals: 2,
        hess: Some(DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0])),
        cov: Some(DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.25])),
        err: Some(DVector::from_vec(vec![
            std::f64::consts::FRAC_1_SQRT_2,
            0.5,
        ])),
    })
}

#[pyfunction]
fn _testing_sample_gradient_free_status() -> PyGradientFreeStatus {
    PyGradientFreeStatus::from(GradientFreeStatus {
        message: StatusMessage::default().set_step_with_message("simplex updated"),
        x: DVector::from_vec(vec![1.0, -1.0]),
        fx: 2.5,
        n_f_evals: 18,
        hess: Some(DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.1, 4.0])),
        cov: Some(DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.25])),
        err: Some(DVector::from_vec(vec![1.0, 0.5])),
    })
}

#[pyfunction]
fn _testing_sample_ensemble_status() -> PyEnsembleStatus {
    let mut first = Walker::new(DVector::from_vec(vec![0.0, 1.0]));
    first.push(Point {
        x: DVector::from_vec(vec![0.5, 1.5]),
        fx: Some(-0.5),
    });
    let mut second = Walker::new(DVector::from_vec(vec![1.0, 0.0]));
    second.push(Point {
        x: DVector::from_vec(vec![1.5, 0.5]),
        fx: Some(-0.25),
    });
    PyEnsembleStatus::from(EnsembleStatus {
        walkers: vec![first, second],
        message: StatusMessage::default().set_initialized_with_message("sampling"),
        n_f_evals: 14,
        n_g_evals: 0,
    })
}

#[pyfunction]
fn _testing_sample_swarm_status() -> PySwarmStatus {
    let best = Point {
        x: DVector::from_vec(vec![0.25, -0.25]),
        fx: Some(0.125),
    };
    let other = Point {
        x: DVector::from_vec(vec![1.0, 1.5]),
        fx: Some(2.0),
    };
    let mut swarm = Swarm::new(SwarmPositionInitializer::RandomInLimits {
        bounds: vec![(-1.0, 1.0), (-2.0, 2.0)],
        n_particles: 2,
    });
    swarm.topology = SwarmTopology::Ring;
    swarm.update_method = SwarmUpdateMethod::Asynchronous;
    swarm.boundary_method = SwarmBoundaryMethod::Shr;
    swarm.velocity_initializer =
        SwarmVelocityInitializer::RandomInLimits(vec![(-0.5, 0.5), (-0.5, 0.5)]);
    swarm.particles = vec![
        SwarmParticle {
            position: best.clone(),
            velocity: DVector::from_vec(vec![0.1, -0.1]),
            best: best.clone(),
        },
        SwarmParticle {
            position: other.clone(),
            velocity: DVector::from_vec(vec![-0.2, 0.05]),
            best: other.clone(),
        },
    ];
    PySwarmStatus::from(SwarmStatus {
        gbest: best,
        initial_gbest: other,
        message: StatusMessage::default().set_step_with_message("swarm moved"),
        swarm,
        n_f_evals: 22,
    })
}

#[pyfunction]
fn _testing_sample_simulated_annealing_status() -> PySimulatedAnnealingStatus {
    PySimulatedAnnealingStatus::from(SimulatedAnnealingStatus {
        temperature: 0.75,
        initial: Point {
            x: DVector::from_vec(vec![1.5, -0.5]),
            fx: Some(1.0),
        },
        best: Point {
            x: DVector::from_vec(vec![0.25, 0.5]),
            fx: Some(0.125),
        },
        current: Point {
            x: DVector::from_vec(vec![0.5, 0.75]),
            fx: Some(0.25),
        },
        message: StatusMessage::default().set_step_with_message("cooling"),
        n_f_evals: 33,
    })
}

#[pymodule]
pub fn _ganesh(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_exceptions(module)?;
    register_status_types(module)?;
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
    module.add_function(wrap_pyfunction!(
        _testing_sample_multistart_summary,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_restore_minimization_summary, module)?)?;
    module.add_function(wrap_pyfunction!(_restore_mcmc_summary, module)?)?;
    module.add_function(wrap_pyfunction!(_restore_multistart_summary, module)?)?;
    module.add_function(wrap_pyfunction!(
        _restore_simulated_annealing_summary,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_testing_sample_gradient_status, module)?)?;
    module.add_function(wrap_pyfunction!(
        _testing_sample_gradient_free_status,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_testing_sample_ensemble_status, module)?)?;
    module.add_function(wrap_pyfunction!(_testing_sample_swarm_status, module)?)?;
    module.add_function(wrap_pyfunction!(
        _testing_sample_simulated_annealing_status,
        module
    )?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
