//! Multistart minimization orchestration helpers.

use crate::{
    core::{Callbacks, MinimizationSummary},
    traits::{Algorithm, Status},
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

/// Lightweight state exposed to restart factories and policies during multistart orchestration.
#[derive(Debug, Clone, Default)]
pub struct MultiStartState {
    runs: Vec<MinimizationSummary>,
}

impl MultiStartState {
    /// Create a new empty multistart state.
    pub const fn new() -> Self {
        Self { runs: Vec::new() }
    }

    /// Get the completed run summaries gathered so far.
    pub fn runs(&self) -> &[MinimizationSummary] {
        &self.runs
    }

    /// Get the number of completed runs.
    pub fn completed_runs(&self) -> usize {
        self.runs.len()
    }

    /// Get the number of completed restarts, counting the first run separately.
    pub fn restart_count(&self) -> usize {
        self.runs.len().saturating_sub(1)
    }

    /// Get the best run summary seen so far.
    pub fn best(&self) -> Option<&MinimizationSummary> {
        self.best_index().map(|index| &self.runs[index])
    }

    /// Get the index of the best run seen so far.
    pub fn best_index(&self) -> Option<usize> {
        self.runs
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.fx.total_cmp(&b.fx))
            .map(|(index, _)| index)
    }

    pub(crate) fn push(&mut self, summary: MinimizationSummary) {
        self.runs.push(summary);
    }
}

/// Final summary returned by multistart minimization orchestration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiStartSummary {
    /// The summary for each completed run.
    pub runs: Vec<MinimizationSummary>,
    /// The index of the best run in [`MultiStartSummary::runs`], if any runs completed.
    pub best_run_index: Option<usize>,
    /// The number of completed restarts, counting the first run separately.
    pub restart_count: usize,
}

impl MultiStartSummary {
    /// Get the best run summary.
    pub fn best(&self) -> Option<&MinimizationSummary> {
        self.best_run_index.map(|index| &self.runs[index])
    }

    /// Get the number of completed runs.
    pub fn completed_runs(&self) -> usize {
        self.runs.len()
    }
}

/// A policy that decides whether another multistart run should be launched.
pub trait RestartPolicy: Send {
    /// Return `true` if the run with `next_run_index` should be launched.
    fn should_run(&mut self, next_run_index: usize, state: &MultiStartState) -> bool;
}

impl<F> RestartPolicy for F
where
    F: FnMut(usize, &MultiStartState) -> bool + Send,
{
    fn should_run(&mut self, next_run_index: usize, state: &MultiStartState) -> bool {
        self(next_run_index, state)
    }
}

/// A simple fixed-run restart policy.
#[derive(Debug, Clone, Copy)]
pub struct FixedRestarts {
    total_runs: usize,
}

impl FixedRestarts {
    /// Create a policy that runs the minimizer exactly `total_runs` times.
    pub const fn new(total_runs: usize) -> Self {
        Self { total_runs }
    }
}

impl RestartPolicy for FixedRestarts {
    fn should_run(&mut self, next_run_index: usize, _state: &MultiStartState) -> bool {
        next_run_index < self.total_runs
    }
}

/// Produces the algorithm, config, and callbacks for each multistart run.
pub trait RestartFactory<A, P, S: Status, U = (), E = Infallible>: Send
where
    A: Algorithm<P, S, U, E, Summary = MinimizationSummary>,
{
    /// Create the next run bundle for the given `run_index`.
    fn create(
        &mut self,
        run_index: usize,
        state: &MultiStartState,
    ) -> (A, A::Init, A::Config, Callbacks<A, P, S, U, E, A::Config>);
}

impl<A, P, S, U, E, F> RestartFactory<A, P, S, U, E> for F
where
    S: Status,
    A: Algorithm<P, S, U, E, Summary = MinimizationSummary>,
    F: FnMut(
            usize,
            &MultiStartState,
        ) -> (A, A::Init, A::Config, Callbacks<A, P, S, U, E, A::Config>)
        + Send,
{
    fn create(
        &mut self,
        run_index: usize,
        state: &MultiStartState,
    ) -> (A, A::Init, A::Config, Callbacks<A, P, S, U, E, A::Config>) {
        self(run_index, state)
    }
}

/// Deterministically derive a per-run seed from a base seed and run index.
pub const fn restart_seed(base_seed: u64, run_index: usize) -> u64 {
    base_seed.wrapping_add(run_index as u64)
}

/// Run a deterministic multistart minimization workflow.
///
/// Each run is created by `restart_factory`, and `restart_policy` decides whether the next run
/// should be launched based on the current [`MultiStartState`].
pub fn minimize_multistart<P, U, E, A, S, F, R>(
    problem: &P,
    user_data: &U,
    restart_factory: &mut F,
    restart_policy: &mut R,
) -> Result<MultiStartSummary, E>
where
    S: Status,
    A: Algorithm<P, S, U, E, Summary = MinimizationSummary>,
    F: RestartFactory<A, P, S, U, E>,
    R: RestartPolicy,
{
    let mut state = MultiStartState::new();
    while restart_policy.should_run(state.completed_runs(), &state) {
        let run_index = state.completed_runs();
        let (mut algorithm, init, config, callbacks) = restart_factory.create(run_index, &state);
        let summary = algorithm.process(problem, user_data, init, config, callbacks)?;
        state.push(summary);
    }

    let best_run_index = state.best_index();
    Ok(MultiStartSummary {
        restart_count: state.restart_count(),
        runs: state.runs,
        best_run_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{Callbacks, MaxSteps},
        traits::{Status, StatusMessage},
        DMatrix, DVector, Float,
    };

    #[derive(Clone, Default, Serialize, Deserialize)]
    struct DummyStatus {
        message: StatusMessage,
    }

    impl Status for DummyStatus {
        fn reset(&mut self) {
            self.message = StatusMessage::default();
        }

        fn message(&self) -> &StatusMessage {
            &self.message
        }

        fn set_message(&mut self) -> &mut StatusMessage {
            &mut self.message
        }
    }

    #[derive(Clone, Default)]
    struct DummyAlgorithm;

    #[derive(Clone)]
    struct DummyConfig {
        x: DVector<Float>,
        fx: Float,
    }

    impl Algorithm<(), DummyStatus, (), Infallible> for DummyAlgorithm {
        type Summary = MinimizationSummary;
        type Config = DummyConfig;
        type Init = DummyConfig;

        fn initialize(
            &mut self,
            _problem: &(),
            status: &mut DummyStatus,
            _args: &(),
            _init: &Self::Init,
            _config: &Self::Config,
        ) -> Result<(), Infallible> {
            status.set_message().initialize();
            Ok(())
        }

        fn step(
            &mut self,
            _current_step: usize,
            _problem: &(),
            status: &mut DummyStatus,
            _args: &(),
            _config: &Self::Config,
        ) -> Result<(), Infallible> {
            status.set_message().succeed_with_message("done");
            Ok(())
        }

        fn summarize(
            &self,
            _current_step: usize,
            _problem: &(),
            status: &DummyStatus,
            _args: &(),
            _init: &Self::Init,
            config: &Self::Config,
        ) -> Result<Self::Summary, Infallible> {
            Ok(MinimizationSummary {
                bounds: None,
                parameter_names: None,
                message: status.message.clone(),
                x0: config.x.clone(),
                x: config.x.clone(),
                std: DVector::zeros(config.x.len()),
                fx: config.fx,
                cost_evals: 1,
                gradient_evals: 0,
                covariance: DMatrix::identity(config.x.len(), config.x.len()),
            })
        }

        fn default_callbacks() -> Callbacks<Self, (), DummyStatus, (), Infallible, Self::Config> {
            Callbacks::empty().with_terminator(MaxSteps(0))
        }
    }

    #[test]
    fn fixed_restarts_runs_expected_number_of_times_and_tracks_best() {
        let mut factory = |run_index: usize, _state: &MultiStartState| {
            (
                DummyAlgorithm,
                DummyConfig {
                    x: DVector::from_element(1, run_index as Float),
                    fx: (3 - run_index) as Float,
                },
                DummyConfig {
                    x: DVector::from_element(1, run_index as Float),
                    fx: (3 - run_index) as Float,
                },
                DummyAlgorithm::default_callbacks(),
            )
        };
        let mut policy = FixedRestarts::new(3);

        let summary = minimize_multistart::<(), (), Infallible, DummyAlgorithm, DummyStatus, _, _>(
            &(),
            &(),
            &mut factory,
            &mut policy,
        )
        .unwrap();

        assert_eq!(summary.completed_runs(), 3);
        assert_eq!(summary.restart_count, 2);
        assert_eq!(summary.best_run_index, Some(2));
        assert_eq!(summary.best().unwrap().fx, 1.0);
    }

    #[test]
    fn closure_restart_policy_can_stop_based_on_seen_runs() {
        let mut factory = |run_index: usize, _state: &MultiStartState| {
            (
                DummyAlgorithm,
                DummyConfig {
                    x: DVector::from_element(1, run_index as Float),
                    fx: run_index as Float,
                },
                DummyConfig {
                    x: DVector::from_element(1, run_index as Float),
                    fx: run_index as Float,
                },
                DummyAlgorithm::default_callbacks(),
            )
        };
        let mut policy = |_: usize, state: &MultiStartState| state.completed_runs() < 2;

        let summary = minimize_multistart::<(), (), Infallible, DummyAlgorithm, DummyStatus, _, _>(
            &(),
            &(),
            &mut factory,
            &mut policy,
        )
        .unwrap();

        assert_eq!(summary.completed_runs(), 2);
        assert_eq!(summary.restart_count, 1);
    }

    #[test]
    fn restart_seed_is_deterministic() {
        assert_eq!(restart_seed(7, 0), 7);
        assert_eq!(restart_seed(7, 3), 10);
    }

    #[test]
    fn zero_run_policy_returns_empty_summary() {
        let mut factory = |run_index: usize, _state: &MultiStartState| {
            (
                DummyAlgorithm,
                DummyConfig {
                    x: DVector::from_element(1, run_index as Float),
                    fx: run_index as Float,
                },
                DummyConfig {
                    x: DVector::from_element(1, run_index as Float),
                    fx: run_index as Float,
                },
                DummyAlgorithm::default_callbacks(),
            )
        };
        let mut policy = FixedRestarts::new(0);

        let summary = minimize_multistart::<(), (), Infallible, DummyAlgorithm, DummyStatus, _, _>(
            &(),
            &(),
            &mut factory,
            &mut policy,
        )
        .unwrap();

        assert_eq!(summary.completed_runs(), 0);
        assert_eq!(summary.restart_count, 0);
        assert_eq!(summary.best_run_index, None);
        assert!(summary.best().is_none());
    }
}
