use crate::{
    algorithms::particles::{Swarm, SwarmPositionInitializer},
    core::{EvalCounts, EvaluatedPoint},
    traits::{ProgressStatus, Status, StatusMessage},
    DVector, Float,
};
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

/// A status for particle swarm optimization and similar methods.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwarmStatus {
    /// The global best position found by all particles
    pub gbest: EvaluatedPoint<DVector<Float>>,
    /// The global best position among the initial swarm before any updates.
    pub initial_gbest: EvaluatedPoint<DVector<Float>>,
    /// A message containing information about the condition of the swarm or convergence
    pub message: StatusMessage,
    /// The swarm
    pub swarm: Swarm,
    /// Evaluation counts requested by the algorithm API.
    #[serde(flatten)]
    pub evals: EvalCounts,
}
impl Default for SwarmStatus {
    fn default() -> Self {
        Self {
            gbest: Default::default(),
            initial_gbest: Default::default(),
            message: Default::default(),
            swarm: Swarm::new(SwarmPositionInitializer::Custom(Vec::default())),
            evals: Default::default(),
        }
    }
}

impl Status for SwarmStatus {
    fn reset(&mut self) {
        self.gbest = Default::default();
        self.initial_gbest = Default::default();
        self.message = Default::default();
        self.swarm.particles = Default::default();
        self.evals = Default::default();
    }

    fn message(&self) -> &StatusMessage {
        &self.message
    }

    fn set_message(&mut self) -> &mut StatusMessage {
        &mut self.message
    }

    fn check_invariants(&mut self) -> ControlFlow<()> {
        let invalid_global_best = !self.gbest.fx.is_finite() || !self.initial_gbest.fx.is_finite();
        let invalid_particle_best = self
            .swarm
            .particles
            .iter()
            .any(|particle| !particle.best.fx.is_finite());

        if invalid_global_best || invalid_particle_best {
            self.set_message().fail_with_message("f(x) is not finite");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

impl ProgressStatus for SwarmStatus {
    fn write_progress(&self, out: &mut String) -> std::fmt::Result {
        use std::fmt::Write;
        write!(
            out,
            "status={} gbest_fx={} n_f_evals={}",
            self.message,
            self.gbest.fx,
            self.evals.f()
        )
    }
}
