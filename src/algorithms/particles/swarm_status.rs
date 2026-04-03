use crate::{
    algorithms::particles::{Swarm, SwarmPositionInitializer},
    core::Point,
    traits::{ProgressStatus, Status, StatusMessage},
    DVector, Float,
};
use serde::{Deserialize, Serialize};

/// A status for particle swarm optimization and similar methods.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SwarmStatus {
    /// The global best position found by all particles
    pub gbest: Point<DVector<Float>>,
    /// The global best position among the initial swarm before any updates.
    pub initial_gbest: Point<DVector<Float>>,
    /// A message containing information about the condition of the swarm or convergence
    pub message: StatusMessage,
    /// The swarm
    pub swarm: Swarm,
    /// The number of function evaluations (approximately, this is left up to individual
    /// [`Algorithm`](crate::traits::Algorithm)s to correctly compute and may not be exact).
    pub n_f_evals: usize,
}
impl Default for SwarmStatus {
    fn default() -> Self {
        Self {
            gbest: Default::default(),
            initial_gbest: Default::default(),
            message: Default::default(),
            swarm: Swarm::new(SwarmPositionInitializer::Custom(Vec::default())),
            n_f_evals: Default::default(),
        }
    }
}

impl SwarmStatus {
    /// Get the global best position found by the swarm.
    pub fn get_best(&self) -> Point<DVector<Float>> {
        self.gbest.clone()
    }
}

impl Status for SwarmStatus {
    fn reset(&mut self) {
        self.gbest = Default::default();
        self.initial_gbest = Default::default();
        self.message = Default::default();
        self.swarm.particles = Default::default();
        self.n_f_evals = Default::default();
    }

    fn message(&self) -> &StatusMessage {
        &self.message
    }

    fn set_message(&mut self) -> &mut StatusMessage {
        &mut self.message
    }
}

impl ProgressStatus for SwarmStatus {
    fn write_progress(&self, out: &mut String) -> std::fmt::Result {
        use std::fmt::Write;
        write!(
            out,
            "status={} gbest_fx={} n_f_evals={}",
            self.message,
            self.gbest.fx.unwrap_or(Float::NAN),
            self.n_f_evals
        )
    }
}
