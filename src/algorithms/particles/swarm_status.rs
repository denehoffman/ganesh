use serde::{Deserialize, Serialize};

use crate::{core::Point, traits::Status};

use super::{swarm::SwarmBoundaryMethod, Swarm};

/// A status for particle swarm optimization and similar methods.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct SwarmStatus {
    /// The global best position found by all particles (in unbounded space)
    pub gbest: Point,
    /// An indicator of whether the swarm has converged
    pub converged: bool,
    /// A message containing information about the condition of the swarm or convergence
    pub message: String,
    /// The swarm
    pub swarm: Swarm,
}

impl SwarmStatus {
    /// Get the global best position found by the swarm. If the boundary method is set to
    /// [`SwarmBoundaryMethod::Transform`], this will return the position in the original bounded space.
    pub fn get_best(&self) -> Point {
        if matches!(self.swarm.boundary_method, SwarmBoundaryMethod::Transform) {
            self.gbest.to_bounded(self.swarm.bounds.as_ref())
        } else {
            self.gbest.clone()
        }
    }
}

impl Status for SwarmStatus {
    fn reset(&mut self) {
        self.converged = false;
        self.message = String::new();
        self.gbest = Point::default();
        self.swarm.particles = vec![];
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn message(&self) -> &str {
        &self.message
    }
    fn update_message(&mut self, message: &str) {
        self.message = message.to_string();
    }
}
