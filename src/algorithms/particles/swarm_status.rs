use serde::{Deserialize, Serialize};

use crate::{
    core::{Bounds, Point},
    traits::Status,
};

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
    pub fn get_best(&self, bounds: Option<&Bounds>) -> Point {
        if matches!(self.swarm.boundary_method, SwarmBoundaryMethod::Transform) {
            self.gbest.to_bounded(bounds)
        } else {
            self.gbest.clone()
        }
    }

    /// Convenience method to configure the swarm.
    pub fn on_swarm<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce(&mut Swarm) -> &mut Swarm,
    {
        f(&mut self.swarm);
        self
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
