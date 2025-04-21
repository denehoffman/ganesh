/// Implementation of Particle Swarm Optimization (PSO) algorithm
pub mod pso;
use std::sync::Arc;

use parking_lot::RwLock;
pub use pso::PSO;

/// [`Swarm`] type for swarm-based optimizers.
pub mod swarm;
use serde::{Deserialize, Serialize};
pub use swarm::{
    Swarm, SwarmBoundaryMethod, SwarmParticle, SwarmPositionInitializer, SwarmTopology,
    SwarmUpdateMethod, SwarmVelocityInitializer,
};

/// [`SwarmStatus`] type for swarm-based optimizers.
pub mod swarm_status;
pub use swarm_status::SwarmStatus;

use crate::{
    core::{Bounds, Point},
    traits::Observer,
};

/// An [`Observer`] which stores the swarm particles' history as well as the
/// history of global best positions.
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct TrackingSwarmObserver {
    /// The history of the swarm particles
    pub history: Vec<Vec<SwarmParticle>>,
    /// The history of the best position in the swarm
    pub best_history: Vec<Point>,
}

impl TrackingSwarmObserver {
    /// Finalize the [`SwarmObserver`] by wrapping it in an [`Arc`] and [`RwLock`]
    pub fn build() -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self::default()))
    }
}

impl<U> Observer<SwarmStatus, U> for TrackingSwarmObserver {
    fn callback(
        &mut self,
        _step: usize,
        bounds: Option<&Bounds>,
        status: &mut SwarmStatus,
        _user_data: &mut U,
    ) -> bool {
        self.history.push(status.swarm.particles.clone());
        self.best_history.push(status.get_best(bounds));
        false
    }
}
