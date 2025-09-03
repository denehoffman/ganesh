use crate::{
    core::Point,
    traits::{Algorithm, Observer},
    DVector, Float,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Implementation of Particle Swarm Optimization (PSO) algorithm
pub mod pso;
pub use pso::{PSOConfig, PSO};

/// [`Swarm`] type for swarm-based optimizers.
pub mod swarm;
pub use swarm::{
    Swarm, SwarmBoundaryMethod, SwarmParticle, SwarmPositionInitializer, SwarmTopology,
    SwarmUpdateMethod, SwarmVelocityInitializer,
};

/// [`SwarmStatus`] type for swarm-based optimizers.
pub mod swarm_status;
pub use swarm_status::SwarmStatus;

/// An [`Observer`] which stores the swarm particles' history as well as the
/// history of global best positions.
#[derive(Serialize, Deserialize, Clone)]
pub struct TrackingSwarmObserver {
    /// The history of the swarm particles
    pub history: Vec<Vec<SwarmParticle>>,
    /// The history of the best position in the swarm
    pub best_history: Vec<Point<DVector<Float>>>,
}
impl TrackingSwarmObserver {
    /// Create a new observer to track the swarm history, wrapped in an [`Arc<Mutex<_>>`]
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            history: Vec::new(),
            best_history: Vec::new(),
        }))
    }
}

impl<A, P, U, E> Observer<A, P, SwarmStatus, U, E> for TrackingSwarmObserver
where
    A: Algorithm<P, SwarmStatus, U, E>,
{
    fn observe(
        &mut self,
        _current_step: usize,
        _algorithm: &A,
        _problem: &P,
        status: &SwarmStatus,
        _user_data: &U,
    ) {
        self.history.push(status.swarm.particles.clone());
        self.best_history.push(status.get_best());
    }
}
