/// Implementation of Particle Swarm Optimization (PSO) algorithm
pub mod pso;
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
    core::Point,
    traits::{cost_function::Updatable, Algorithm, Callback},
};
use std::ops::ControlFlow;

/// An [`Observer`] which stores the swarm particles' history as well as the
/// history of global best positions.
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct TrackingSwarmObserver {
    /// The history of the swarm particles
    pub history: Vec<Vec<SwarmParticle>>,
    /// The history of the best position in the swarm
    pub best_history: Vec<Point>,
}

impl<A, P, U, E> Callback<A, P, SwarmStatus, U, E> for TrackingSwarmObserver
where
    A: Algorithm<P, SwarmStatus, U, E>,
    P: Updatable<U, E>,
{
    fn callback(
        &mut self,
        _current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        status: &mut SwarmStatus,
        _user_data: &mut U,
    ) -> ControlFlow<()> {
        self.history.push(status.swarm.particles.clone());
        self.best_history.push(status.get_best());
        ControlFlow::Continue(())
    }
}
