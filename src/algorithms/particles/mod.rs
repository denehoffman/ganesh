use crate::{
    core::EvaluatedPoint,
    traits::{LegacyCostFunction, Observer},
    DVector, Float,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub mod backend_pso;
/// Implementation of Particle Swarm Optimization (PSO) algorithm
pub mod pso;
#[doc(hidden)]
pub use backend_pso::{BackendPSO, BackendPSOConfig};
pub use backend_pso::{BackendPSO as PSO, BackendPSOConfig as PSOConfig};
#[doc(hidden)]
pub use pso::{PSOConfig as LegacyPSOConfig, PSO as LegacyPSO};

/// [`Swarm`] type for swarm-based optimizers.
pub mod swarm;
pub use swarm::{
    Swarm, SwarmBoundaryMethod, SwarmParticle, SwarmPositionInitializer, SwarmTopology,
    SwarmUpdateMethod, SwarmVelocityInitializer,
};

/// [`SwarmStatus`] type for swarm-based optimizers.
pub mod swarm_status;
pub use crate::algorithms::gradient_free::GradientFreeStatus as SwarmStatus;
#[doc(hidden)]
pub use swarm_status::SwarmStatus as LegacySwarmStatus;

/// An [`Observer`] which stores the swarm particles' history as well as the
/// history of global best positions.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrackingSwarmObserver {
    /// The history of the swarm particles
    pub history: Vec<Vec<SwarmParticle>>,
    /// The history of the best position in the swarm
    pub best_history: Vec<EvaluatedPoint<DVector<Float>>>,
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

impl<P, U, E> Observer<LegacyPSO, P, LegacySwarmStatus, U, E, LegacyPSOConfig>
    for TrackingSwarmObserver
where
    P: LegacyCostFunction<U, E>,
{
    fn observe(
        &mut self,
        _current_step: usize,
        _algorithm: &LegacyPSO,
        _problem: &P,
        status: &LegacySwarmStatus,
        _args: &U,
        _config: &LegacyPSOConfig,
    ) {
        self.history.push(status.swarm.particles.clone());
        self.best_history.push(status.gbest.clone());
    }
}
