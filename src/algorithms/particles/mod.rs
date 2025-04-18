/// Implementation of Particle Swarm Optimization (PSO) algorithm
pub mod pso;
pub use pso::PSO;

/// [`Swarm`] type for swarm-based optimizers.
pub mod swarm;
pub use swarm::{
    Swarm, SwarmBoundaryMethod, SwarmParticle, SwarmPositionInitializer, SwarmTopology,
    SwarmUpdateMethod, SwarmVelocityInitializer,
};

/// [`SwarmStatus`] type for swarm-based optimizers.
pub mod swarm_status;
pub use swarm_status::SwarmStatus;
