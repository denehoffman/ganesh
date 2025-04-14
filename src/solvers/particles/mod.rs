/// Implementation of Particle Swarm Optimization (PSO) algorithm
pub mod pso;
pub use pso::PSO;

/// [`SwarmStatus`] type for swarm-based optimizers.
pub mod swarm_status;
pub use swarm_status::{
    SwarmBoundaryMethod, SwarmParticle, SwarmPositionInitializer, SwarmStatus, SwarmTopology,
    SwarmUpdateMethod, SwarmVelocityInitializer,
};
