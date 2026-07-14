//! Generic particle optimization algorithms.

/// Particle swarm optimization.
pub mod pso;
pub use pso::{
    PSOConfig, SwarmParticle, SwarmPositionInitializer, SwarmTopology, SwarmUpdateMethod,
    SwarmVelocityInitializer, PSO,
};
