/// Implementation of the Nelder-Mead simplex algorithm
pub mod nelder_mead;
pub use nelder_mead::{NelderMead, NelderMeadConfig};

/// [`GradientFreeStatus`] type for gradient-free minimizers.
pub mod gradient_free_status;
pub use gradient_free_status::GradientFreeStatus;

/// [`SimulatedAnnealing`] type for simulated annealing minimizers.
pub mod simulated_annealing;
pub use simulated_annealing::{
    SimulatedAnnealing, SimulatedAnnealingConfig, SimulatedAnnealingGenerator,
    SimulatedAnnealingStatus,
};

/// [`CMAES`] type for covariance-matrix adaptation evolution strategy minimizers.
pub mod cmaes;
pub use cmaes::{
    CMAESConditionCovTerminator, CMAESConfig, CMAESEqualFunValuesTerminator,
    CMAESNoEffectAxisTerminator, CMAESNoEffectCoordTerminator, CMAESSigmaTerminator,
    CMAESStagnationTerminator, CMAESTolFunTerminator, CMAESTolXTerminator, CMAESTolXUpTerminator,
    CMAES,
};

/// [`DifferentialEvolution`] type for differential evolution minimizers.
pub mod differential_evolution;
pub use differential_evolution::{DifferentialEvolution, DifferentialEvolutionConfig};
