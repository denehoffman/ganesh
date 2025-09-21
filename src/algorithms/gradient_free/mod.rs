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
