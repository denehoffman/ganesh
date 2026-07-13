pub mod backend_nelder_mead;
/// Implementation of the Nelder-Mead simplex algorithm
pub mod nelder_mead;
#[doc(hidden)]
pub use backend_nelder_mead::{BackendNelderMead, BackendNelderMeadConfig};
pub use backend_nelder_mead::{
    BackendNelderMead as NelderMead, BackendNelderMeadConfig as NelderMeadConfig,
};
#[doc(hidden)]
pub use nelder_mead::NelderMeadInit as LegacyNelderMeadInit;
#[doc(hidden)]
pub use nelder_mead::{NelderMead as LegacyNelderMead, NelderMeadConfig as LegacyNelderMeadConfig};

/// [`GradientFreeStatus`] type for gradient-free minimizers.
pub mod gradient_free_status;
#[doc(hidden)]
pub use gradient_free_status::BackendGradientFreeStatus;
pub use gradient_free_status::BackendGradientFreeStatus as GradientFreeStatus;
#[doc(hidden)]
pub use gradient_free_status::GradientFreeStatus as LegacyGradientFreeStatus;

pub mod backend_simulated_annealing;
/// [`SimulatedAnnealing`] type for simulated annealing minimizers.
pub mod simulated_annealing;
#[doc(hidden)]
pub use backend_simulated_annealing::{BackendSimulatedAnnealing, BackendSimulatedAnnealingConfig};
pub use backend_simulated_annealing::{
    BackendSimulatedAnnealing as SimulatedAnnealing,
    BackendSimulatedAnnealingConfig as SimulatedAnnealingConfig,
};
#[doc(hidden)]
pub use simulated_annealing::{
    SimulatedAnnealing as LegacySimulatedAnnealing,
    SimulatedAnnealingConfig as LegacySimulatedAnnealingConfig,
    SimulatedAnnealingGenerator as LegacySimulatedAnnealingGenerator,
    SimulatedAnnealingStatus as LegacySimulatedAnnealingStatus,
};

pub mod backend_cmaes;
/// [`CMAES`] type for covariance-matrix adaptation evolution strategy minimizers.
pub mod cmaes;
#[doc(hidden)]
pub use backend_cmaes::{BackendCMAES, BackendCMAESConfig};
pub use backend_cmaes::{BackendCMAES as CMAES, BackendCMAESConfig as CMAESConfig};
#[doc(hidden)]
pub use cmaes::{
    CMAESConditionCovTerminator, CMAESConfig as LegacyCMAESConfig, CMAESEqualFunValuesTerminator,
    CMAESInit as LegacyCMAESInit, CMAESNoEffectAxisTerminator, CMAESNoEffectCoordTerminator,
    CMAESSigmaTerminator, CMAESStagnationTerminator, CMAESTolFunTerminator, CMAESTolXTerminator,
    CMAESTolXUpTerminator, CMAES as LegacyCMAES,
};

/// [`DifferentialEvolution`] type for differential evolution minimizers.
pub mod differential_evolution;
pub use crate::prototype::scalar::{
    DifferentialEvolution, DifferentialEvolution as BackendDifferentialEvolution,
    DifferentialEvolutionConfig, DifferentialEvolutionConfig as BackendDifferentialEvolutionConfig,
};
#[doc(hidden)]
pub use differential_evolution::{
    DifferentialEvolution as LegacyDifferentialEvolution,
    DifferentialEvolutionConfig as LegacyDifferentialEvolutionConfig,
    DifferentialEvolutionInit as LegacyDifferentialEvolutionInit,
};
