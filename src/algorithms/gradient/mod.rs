/// L-BFGS-B minimization.
pub mod lbfgsb;
pub use lbfgsb::{
    LBFGSBCheckpoint, LBFGSBConfig, LBFGSBErrorMode, LBFGSBFTerminator, LBFGSBGTerminator,
    LBFGSBInfNormGTerminator, LBFGSB,
};

/// Adam minimization.
pub mod adam;
pub use adam::{Adam, AdamConfig, AdamEMATerminator};

/// Nonlinear conjugate-gradient minimization.
pub mod conjugate_gradient;
pub use conjugate_gradient::{
    ConjugateGradient, ConjugateGradientConfig, ConjugateGradientGTerminator,
    ConjugateGradientUpdate,
};

/// Trust-region minimization.
pub mod trust_region;
pub use trust_region::{
    TrustRegion, TrustRegionConfig, TrustRegionGTerminator, TrustRegionSubproblem,
};

/// Status used by gradient-based minimizers.
pub mod gradient_status;
pub use gradient_status::GradientStatus;
