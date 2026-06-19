/// Implementation of the L-BFGS-B algorithm.
pub mod lbfgsb;
pub use lbfgsb::{LBFGSBConfig, LBFGSB};

/// Implementation of the Adam algorithm.
pub mod adam;
pub use adam::{Adam, AdamConfig};

/// Implementation of the nonlinear Conjugate Gradient algorithm.
pub mod conjugate_gradient;
pub use conjugate_gradient::{
    ConjugateGradient, ConjugateGradientConfig, ConjugateGradientGTerminator,
    ConjugateGradientUpdate,
};

/// Implementation of the trust-region algorithm.
pub mod trust_region;
pub use trust_region::{
    TrustRegion, TrustRegionConfig, TrustRegionGTerminator, TrustRegionSubproblem,
};

/// [`GradientStatus`] type for gradient-based minimizers.
pub mod gradient_status;
pub use gradient_status::GradientStatus;
