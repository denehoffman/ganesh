pub mod backend_lbfgsb;
/// Implementation of the L-BFGS-B algorithm.
pub mod lbfgsb;
#[doc(hidden)]
pub use backend_lbfgsb::{BackendLBFGSB, BackendLBFGSBConfig};
pub use backend_lbfgsb::{BackendLBFGSB as LBFGSB, BackendLBFGSBConfig as LBFGSBConfig};
#[doc(hidden)]
pub use lbfgsb::{LBFGSBConfig as LegacyLBFGSBConfig, LBFGSB as LegacyLBFGSB};

/// Implementation of the Adam algorithm.
pub mod adam;
#[doc(hidden)]
pub use adam::{Adam as LegacyAdam, AdamConfig as LegacyAdamConfig};
#[doc(hidden)]
pub use adam::{BackendAdam, BackendAdamConfig};
pub use adam::{BackendAdam as Adam, BackendAdamConfig as AdamConfig};

/// Implementation of the nonlinear conjugate-gradient algorithm.
pub mod conjugate_gradient;
#[doc(hidden)]
pub use conjugate_gradient::{BackendConjugateGradient, BackendConjugateGradientConfig};
pub use conjugate_gradient::{
    BackendConjugateGradient as ConjugateGradient,
    BackendConjugateGradientConfig as ConjugateGradientConfig, ConjugateGradientUpdate,
};
#[doc(hidden)]
pub use conjugate_gradient::{
    ConjugateGradient as LegacyConjugateGradient,
    ConjugateGradientConfig as LegacyConjugateGradientConfig,
    ConjugateGradientGTerminator as LegacyConjugateGradientGTerminator,
};

/// Implementation of the trust-region algorithm.
pub mod trust_region;
pub use crate::prototype::scalar::{
    TrustRegion, TrustRegion as BackendTrustRegion, TrustRegionConfig,
    TrustRegionConfig as BackendTrustRegionConfig, TrustRegionSubproblem,
};
#[doc(hidden)]
pub use trust_region::{
    TrustRegion as LegacyTrustRegion, TrustRegionConfig as LegacyTrustRegionConfig,
    TrustRegionGTerminator as LegacyTrustRegionGTerminator,
    TrustRegionSubproblem as LegacyTrustRegionSubproblem,
};

/// [`GradientStatus`] type for gradient-based minimizers.
pub mod gradient_status;
#[doc(hidden)]
pub use gradient_status::BackendGradientStatus;
pub use gradient_status::BackendGradientStatus as GradientStatus;
#[doc(hidden)]
pub use gradient_status::GradientStatus as LegacyGradientStatus;
