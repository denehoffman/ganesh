/// Module containing the [`AbortSignal`] trait.
pub mod abort_signal;
pub use abort_signal::AbortSignal;

/// Module containing the [`Algorithm`], [`SupportsBounds`], and [`SupportsTransform`] traits.
pub mod algorithm;
pub use algorithm::{Algorithm, SupportsBounds, SupportsParameterNames, SupportsTransform};

/// Module containing the [`Observer`] and [`Terminator`] traits.
pub mod callback;
pub use callback::{Observer, Terminator};

/// Module containing the [`CheckpointableAlgorithm`] trait.
pub mod checkpoint;
pub use checkpoint::CheckpointableAlgorithm;

/// Legacy f64/nalgebra problem traits retained while built-in algorithms migrate.
pub mod cost_function;
#[doc(hidden)]
pub use cost_function::{
    CostFunction as LegacyCostFunction, GenericCostFunction, GenericGradient,
    Gradient as LegacyGradient, LogDensity as LegacyLogDensity,
};

/// Scalar- and backend-generic numerical problem traits.
pub mod numeric_problem;
pub use numeric_problem::{
    GradientHessian, ScalarCostFunction as CostFunction, ScalarGradient as Gradient,
    ScalarLogDensity as LogDensity, ValueGradientHessian,
};

/// Module containing various line-search methods.
pub mod linesearch;
pub use linesearch::{
    BackendLineSearch as LineSearch, BackendLineSearchOutput as LineSearchOutput,
};
#[doc(hidden)]
pub use linesearch::{BackendLineSearch, BackendLineSearchOutput, BackendLineSearchResult};
#[doc(hidden)]
pub use linesearch::{LineSearch as LegacyLineSearch, LineSearchOutput as LegacyLineSearchOutput};

/// Module containing the [`Status`] trait.
pub mod status;
pub use status::{ProgressStatus, Status, StatusMessage, StatusType};

/// Module containing the [`BoundLike`] trait and the [`Bound`] enum.
pub mod boundlike;
pub use boundlike::{Bound, BoundLike};

/// Module containing the [`Transform`] trait.
pub mod transform;
pub use transform::{Transform, TransformExt, TransformedProblem};

/// Scalar- and backend-generic coordinate transforms.
pub mod backend_transform;
pub use backend_transform::{
    BackendBounds, BackendScaleTransform, BackendTransform, BackendTransformedProblem,
    IdentityTransform, ScalarBound,
};
