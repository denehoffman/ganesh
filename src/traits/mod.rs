/// Module containing the [`AbortSignal`] trait.
pub mod abort_signal;
pub use abort_signal::AbortSignal;

/// Module containing the [`Algorithm`] and configuration helper traits.
pub mod algorithm;
pub use algorithm::{Algorithm, SupportsParameterNames};

/// Module containing the [`Observer`] and [`Terminator`] traits.
pub mod callback;
pub use callback::{Observer, Terminator};

/// Module containing the [`CheckpointableAlgorithm`] trait.
pub mod checkpoint;
pub use checkpoint::CheckpointableAlgorithm;

/// Scalar- and linear-algebra-generic numerical problem traits.
pub mod numeric_problem;
pub use numeric_problem::{
    GradientHessian, ScalarCostFunction as CostFunction, ScalarGradient as Gradient,
    ScalarLogDensity as LogDensity, ValueGradientHessian,
};

/// Module containing various line-search methods.
pub mod linesearch;
pub use linesearch::{LineSearch, LineSearchOutput, LineSearchResult};

/// Module containing the [`Status`] trait.
pub mod status;
pub use status::{ProgressStatus, Status, StatusMessage, StatusType};

/// Scalar- and linear-algebra-generic coordinate transforms.
pub mod transform;
pub use transform::{
    Bounds, IdentityTransform, ScalarBound, ScaleTransform, Transform, TransformedProblem,
};
