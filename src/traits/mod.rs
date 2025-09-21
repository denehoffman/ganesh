/// Module containing the [`AbortSignal`] trait.
pub mod abort_signal;
pub use abort_signal::AbortSignal;

/// Module containing the [`Algorithm`], [`SupportsBounds`], and [`SupportsTransform`] traits.
pub mod algorithm;
pub use algorithm::{Algorithm, SupportsBounds, SupportsTransform};

/// Module containing the [`Observer`] and [`Terminator`] traits.
pub mod callback;
pub use callback::{Observer, Terminator};

/// Module containing the [`CostFunction`] trait.
pub mod cost_function;
pub use cost_function::{CostFunction, GenericCostFunction, GenericGradient, Gradient, LogDensity};

/// Module containing various line-search methods.
pub mod linesearch;
pub use linesearch::LineSearch;

/// Module containing the [`Status`] trait.
pub mod status;
pub use status::Status;

/// Module containing the [`BoundLike`] trait and the [`Bound`] enum.
pub mod boundlike;
pub use boundlike::{Bound, BoundLike};

/// Module containing the [`Transform`] trait.
pub mod transform;
pub use transform::{Transform, TransformExt, TransformedProblem};
