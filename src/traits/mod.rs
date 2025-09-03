/// Module containing the [`AbortSignal`] trait.
pub mod abort_signal;
pub use abort_signal::AbortSignal;

/// Module containing the [`Algorithm`] and [`Bounded`] traits.
pub mod algorithm;
pub use algorithm::{Algorithm, Bounded};

/// Module containing the [`Callback`] trait.
pub mod callback;
pub use callback::{Callback, Observer, Terminator};

/// Module containing the [`CostFunction`] trait.
pub mod cost_function;
pub use cost_function::{CostFunction, Gradient, LogDensity};

/// Module containing various line-search methods.
pub mod linesearch;
pub use linesearch::LineSearch;

/// Module containing the [`Status`] trait.
pub mod status;
pub use status::Status;

/// Module containing the [`Boundable`] trait.
pub mod boundable;
pub use boundable::Boundable;
