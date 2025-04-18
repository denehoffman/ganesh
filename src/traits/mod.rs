/// Module containing the [`AbortSignal`] trait.
pub mod abort_signal;
/// Module containing the [`Algorithm`] trait.
pub mod algorithm;
/// Module containing the [`CostFunction`] trait.
pub mod cost_function;
/// Module containing various line-search methods.
pub mod linesearch;
/// Module containing the [`Observer`] trait and its implementations.
pub mod observer;
/// Module containing the [`Status`] trait and its implementations.
pub mod status;

pub use abort_signal::AbortSignal;
pub use algorithm::Algorithm;
pub use cost_function::{CostFunction, Gradient, Hessian};
pub use linesearch::LineSearch;
pub use observer::Observer;
pub use status::Status;
