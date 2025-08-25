/// Module containing the [`AbortSignal`] trait.
pub mod abort_signal;
/// Module containing the [`Algorithm`] and [`Bounded`] traits.
pub mod algorithm;
/// Module containing the [`Callback`] trait and its implementations.
pub mod callback;
/// Module containing the [`CostFunction`] trait.
pub mod cost_function;
/// Module containing various line-search methods.
pub mod linesearch;
/// Module containing the [`Status`] trait and its implementations.
pub mod status;

pub use abort_signal::AbortSignal;
pub use algorithm::{Algorithm, Bounded};
pub use callback::Callback;
pub use cost_function::{CostFunction, Gradient};
pub use linesearch::LineSearch;
pub use status::Status;
