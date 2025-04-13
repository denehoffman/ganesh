/// Module containing the [`AbortSignal`] trait.
pub mod abort_signal;
/// Module containing the [`CostFunction`] trait.
pub mod cost_function;
/// Module containing various line-search methods.
pub mod linesearch;
/// Module containing the [`Observer`] trait and its implementations.
pub mod observer;
/// Module containing the [`Solver`] trait.
pub mod solver;

pub use abort_signal::AbortSignal;
pub use cost_function::CostFunction;
pub use linesearch::LineSearch;
pub use observer::{MCMCObserver, Observer, SwarmObserver};
pub use solver::Solver;
