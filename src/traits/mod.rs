/// Module containing the [`AbortSignal`] trait and some useful implementations.
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

pub use abort_signal::{AbortSignal, AtomicAbortSignal, CtrlCAbortSignal};
pub use algorithm::{Algorithm, Bounded};
pub use callback::{Callback, Callbacks, DebugObserver, MaxSteps, Observer, Terminator};
pub use cost_function::{CostFunction, Gradient, LogDensity};
pub use linesearch::LineSearch;
pub use status::Status;
