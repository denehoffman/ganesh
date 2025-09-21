/// Useful implementations of [`AbortSignal`](`crate::traits::AbortSignal`).
pub mod abort_signals;
pub use abort_signals::{AtomicAbortSignal, CtrlCAbortSignal};

/// [`Bounds`] and other implementations of [`Transform`](`crate::traits::Transform`)
pub mod transforms;
pub use transforms::Bounds;

/// [`Callbacks`] and some other implementors of [`Terminator`](`crate::traits::Terminator`) and [`Observer`](`crate::traits::Observer`).
pub mod callbacks;
pub use callbacks::{Callbacks, DebugObserver, MaxSteps};

/// [`Point`] type for defining a point in the parameter space.
pub mod point;
pub use point::Point;

/// Summary types for the result of the minimization.
pub mod summary;
pub use summary::{MCMCSummary, MinimizationSummary, SimulatedAnnealingSummary};

/// Utility functions.
pub mod utils;
