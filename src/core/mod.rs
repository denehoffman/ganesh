/// Useful implementations of [`AbortSignal`](`crate::traits::AbortSignal`).
pub mod abort_signals;
pub use abort_signals::{AtomicAbortSignal, CtrlCAbortSignal};

/// [`Bounds`] and other implementations of [`Transform`](`crate::traits::Transform`)
pub mod transforms;
pub use transforms::Bounds;

/// [`Callbacks`] and some other implementors of [`Terminator`](`crate::traits::Terminator`) and [`Observer`](`crate::traits::Observer`).
pub mod callbacks;
pub use callbacks::{Callbacks, DebugObserver, MaxSteps, ProgressObserver};

/// Checkpoint helpers and signal-triggered checkpointing utilities.
pub mod checkpoints;
pub use checkpoints::{
    AtomicCheckpointSignal, CheckpointAction, CheckpointOnSignal, CheckpointStore,
    CtrlCCheckpointSignal,
};

/// [`Point`] type for defining a point in the parameter space.
pub mod point;
pub use point::Point;

/// Summary types for the result of the minimization.
pub mod summary;
pub use summary::{
    HasParameterNames, MCMCSummary, MinimizationSummary, RenderedSummary,
    SimulatedAnnealingSummary, SummaryExport,
};

/// Multistart minimization orchestration helpers.
pub mod multistart;
pub use multistart::{
    FixedRestarts, MultiStartState, MultiStartSummary, RestartFactory, RestartPolicy,
    minimize_multistart, restart_seed,
};

/// Utility functions.
pub mod utils;
