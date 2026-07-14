/// Useful implementations of [`AbortSignal`](`crate::traits::AbortSignal`).
pub mod abort_signals;
pub use abort_signals::{AtomicAbortSignal, CtrlCAbortSignal};

/// [`Callbacks`] and some other implementors of [`Terminator`](`crate::traits::Terminator`) and [`Observer`](`crate::traits::Observer`).
pub mod callbacks;
pub use callbacks::{Callbacks, DebugObserver, MaxSteps, ProgressObserver};

/// Checkpoint helpers and signal-triggered checkpointing utilities.
pub mod checkpoints;
pub use checkpoints::{
    AtomicCheckpointSignal, CheckpointAction, CheckpointOnSignal, CheckpointStore,
    CtrlCCheckpointSignal,
};

/// Diagnostics computed from retained MCMC chains.
pub mod mcmc_diagnostics;
pub use mcmc_diagnostics::MCMCDiagnostics;

/// Shared evaluation-count bookkeeping.
pub mod eval_counts;
pub use eval_counts::EvalCounts;

/// Scalar support shared by current and future generic optimizer APIs.
pub mod scalar;
pub use scalar::{RandomScalar, RealScalar};

/// Linear algebra provider traits and implementations.
pub mod linalg;
#[cfg(feature = "backend-ndarray")]
pub use linalg::NdArrayProvider;
pub use linalg::{
    Determinant, LinearAlgebra, LinearSolve, Matrix, NalgebraProvider, PseudoInverse, Scalar,
    SymmetricEigen, Vector,
};

/// [`Point`] type for defining a point in the parameter space.
pub mod point;
pub use point::{EvaluatedPoint, Point};

/// Summary types for the result of the minimization.
pub mod summary;
pub use summary::{
    HasParameterNames, MCMCSummary, MinimizationSummary, RenderedSummary, SummaryExport,
};

/// Multistart minimization orchestration helpers.
pub mod multistart;
pub use multistart::{
    minimize_multistart, restart_seed, FixedRestarts, MultiStartState, MultiStartSummary,
    RestartFactory, RestartPolicy,
};

/// Utility functions.
pub mod utils;
