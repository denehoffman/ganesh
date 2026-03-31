use crate::traits::{AbortSignal, CheckpointableAlgorithm, Status, Terminator};
use parking_lot::Mutex;
use std::{ops::ControlFlow, sync::Arc};

/// Action taken after a checkpoint is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CheckpointAction {
    /// Save the checkpoint and continue running.
    Continue,
    /// Save the checkpoint and stop the current run.
    #[default]
    Stop,
}

/// An in-memory checkpoint sink for callbacks and signal-triggered saves.
#[derive(Clone)]
pub struct CheckpointStore<T> {
    checkpoint: Arc<Mutex<Option<T>>>,
}

impl<T> Default for CheckpointStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> CheckpointStore<T> {
    /// Create an empty checkpoint store.
    pub fn new() -> Self {
        Self {
            checkpoint: Arc::new(Mutex::new(None)),
        }
    }

    /// Replace the stored checkpoint.
    pub fn save(&self, checkpoint: T) {
        *self.checkpoint.lock() = Some(checkpoint);
    }

    /// Return a clone of the latest saved checkpoint, if any.
    pub fn load(&self) -> Option<T>
    where
        T: Clone,
    {
        self.checkpoint.lock().clone()
    }

    /// Return `true` if a checkpoint is currently stored.
    pub fn has_checkpoint(&self) -> bool {
        self.checkpoint.lock().is_some()
    }
}

/// A terminator which checkpoints a run when an [`AbortSignal`] is triggered.
#[derive(Clone)]
pub struct CheckpointOnSignal<Sig, Sink> {
    signal: Sig,
    sink: Sink,
    action: CheckpointAction,
}

impl<Sig, Sink> CheckpointOnSignal<Sig, Sink> {
    /// Create a new signal-triggered checkpoint terminator.
    pub const fn new(signal: Sig, sink: Sink) -> Self {
        Self {
            signal,
            sink,
            action: CheckpointAction::Stop,
        }
    }

    /// Set whether checkpointing should stop or continue the run.
    pub const fn with_action(mut self, action: CheckpointAction) -> Self {
        self.action = action;
        self
    }
}

impl<A, P, S, U, E, C, Sig, Sink> Terminator<A, P, S, U, E, C> for CheckpointOnSignal<Sig, Sink>
where
    A: CheckpointableAlgorithm<P, S, U, E, Config = C>,
    S: Status,
    Sig: AbortSignal + Clone,
    Sink: FnMut(A::Checkpoint) + Clone + Send + Sync + 'static,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        _problem: &P,
        status: &mut S,
        _args: &U,
        _config: &C,
    ) -> ControlFlow<()> {
        if self.signal.is_aborted() {
            let checkpoint = algorithm.checkpoint(status, current_step.saturating_add(1));
            (self.sink)(checkpoint);
            self.signal.reset();
            status.set_message().custom("Checkpoint requested");
            if self.action == CheckpointAction::Stop {
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

/// A signal alias intended for checkpoint-triggered pause/resume workflows.
pub type AtomicCheckpointSignal = crate::core::AtomicAbortSignal;

/// A signal alias intended for checkpoint-triggered pause/resume workflows.
pub type CtrlCCheckpointSignal = crate::core::CtrlCAbortSignal;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_store_roundtrip() {
        let store = CheckpointStore::new();
        assert!(!store.has_checkpoint());
        store.save(7usize);
        assert!(store.has_checkpoint());
        assert_eq!(store.load(), Some(7));
    }
}
