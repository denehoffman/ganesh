use crate::{
    core::Callbacks,
    traits::{Algorithm, Status},
};
use std::convert::Infallible;

/// A trait for algorithms which can save and restore step-boundary checkpoints.
pub trait CheckpointableAlgorithm<P, S: Status, U = (), E = Infallible>: Algorithm<P, S, U, E> {
    /// The checkpoint type used to resume the algorithm.
    type Checkpoint: Clone;

    /// Create a checkpoint that resumes at `next_step`.
    fn checkpoint(&self, status: &S, next_step: usize) -> Self::Checkpoint;

    /// Restore the algorithm and status from a prior checkpoint.
    ///
    /// The supplied config should be compatible with the original run. Algorithms may recompute
    /// derived internal state from it during restore.
    fn restore(&mut self, checkpoint: &Self::Checkpoint, config: &Self::Config) -> (S, usize);

    /// Continue processing from a prior checkpoint.
    ///
    /// This resumes from the saved `next_step` boundary recorded in the checkpoint rather than
    /// rerunning initialization.
    fn process_from_checkpoint<C>(
        &mut self,
        problem: &P,
        args: &U,
        config: Self::Config,
        checkpoint: &Self::Checkpoint,
        callbacks: C,
    ) -> Result<Self::Summary, E>
    where
        C: Into<Callbacks<Self, P, S, U, E, Self::Config>>,
        Self: Sized,
    {
        let (mut status, mut current_step) = self.restore(checkpoint, &config);
        let mut cbs: Callbacks<Self, P, S, U, E, Self::Config> = callbacks.into();
        loop {
            self.step(current_step, problem, &mut status, args, &config)?;

            if cbs
                .check_for_termination(current_step, self, problem, &mut status, args, &config)
                .is_break()
            {
                break;
            }
            current_step += 1;
        }
        self.postprocessing(problem, &mut status, args, &config)?;
        self.summarize(current_step, problem, &status, args, &config)
    }
}
