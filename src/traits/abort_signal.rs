use crate::traits::{Algorithm, Status, Terminator};
use std::ops::ControlFlow;

/// A trait for abort signals.
/// This trait is used in minimizers to check if the user has requested to abort the calculation.
pub trait AbortSignal {
    /// Return `true` if the user has requested to abort the calculation.
    fn is_aborted(&self) -> bool;
    /// Abort the calculation. Make `is_aborted()` return `true`.
    fn abort(&self);
    /// Reset the abort signal. Make `is_aborted()` return `false`.
    fn reset(&self);
}

impl<T, A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for T
where
    T: AbortSignal,
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        _status: &mut S,
        _args: &U,
        _config: &C,
    ) -> ControlFlow<()> {
        if self.is_aborted() {
            self.reset();
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}
