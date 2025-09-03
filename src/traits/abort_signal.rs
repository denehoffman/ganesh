use crate::traits::{Algorithm, Callback, Status};
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

impl<T, A, P, S, U, E> Callback<A, P, S, U, E> for T
where
    T: AbortSignal,
    A: Algorithm<P, S, U, E>,
    S: Status,
{
    fn callback(
        &mut self,
        _current_step: usize,
        _algorithm: &mut A,
        _problem: &mut P,
        _status: &mut S,
        _user_data: &mut U,
    ) -> ControlFlow<()> {
        if self.is_aborted() {
            self.reset();
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}
