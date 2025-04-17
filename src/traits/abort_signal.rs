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
