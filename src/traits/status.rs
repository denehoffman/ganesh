use serde::{de::DeserializeOwned, Serialize};

/// A trait which holds the status of a [`Solver`] and has to be implemented for own [`Solver`]s that need
/// different status information than the ones implemented in this crate.
pub trait Status: Clone + Default + Serialize + DeserializeOwned {
    /// Resets the status to its default state. This is called at the beginning of every
    /// [`Solver`] run.
    /// Take care that this does not reset the [`Config`] struct or any other information that is
    /// needed to be kept between runs.
    fn reset(&mut self);
    /// Returns the convergence flag of the minimization.
    fn converged(&self) -> bool;
    /// Returns the message of the minimization.
    fn message(&self) -> &str;
    /// Sets the message of the minimization.
    fn update_message(&mut self, message: &str);
}
