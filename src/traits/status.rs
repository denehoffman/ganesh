use serde::{de::DeserializeOwned, Serialize};

/// A trait which holds the status of a [`Solver`](crate::traits::Solver) and has to be implemented for own [`Solver`](crate::traits::Solver)s that need
/// different status information than the ones implemented in this crate.
pub trait Status: Clone + Default + Serialize + DeserializeOwned {
    /// Resets the status to its default state. This is called at the beginning of every
    /// [`Minimizer`](crate::core::Minimizer) run. Only members that are not persistent between runs should be reset.
    /// For example, the initial parameters of the minimization should not be reset.
    fn reset(&mut self);
    /// Returns the convergence flag of the minimization.
    fn converged(&self) -> bool;
    /// Returns the message of the minimization.
    fn message(&self) -> &str;
    /// Sets the message of the minimization.
    fn update_message(&mut self, message: &str);
}
