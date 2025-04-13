use crate::core::Status;

use super::CostFunction;

/// A trait representing a minimization algorithm.
///
/// This trait is implemented for the algorithms found in the [`solvers`](super) module, and contains
/// all the methods needed to be run by a [`Minimizer`].
pub trait Solver<U, E> {
    /// Any setup work done before the main steps of the algorithm should be done here.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    fn initialize(
        &mut self,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E>;
    /// The main "step" of an algorithm, which is repeated until termination conditions are met or
    /// the max number of steps have been taken.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    fn step(
        &mut self,
        i_step: usize,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E>;
    /// Runs any termination/convergence checks and returns true if the algorithm has converged.
    /// Developers should also update the internal [`Status`] of the algorithm here if converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    fn check_for_termination(
        &mut self,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<bool, E>;
    /// Runs any steps needed by the [`Solver`] after termination or convergence. This will run
    /// regardless of whether the [`Solver`] converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    #[allow(unused_variables)]
    fn postprocessing(
        &mut self,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        Ok(())
    }
}
