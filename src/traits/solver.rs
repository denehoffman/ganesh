use crate::core::{Bounds, Summary};

use super::CostFunction;

/// A trait representing a minimization algorithm.
///
/// This trait is implemented for the algorithms found in the [`solvers`](super) module, and contains
/// all the methods needed to be run by a [`Minimizer`](crate::core::Minimizer).
pub trait Solver<S, U, E> {
    /// Any setup work done before the main steps of the algorithm should be done here.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    fn initialize(
        &mut self,
        func: &dyn CostFunction<U, E>,
        bounds: Option<&Bounds>,
        status: &mut S,
        user_data: &mut U,
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
        bounds: Option<&Bounds>,
        status: &mut S,
        user_data: &mut U,
    ) -> Result<(), E>;

    /*
    TODO: replace this with a terminator trait. There should be some basic traits:
    - max iterations terminator, which just checks if the max iterations have been reached
    - convergence terminator, which checks if the cost function has converged
    - gradient terminator, which checks if the gradient has converged or is small enough
    - time terminator, which checks if the time has been reached
    The minimizer should hold a vector of terminators and check them after each step.
    A lambda function with the correct parameters should implement the trait by default.
    */

    /// Runs any termination/convergence checks and returns true if the algorithm has converged.
    /// Developers should also update the internal [`Status`](crate::traits::Status) of the algorithm here if converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    fn check_for_termination(
        &mut self,
        func: &dyn CostFunction<U, E>,
        bounds: Option<&Bounds>,
        status: &mut S,
        user_data: &mut U,
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
        bounds: Option<&Bounds>,
        status: &mut S,
        user_data: &mut U,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Generates a new [`Summary`] from the current state of the [`Solver`], which can be displayed or used elsewhere.
    #[allow(unused_variables)]
    fn summarize(
        &self,
        func: &dyn CostFunction<U, E>,
        bounds: Option<&Bounds>,
        parameter_names: Option<&Vec<String>>,
        status: &S,
        user_data: &U,
    ) -> Result<Summary, E>;
}
