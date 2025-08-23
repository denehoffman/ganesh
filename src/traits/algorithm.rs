use crate::{
    core::{Bound, Bounds},
    traits::{cost_function::Updatable, Callback, Status},
};
use parking_lot::RwLock;
use std::{convert::Infallible, sync::Arc};

/// A trait representing a minimization algorithm.
///
/// This trait is implemented for the algorithms found in the [`solvers`](super) module, and contains
/// all the methods needed to be run by a [`Engine`](crate::core::Engine).
pub trait Algorithm<P, S: Status, U = (), E = Infallible>
where
    P: Updatable<U, E>,
{
    /// A type which holds a summary of the algorithm's ending state.
    type Summary;
    /// The configuration struct for the algorithm.
    type Config;

    /// Any setup work done before the main steps of the algorithm should be done here.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    fn initialize(
        &mut self,
        config: Self::Config,
        problem: &P,
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
        current_step: usize,
        problem: &P,
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

    /// Runs any steps needed by the [`Algorithm`] after termination or convergence. This will run
    /// regardless of whether the [`Algorithm`] converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    #[allow(unused_variables)]
    fn postprocessing(&mut self, problem: &P, status: &mut S, user_data: &mut U) -> Result<(), E> {
        Ok(())
    }

    /// Generates a new [`Algorithm::Summary`] from the current state of the [`Algorithm`], which can be displayed or used elsewhere.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation fails while creating the [`Algorithm::Summary`].
    /// See [`CostFunction::evaluate`] for more information.
    #[allow(unused_variables)]
    fn summarize(
        &self,
        current_step: usize,
        problem: &P,
        status: &S,
        user_data: &U,
    ) -> Result<Self::Summary, E>;

    /// Reset the algorithm to its initial state.
    fn reset(&mut self) {}

    fn process(
        &mut self,
        problem: &mut P,
        user_data: &mut U,
        config: Self::Config,
        callbacks: &[Arc<RwLock<dyn Callback<Self, P, S, U, E>>>],
    ) -> Result<Self::Summary, E>
    where
        Self: Sized,
    {
        let mut status = S::default();
        self.initialize(config, problem, &mut status, user_data)?;
        let mut current_step = 0;
        loop {
            self.step(current_step, problem, &mut status, user_data)?;

            if callbacks.iter().any(|callback| {
                callback
                    .write()
                    .callback(current_step, self, problem, &mut status, user_data)
                    .is_break()
            }) {
                break;
            }
            problem.update(user_data)?;
            current_step += 1;
        }
        self.postprocessing(problem, &mut status, user_data)?;
        self.summarize(current_step, problem, &status, user_data)
    }
}

/// A trait which can be implemented on the configuration structs of [`Algorithm`](`crate::traits::Algorithm`)s to imply that the algorithm can be run with parameter bounds.
pub trait Bounded
where
    Self: Sized,
{
    /// A helper method to get the mutable internal [`Bounds`] object.
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds>;
    /// Sets all [`Bound`]s used by the [`Algorithm`]. This can be [`None`] for an unbounded problem, or
    /// [`Some`] [`Vec<(T, T)>`] with length equal to the number of free parameters. Individual
    /// upper or lower bounds can be unbounded by setting them equal to `T::infinity()` or
    /// `T::neg_infinity()` (e.g. `f64::INFINITY` and `f64::NEG_INFINITY`).
    fn with_bounds<I: IntoIterator<Item = B>, B: Into<Bound>>(mut self, bounds: I) -> Self {
        let bounds = bounds
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>()
            .into();
        *self.get_bounds_mut() = Some(bounds);
        self
    }
}
