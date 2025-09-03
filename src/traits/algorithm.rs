use crate::{
    core::{Bound, Bounds, Callbacks},
    traits::{Callback, Status},
};
use std::convert::Infallible;

/// A trait representing an [`Algorithm`] which can be used to solve a problem `P`.
///
/// This trait is implemented for the algorithms found in the [`algorithms`](`crate::algorithms`) module and contains
/// all the methods needed to [`process`](`Algorithm::process`) a problem.
pub trait Algorithm<P, S: Status, U = (), E = Infallible> {
    /// A type which holds a summary of the algorithm's ending state.
    type Summary;
    /// The configuration struct for the algorithm.
    type Config;

    /// Any setup work done before the main steps of the algorithm should be done here.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    fn initialize(
        &mut self,
        config: Self::Config,
        problem: &mut P,
        status: &mut S,
        user_data: &mut U,
    ) -> Result<(), E>;
    /// The main "step" of an algorithm, which is repeated until termination conditions are met or
    /// the max number of steps have been taken.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    fn step(
        &mut self,
        current_step: usize,
        problem: &mut P,
        status: &mut S,
        user_data: &mut U,
    ) -> Result<(), E>;

    /// Runs any steps needed by the [`Algorithm`] after termination or convergence. This will run
    /// regardless of whether the [`Algorithm`] converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    #[allow(unused_variables)]
    fn postprocessing(&mut self, problem: &P, status: &mut S, user_data: &mut U) -> Result<(), E> {
        Ok(())
    }

    /// Generates a new [`Algorithm::Summary`] from the current state of the [`Algorithm`], which can be displayed or used elsewhere.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
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

    /// Process the given problem using this [`Algorithm`].
    ///
    /// This method first runs [`Algorithm::initialize`], then runs [`Algorithm::step`] in a loop,
    /// terminating if any supplied [`Callback`]s return
    /// [`ControlFlow::Break`](`std::ops::ControlFlow::Break`). Finally, regardless of convergence,
    /// [`Algorithm::postprocessing`] is called. [`Algorithm::summarize`] is called to create a
    /// summary of the [`Algorithm`]'s state.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    fn process<C>(
        &mut self,
        problem: &mut P,
        user_data: &mut U,
        config: Self::Config,
        callbacks: C,
    ) -> Result<Self::Summary, E>
    where
        C: Into<Callbacks<Self, P, S, U, E>>,
        Self: Sized,
    {
        let mut status = S::default();
        let mut cbs: Callbacks<Self, P, S, U, E> = callbacks.into();
        self.initialize(config, problem, &mut status, user_data)?;
        let mut current_step = 0;
        loop {
            self.step(current_step, problem, &mut status, user_data)?;

            if cbs
                .callback(current_step, self, problem, &mut status, user_data)
                .is_break()
            {
                break;
            }
            current_step += 1;
        }
        self.postprocessing(problem, &mut status, user_data)?;
        self.summarize(current_step, problem, &status, user_data)
    }

    /// Provides a set of reasonable default callbacks specific to this [`Algorithm`].
    fn default_callbacks() -> Callbacks<Self, P, S, U, E>
    where
        Self: Sized,
    {
        Callbacks::empty()
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
