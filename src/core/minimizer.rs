use std::sync::Arc;

use parking_lot::RwLock;

use crate::traits::{AbortSignal, CostFunction, Observer, Solver, Status};

use super::{Bound, NopAbortSignal, Summary};

const DEFAULT_MAX_STEPS: usize = 4000;
/// The main struct used for running [`Solver`]s on [`CostFunction`]s.
pub struct Minimizer<S, U, E> {
    /// The [`Status`] of the [`Solver`], usually read after minimization.
    pub status: S,
    /// The [`Summary`] of the [`Solver`], usually read after minimization.
    pub result: Option<Summary>,

    solver: Box<dyn Solver<S, U, E>>,
    observers: Vec<Arc<RwLock<dyn Observer<S, U>>>>,
    abort_signal: Box<dyn AbortSignal>,
    user_data: U,

    bounds: Option<Vec<Bound>>,
    parameter_names: Option<Vec<String>>,
    max_steps: usize,
}

impl<S: Status, U: Default, E> Minimizer<S, U, E> {
    /// Creates a new [`Minimizer`] with the given (boxed) [`Solver`].
    pub fn new(solver: Box<dyn Solver<S, U, E>>) -> Self {
        Self {
            status: S::default(),
            bounds: None,
            parameter_names: None,
            max_steps: DEFAULT_MAX_STEPS,
            solver,
            observers: Vec::default(),
            abort_signal: NopAbortSignal.boxed(),
            user_data: Default::default(),
            result: None,
        }
    }

    /// Convenience method to use chainable methods to set up the [`Minimizer`].
    /// Example usage:
    /// ```rust
    /// let solver = LBFGSB::default();
    /// let mut m = Minimizer::new(Box::new(solver), 2)
    ///   .setup(|m| {
    ///     m.on_config(|c|
    ///       c.with_bounds(vec![(-4.0, 4.0), (-4.0, 4.0)]))
    ///     .with_abort_signal(CtrlCAbortSignal::new().boxed())
    ///   });
    /// ```
    pub fn setup<F>(mut self, mut f: F) -> Self
    where
        F: FnMut(&mut Self) -> &mut Self,
    {
        f(&mut self);
        self
    }

    /// Edit the [`Status`] of the [`Minimizer`].
    pub fn on_status<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(&mut S) -> &mut S,
    {
        f(&mut self.status);
        self
    }

    /// Sets all [`Bound`]s of the [`Config`] used by the [`Solver`](crate::traits::Solver). This can be [`None`] for an unbounded problem, or
    /// [`Some`] [`Vec<(T, T)>`] with length equal to the number of free parameters. Individual
    /// upper or lower bounds can be unbounded by setting them equal to `T::infinity()` or
    /// `T::neg_infinity()` (e.g. `f64::INFINITY` and `f64::NEG_INFINITY`).
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bounds is not equal to the number of free
    /// parameters.
    pub fn with_bounds<I: IntoIterator<Item = B>, B: Into<Bound>>(
        &mut self,
        bounds: I,
    ) -> &mut Self {
        let bounds = bounds.into_iter().map(Into::into).collect::<Vec<Bound>>();
        self.bounds = Some(bounds);
        self
    }

    /// Sets the names of the parameters. This is only used for printing and debugging purposes.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bounds is not equal to the number of free
    /// parameters.
    pub fn with_parameter_names<I: IntoIterator<Item = String>>(&mut self, names: I) -> &mut Self {
        let names = names.into_iter().collect::<Vec<String>>();
        self.parameter_names = Some(names);
        self
    }
    /// Set the maximum number of steps to perform before failure (default: 4000).
    pub fn with_max_steps(&mut self, max_steps: usize) -> &mut Self {
        self.max_steps = max_steps;
        self
    }
    /// Set the [`AbortSignal`] of the [`Minimizer`].
    pub fn with_abort_signal(&mut self, abort_signal: Box<dyn AbortSignal>) -> &mut Self {
        self.abort_signal = abort_signal;
        self
    }

    /// Set user data for the [`Minimizer`].
    pub fn with_user_data<T: Into<U>>(&mut self, data: T) -> &mut Self {
        self.user_data = data.into();
        self
    }

    /// Adds a single [`Observer`] to the [`Minimizer`].
    pub fn add_observer(&mut self, observer: Arc<RwLock<dyn Observer<S, U>>>) -> &mut Self {
        self.observers.push(observer);
        self
    }

    /// Check parameters against the bounds
    pub fn assert_parameters(&self, x: &[f64]) {
        if let Some(bounds) = &self.bounds {
            for (i, (x_i, bound_i)) in x.iter().zip(bounds).enumerate() {
                assert!(
                    bound_i.contains(*x_i),
                    "Parameter #{} = {} is outside of the given bound: {}",
                    i,
                    x_i,
                    bound_i
                )
            }
        }
    }

    /// Minimize the given [`CostFunction`] starting at the point `x0`.
    ///
    /// This method first runs [`Solver::initialize`], then runs [`Solver::step`] in a loop,
    /// terminating if [`Solver::check_for_termination`] returns `true` or if
    /// the maximum number of allowed steps is exceeded. Each step will be followed by a sequential
    /// call to all given [`Observer`]s' callback functions. Finally, regardless of convergence,
    /// [`Solver::postprocessing`] is called. Finally [`Solver::summarize`] is called to create a summary of the minimization run. If the algorithm did not converge in the given
    /// step limit, the [`Status::message`] will be set to `"MAX EVALS"` at termination.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    ///
    /// # Panics
    ///
    /// This method will panic if the length of `x0` is not equal to the dimension of the problem
    /// (number of free parameters) or if any values of `x0` are outside the [`Bound`](crate::core::Bound)s given to the
    /// [`Minimizer`].
    pub fn minimize(&mut self, func: &dyn CostFunction<U, E>) -> Result<(), E> {
        self.status.reset();
        self.abort_signal.reset();
        self.solver.initialize(
            func,
            self.bounds.as_ref(),
            &mut self.status,
            &mut self.user_data,
        )?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step <= self.max_steps
            && !observer_termination
            && !self.solver.check_for_termination(
                func,
                self.bounds.as_ref(),
                &mut self.status,
                &mut self.user_data,
            )?
            && !self.abort_signal.is_aborted()
        {
            self.solver.step(
                current_step,
                func,
                self.bounds.as_ref(),
                &mut self.status,
                &mut self.user_data,
            )?;
            current_step += 1;
            if !self.observers.is_empty() {
                for observer in self.observers.iter_mut() {
                    observer_termination = observer.write().callback(
                        current_step,
                        self.bounds.as_ref(),
                        &mut self.status,
                        &mut self.user_data,
                    ) || observer_termination;
                }
            }
        }
        self.solver.postprocessing(
            func,
            self.bounds.as_ref(),
            &mut self.status,
            &mut self.user_data,
        )?;
        if current_step > self.max_steps && !self.status.converged() {
            self.status.update_message("MAX EVALS");
        }
        if self.abort_signal.is_aborted() {
            self.status.update_message("Abort signal received");
        }
        self.result = self
            .solver
            .summarize(
                func,
                self.bounds.as_ref(),
                self.parameter_names.as_ref(),
                &self.status,
                &self.user_data,
            )
            .ok();
        Ok(())
    }
}
