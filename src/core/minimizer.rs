use std::sync::Arc;

use parking_lot::RwLock;

use crate::traits::{AbortSignal, CostFunction, Observer, Solver, Status};

use super::{Config, NopAbortSignal, Summary};

/// The main struct used for running [`Solver`]s on [`CostFunction`]s.
pub struct Minimizer<S, U, E> {
    /// The [`Status`] of the [`Solver`], usually read after minimization.
    pub status: S,
    solver: Box<dyn Solver<S, U, E>>,
    observers: Vec<Arc<RwLock<dyn Observer<S, U>>>>,
    abort_signal: Box<dyn AbortSignal>,
    config: Config,
    user_data: U,
    /// The [`Summary`] of the [`Solver`], usually read after minimization.
    pub result: Option<Summary>,
}

impl<S: Status, U: Default, E> Minimizer<S, U, E> {
    /// Creates a new [`Minimizer`] with the given (boxed) [`Solver`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new(solver: Box<dyn Solver<S, U, E>>, dimension: usize) -> Self {
        Self {
            status: S::default(),
            config: Config {
                dimension,
                ..Default::default()
            },
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

    /// Edit the [`Config`] of the [`Minimizer`].
    pub fn on_config<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(&mut Config) -> &mut Config,
    {
        f(&mut self.config);
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
        self.solver
            .initialize(func, &self.config, &mut self.status, &mut self.user_data)?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step <= self.config.max_steps
            && !observer_termination
            && !self.solver.check_for_termination(
                func,
                &self.config,
                &mut self.status,
                &mut self.user_data,
            )?
            && !self.abort_signal.is_aborted()
        {
            self.solver.step(
                current_step,
                func,
                &self.config,
                &mut self.status,
                &mut self.user_data,
            )?;
            current_step += 1;
            if !self.observers.is_empty() {
                for observer in self.observers.iter_mut() {
                    observer_termination = observer.write().callback(
                        current_step,
                        &mut self.status,
                        &mut self.user_data,
                    ) || observer_termination;
                }
            }
        }
        self.solver
            .postprocessing(func, &self.config, &mut self.status, &mut self.user_data)?;
        if current_step > self.config.max_steps && !self.status.converged() {
            self.status.update_message("MAX EVALS");
        }
        if self.abort_signal.is_aborted() {
            self.status.update_message("Abort signal received");
        }
        self.result = self
            .solver
            .summarize(func, &self.config, &self.status, &self.user_data)
            .ok();
        Ok(())
    }
}
