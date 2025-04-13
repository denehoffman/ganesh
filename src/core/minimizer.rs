use std::{fmt::Display, sync::Arc};

use parking_lot::RwLock;

use crate::traits::{AbortSignal, CostFunction, Observer, Solver, Status};

use super::{Config, NopAbortSignal};

/// The main struct used for running [`Solver`]s on [`Function`]s.
pub struct Minimizer<S: Status, U: Default, E> {
    /// The [`Status`] of the [`Problem`], usually read after minimization.
    pub status: S,
    solver: Box<dyn Solver<S, U, E>>,
    observers: Vec<Arc<RwLock<dyn Observer<S, U>>>>,
    abort_signal: Box<dyn AbortSignal>,
    config: Config,
    user_data: U,
}
impl<S: Status, U: Default, E> Display for Minimizer<S, U, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.status)
    }
}
impl<S: Status, U: Default, E> Minimizer<S, U, E> {
    const DEFAULT_MAX_STEPS: usize = 4000;
    /// Creates a new [`Problem`] with the given (boxed) [`Solver`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new(solver: Box<dyn Solver<S, U, E>>, dimension: usize) -> Self {
        Self {
            status: S::default(),
            config: Config {
                dimension,
                bounds: None,
                parameter_names: None,
                max_steps: Self::DEFAULT_MAX_STEPS,
            },
            solver,
            observers: Vec::default(),
            abort_signal: NopAbortSignal.boxed(),
            user_data: Default::default(),
        }
    }
    pub fn reset_status(&mut self) {
        self.status.reset();
    }

    pub fn on_config<F>(mut self, mut f: F) -> Self
    where
        F: FnMut(Config) -> Config,
    {
        self.config = f(self.config);
        self
    }

    pub fn on_status<F>(mut self, mut f: F) -> Self
    where
        F: FnMut(S) -> S,
    {
        self.status = f(self.status);
        self
    }

    pub fn update_config<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(Config) -> Config,
    {
        self.config = f(self.config.clone());
        self
    }

    pub fn update_status<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(S) -> S,
    {
        self.status = f(self.status.clone());
        self
    }

    pub fn with_abort_signal(mut self, abort_signal: Box<dyn AbortSignal>) -> Self {
        self.abort_signal = abort_signal;
        self
    }

    pub fn with_user_data<T: Into<U>>(mut self, data: T) -> Self {
        self.user_data = data.into();
        self
    }

    pub fn update_user_data<T: Into<U>>(&mut self, data: T) -> &mut Self {
        self.user_data = data.into();
        self
    }

    /// Adds a single [`Observer`] to the [`Minimizer`].
    pub fn with_observer(mut self, observer: Arc<RwLock<dyn Observer<S, U>>>) -> Self {
        self.observers.push(observer);
        self
    }
    /// Minimize the given [`Function`] starting at the point `x0`.
    ///
    /// This method first runs [`Algorithm::initialize`], then runs [`Algorithm::step`] in a loop,
    /// terminating if [`Algorithm::check_for_termination`] returns `true` or if
    /// the maximum number of allowed steps is exceeded. Each step will be followed by a sequential
    /// call to all given [`Observer`]s' callback functions. Finally, regardless of convergence,
    /// [`Algorithm::postprocessing`] is called. If the algorithm did not converge in the given
    /// step limit, the [`Status::message`] will be set to `"MAX EVALS"` at termination.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    ///
    /// # Panics
    ///
    /// This method will panic if the length of `x0` is not equal to the dimension of the problem
    /// (number of free parameters) or if any values of `x0` are outside the [`Bound`]s given to the
    /// [`Minimizer`].
    pub fn minimize(&mut self, func: &dyn CostFunction<U, E>) -> Result<(), E> {
        self.reset_status();
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
        Ok(())
    }
}
