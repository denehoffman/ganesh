use std::{fmt::Display, sync::Arc};

use nalgebra::DVector;
use parking_lot::RwLock;

use crate::{
    traits::{AbortSignal, CostFunction, Observer, Solver},
    Float,
};

use super::{Bound, NopAbortSignal, Status};

/// The main struct used for running [`Solver`]s on [`Function`]s.
pub struct Problem<U: Default, E> {
    /// The [`Status`] of the [`Problem`], usually read after minimization.
    pub status: Status,
    solver: Box<dyn Solver<U, E>>,
    observers: Vec<Arc<RwLock<dyn Observer<U>>>>,
    abort_signal: Box<dyn AbortSignal>,
    user_data: U,
}
impl<U: Default, E> Display for Problem<U, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.status)
    }
}
impl<U: Default, E> Problem<U, E> {
    const DEFAULT_MAX_STEPS: usize = 4000;
    /// Creates a new [`Problem`] with the given (boxed) [`Solver`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new(solver: Box<dyn Solver<U, E>>, dimension: usize) -> Self {
        Self {
            status: Status {
                max_steps: Self::DEFAULT_MAX_STEPS,
                dimension,
                ..Default::default()
            },
            solver,
            observers: Vec::default(),
            abort_signal: NopAbortSignal.boxed(),
            user_data: Default::default(),
        }
    }
    fn reset_status(&mut self) {
        let new_status = Status {
            bounds: self.status.bounds.clone(),
            max_steps: self.status.max_steps,
            x0: self.status.x0.clone(),
            parameter_names: self.status.parameter_names.clone(),
            dimension: self.status.dimension,
            ..Default::default()
        };
        self.status = new_status;
    }
    /// Sets all [`Bound`]s of the [`Problem`]. This can be [`None`] for an unbounded problem, or
    /// [`Some`] [`Vec<(T, T)>`] with length equal to the number of free parameters. Individual
    /// upper or lower bounds can be unbounded by setting them equal to `T::infinity()` or
    /// `T::neg_infinity()` (e.g. `f64::INFINITY` and `f64::NEG_INFINITY`).
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bounds is not equal to the number of free
    /// parameters.
    pub fn with_bounds<I: IntoIterator<Item = B>, B: Into<Bound>>(mut self, bounds: I) -> Self {
        let bounds = bounds.into_iter().map(Into::into).collect::<Vec<Bound>>();
        assert!(bounds.len() == self.status.dimension);
        self.status.bounds = Some(bounds);
        self
    }

    pub fn with_parameter_names<I: IntoIterator<Item = String>>(mut self, names: I) -> Self {
        let names = names.into_iter().collect::<Vec<String>>();
        assert!(names.len() == self.status.dimension);
        self.status.parameter_names = Some(names);
        self
    }

    pub fn with_abort_signal(mut self, abort_signal: Box<dyn AbortSignal>) -> Self {
        self.abort_signal = abort_signal;
        self
    }

    pub fn with_user_data<S: Into<U>>(mut self, data: S) -> Self {
        self.user_data = data.into();
        self
    }

    pub fn with_initial_guess<I: IntoIterator<Item = Float>>(mut self, x0: I) -> Self {
        let x0 = x0.into_iter().collect::<Vec<Float>>();
        assert!(x0.len() == self.status.dimension);
        self.status.x0 = DVector::from_column_slice(&x0);
        self
    }

    pub fn update_bounds<I: IntoIterator<Item = B>, B: Into<Bound>>(
        &mut self,
        bounds: I,
    ) -> &mut Self {
        let bounds = bounds.into_iter().map(Into::into).collect::<Vec<Bound>>();
        assert!(bounds.len() == self.status.dimension);
        self.status.bounds = Some(bounds);
        self
    }

    pub fn update_initial_guess<I: IntoIterator<Item = Float>>(&mut self, x0: I) -> &mut Self {
        let x0 = x0.into_iter().collect::<Vec<Float>>();
        assert!(x0.len() == self.status.dimension);
        self.status.x0 = DVector::from_column_slice(&x0);
        self
    }

    pub fn update_user_data<S: Into<U>>(&mut self, data: S) -> &mut Self {
        self.user_data = data.into();
        self
    }

    /// Set the maximum number of steps to perform before failure (default: 4000).
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.status.max_steps = max_steps;
        self
    }
    /// Adds a single [`Observer`] to the [`Minimizer`].
    pub fn with_observer(mut self, observer: Arc<RwLock<dyn Observer<U>>>) -> Self {
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
        if let Some(bounds) = &self.status.bounds {
            for (i, (x_i, bound_i)) in self.status.x0.iter().zip(bounds).enumerate() {
                assert!(
                    bound_i.contains(*x_i),
                    "Parameter #{} = {} is outside of the given bound: {}",
                    i,
                    x_i,
                    bound_i
                )
            }
        }
        self.solver
            .initialize(func, &mut self.user_data, &mut self.status)?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step <= self.status.max_steps
            && !observer_termination
            && !self
                .solver
                .check_for_termination(func, &mut self.user_data, &mut self.status)?
            && !self.abort_signal.is_aborted()
        {
            self.solver
                .step(current_step, func, &mut self.user_data, &mut self.status)?;
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
            .postprocessing(func, &mut self.user_data, &mut self.status)?;
        if current_step > self.status.max_steps && !self.status.converged {
            self.status.update_message("MAX EVALS");
        }
        if self.abort_signal.is_aborted() {
            self.status.update_message("Abort signal received");
        }
        Ok(())
    }
}
