use std::sync::Arc;

use parking_lot::RwLock;

use crate::{
    traits::{AbortSignal, Algorithm, CostFunction, Observer, Status},
    Float,
};

use super::{Bound, Bounds, NopAbortSignal};

const DEFAULT_MAX_STEPS: usize = 4000;
/// The main struct used for running [`Algorithm`]s on [`CostFunction`]s.
pub struct Engine<S, U, E, Summary> {
    /// The [`Status`] of the [`Algorithm`], usually read after minimization.
    pub status: S,
    /// The [`Algorithm::Summary`], usually read after minimization.
    pub result: Summary,

    algorithm: Box<dyn Algorithm<S, U, E, Summary = Summary>>,
    observers: Vec<Arc<RwLock<dyn Observer<S, U>>>>,
    abort_signal: Box<dyn AbortSignal>,
    user_data: U,

    bounds: Option<Bounds>,
    parameter_names: Option<Vec<String>>,
    max_steps: usize,
}

impl<S: Status, U: Default, E, Summary: Default> Engine<S, U, E, Summary> {
    /// Creates a new [`Engine`] with the given [`Algorithm`].
    pub fn new<T: Algorithm<S, U, E, Summary = Summary> + 'static>(solver: T) -> Self {
        Self {
            status: S::default(),
            bounds: None,
            parameter_names: None,
            max_steps: DEFAULT_MAX_STEPS,
            algorithm: Box::new(solver),
            observers: Vec::default(),
            abort_signal: Box::new(NopAbortSignal),
            user_data: Default::default(),
            result: Default::default(),
        }
    }

    /// Convenience method to use chainable methods to set up the [`Engine`].
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

    /// Edit the [`Status`] of the [`Engine`].
    pub fn on_status<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(&mut S) -> &mut S,
    {
        f(&mut self.status);
        self
    }

    /// Sets all [`Bound`]s used by the [`Algorithm`]. This can be [`None`] for an unbounded problem, or
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
        let bounds = bounds
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>()
            .into();
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
    /// Set the [`AbortSignal`] of the [`Engine`].
    pub fn with_abort_signal<A: AbortSignal + 'static>(&mut self, abort_signal: A) -> &mut Self {
        self.abort_signal = Box::new(abort_signal);
        self
    }

    /// Set user data for the [`Engine`].
    pub fn with_user_data<T: Into<U>>(&mut self, data: T) -> &mut Self {
        self.user_data = data.into();
        self
    }

    /// Adds a single [`Observer`] to the [`Engine`].
    pub fn add_observer(&mut self, observer: Arc<RwLock<dyn Observer<S, U>>>) -> &mut Self {
        self.observers.push(observer);
        self
    }

    /// Check parameters against the bounds
    ///
    /// # Panics
    ///
    /// This method will panic if any parameter is outside of its given bound.
    pub fn assert_parameters(&self, x: &[Float]) {
        if let Some(bounds) = &self.bounds {
            for (i, (x_i, bound_i)) in x.iter().zip(bounds.iter()).enumerate() {
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
    /// This method first runs [`Algorithm::initialize`], then runs [`Algorithm::step`] in a loop,
    /// terminating if [`Algorithm::check_for_termination`] returns `true` or if
    /// the maximum number of allowed steps is exceeded. Each step will be followed by a sequential
    /// call to all given [`Observer`]s' callback functions. Finally, regardless of convergence,
    /// [`Algorithm::postprocessing`] is called. Finally [`Algorithm::summarize`] is called to create a summary of the minimization run. If the algorithm did not converge in the given
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
    /// (number of free parameters) or if any values of `x0` are outside the [`Bound`]s given to the
    /// [`Engine`].
    pub fn minimize(&mut self, func: &dyn CostFunction<U, E>) -> Result<(), E> {
        self.status.reset();
        self.abort_signal.reset();
        self.algorithm.initialize(
            func,
            self.bounds.as_ref(),
            &mut self.status,
            &mut self.user_data,
        )?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step <= self.max_steps
            && !observer_termination
            && !self.algorithm.check_for_termination(
                func,
                self.bounds.as_ref(),
                &mut self.status,
                &mut self.user_data,
            )?
            && !self.abort_signal.is_aborted()
        {
            self.algorithm.step(
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
        self.algorithm.postprocessing(
            func,
            self.bounds.as_ref(),
            &mut self.status,
            &mut self.user_data,
        )?;
        if !observer_termination && current_step > self.max_steps && !self.status.converged() {
            self.status.update_message("MAX EVALS");
        }
        if self.abort_signal.is_aborted() {
            self.status.update_message("Abort signal received");
        }
        self.result = self.algorithm.summarize(
            func,
            self.bounds.as_ref(),
            self.parameter_names.as_ref(),
            &self.status,
            &self.user_data,
        )?;
        Ok(())
    }
}
