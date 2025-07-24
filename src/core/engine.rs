use std::sync::Arc;

use parking_lot::RwLock;

use crate::traits::{AbortSignal, Algorithm, CostFunction, Observer, Status};

use super::NopAbortSignal;

const DEFAULT_MAX_STEPS: usize = 4000;
/// The main struct used for running [`Algorithm`]s on [`CostFunction`]s.
pub struct Engine<S, U, E, Summary, A>
where
    A: Algorithm<S, U, E, Summary = Summary>,
{
    /// The [`Status`] of the [`Algorithm`], usually read after minimization.
    pub status: S,
    /// The [`Algorithm::Summary`], usually read after minimization.
    pub result: Summary,

    algorithm: A,
    observers: Vec<Arc<RwLock<dyn Observer<S, U>>>>,
    abort_signal: Box<dyn AbortSignal>,
    user_data: U,

    // bounds: Option<Bounds>,
    parameter_names: Option<Vec<String>>,
    max_steps: usize,

    _error: std::marker::PhantomData<E>,
}

impl<S, U, E, Summary, A> Engine<S, U, E, Summary, A>
where
    S: Status,
    U: Default,
    Summary: Default,
    A: Algorithm<S, U, E, Summary = Summary>,
{
    /// Creates a new [`Engine`] with the given [`Algorithm`].
    pub fn new(solver: A) -> Self {
        Self {
            status: S::default(),
            // bounds: None,
            parameter_names: None,
            max_steps: DEFAULT_MAX_STEPS,
            algorithm: solver,
            observers: Vec::default(),
            abort_signal: Box::new(NopAbortSignal),
            user_data: Default::default(),
            result: Default::default(),
            _error: std::marker::PhantomData,
        }
    }

    /// Convenience method to use chainable methods to setup the [`Engine`].
    ///
    /// # Example:
    ///
    /// ```rust
    /// # use ganesh::algorithms::gradient::LBFGSB;
    /// # use ganesh::core::CtrlCAbortSignal;
    /// # use ganesh::core::Engine;
    /// # use ganesh::traits::Bounded;
    /// # use std::convert::Infallible;
    /// let solver: LBFGSB<(), Infallible> = LBFGSB::default();
    /// let mut m = Engine::new(solver)
    ///   .setup(|e| {
    ///     e.configure(|c| {
    ///         c.with_bounds([(-4.0, 4.0), (-4.0, 4.0)])
    ///     }).with_abort_signal(CtrlCAbortSignal::new())
    ///   });
    /// ```
    pub fn setup<F>(mut self, mut f: F) -> Self
    where
        F: FnMut(&mut Self) -> &mut Self,
    {
        f(&mut self);
        self
    }

    /// Convenience method to use chainable methods to setup the [`Algorithm::Config`]. This is
    /// typically where things like initial conditions and bounds are set.
    ///
    /// # Example:
    ///
    /// ```rust
    /// # use ganesh::algorithms::gradient::LBFGSB;
    /// # use ganesh::core::CtrlCAbortSignal;
    /// # use ganesh::core::Engine;
    /// # use ganesh::traits::Bounded;
    /// # use std::convert::Infallible;
    /// let solver: LBFGSB<(), Infallible> = LBFGSB::default();
    /// let mut m = Engine::new(solver).configure(|c| {
    ///     c.with_bounds([(-4.0, 4.0), (-4.0, 4.0)])
    ///      .with_x0([1.2, 2.3])
    /// });
    /// ```
    pub fn configure<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(&mut A::Config) -> &mut A::Config,
    {
        f(self.algorithm.get_config_mut());
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
    pub fn with_abort_signal<Ab: AbortSignal + 'static>(&mut self, abort_signal: Ab) -> &mut Self {
        self.abort_signal = Box::new(abort_signal);
        self
    }

    /// Set user data for the [`Engine`].
    pub fn with_user_data<T: Into<U>>(&mut self, data: T) -> &mut Self {
        self.user_data = data.into();
        self
    }

    /// Adds a single [`Observer`] to the [`Engine`].
    pub fn with_observer(&mut self, observer: Arc<RwLock<dyn Observer<S, U>>>) -> &mut Self {
        self.observers.push(observer);
        self
    }

    /// Process the given [`CostFunction`] using the [`Engine`]'s stored [`Algorithm`].
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
    pub fn process(&mut self, func: &mut dyn CostFunction<U, E>) -> Result<(), E> {
        self.status.reset();
        self.abort_signal.reset();
        self.algorithm.reset();
        self.algorithm
            .initialize(func, &mut self.status, &mut self.user_data)?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step <= self.max_steps
            && !observer_termination
            && !self
                .algorithm
                .check_for_termination(func, &mut self.status, &mut self.user_data)?
            && !self.abort_signal.is_aborted()
        {
            self.algorithm
                .step(current_step, func, &mut self.status, &mut self.user_data)?;
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
            func.update_user_data(&mut self.user_data);
        }
        self.algorithm
            .postprocessing(func, &mut self.status, &mut self.user_data)?;
        if !observer_termination && current_step > self.max_steps && !self.status.converged() {
            self.status.update_message("MAX EVALS");
        }
        if self.abort_signal.is_aborted() {
            self.status.update_message("Abort signal received");
        }
        self.result = self.algorithm.summarize(
            func,
            self.parameter_names.as_ref(),
            &self.status,
            &self.user_data,
        )?;
        Ok(())
    }
}
