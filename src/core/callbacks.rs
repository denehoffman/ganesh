use crate::traits::{Algorithm, Observer, Status, Terminator};
use std::{fmt::Debug, ops::ControlFlow};

enum CallbackLike<A, P, S, U, E, C> {
    Terminator(Box<dyn Terminator<A, P, S, U, E, C>>),
    Observer(Box<dyn Observer<A, P, S, U, E, C>>),
}
impl<A, P, S, U, E, C> CallbackLike<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn callback(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        args: &U,
        config: &C,
    ) -> ControlFlow<()> {
        match self {
            Self::Terminator(terminator) => terminator.check_for_termination(
                current_step,
                algorithm,
                problem,
                status,
                args,
                config,
            ),
            Self::Observer(observer) => {
                observer.observe(current_step, algorithm, problem, status, args, config);
                ControlFlow::Continue(())
            }
        }
    }
}

/// A set of [`Terminator`]s and/or [`Observer`]s which can be used as an input to [`Algorithm::process`].
pub struct Callbacks<A, P, S, U, E, C>(Vec<CallbackLike<A, P, S, U, E, C>>);
impl<A, P, S, U, E, C> Callbacks<A, P, S, U, E, C> {
    /// Create an empty set of callbacks.
    pub const fn empty() -> Self {
        Self(Vec::new())
    }

    /// Return the set of [`Callbacks`] with an additional [`Terminator`] added.
    pub fn with_terminator<T>(mut self, terminator: T) -> Self
    where
        T: Terminator<A, P, S, U, E, C> + 'static,
        A: Algorithm<P, S, U, E, Config = C>,
        S: Status,
    {
        self.0.push(CallbackLike::Terminator(Box::new(terminator)));
        self
    }

    /// Return the set of [`Callbacks`] with an additional [`Observer`] added.
    pub fn with_observer<O>(mut self, observer: O) -> Self
    where
        O: Observer<A, P, S, U, E, C> + 'static,
        A: Algorithm<P, S, U, E, Config = C>,
        S: Status,
    {
        self.0.push(CallbackLike::Observer(Box::new(observer)));
        self
    }
}
impl<A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for Callbacks<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        args: &U,
        config: &C,
    ) -> ControlFlow<()> {
        if self.0.iter_mut().any(|callback| {
            callback
                .callback(current_step, algorithm, problem, status, args, config)
                .is_break()
        }) {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// A [`Terminator`] which terminates the algorithm after a number of steps.
pub struct MaxSteps(pub usize);
impl Default for MaxSteps {
    fn default() -> Self {
        Self(4000)
    }
}
impl<A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for MaxSteps
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        status: &mut S,
        _args: &U,
        _config: &C,
    ) -> ControlFlow<()> {
        if current_step >= self.0.saturating_sub(1) {
            status.update_message(&format!("Maximum number of steps reached ({})!", self.0));
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// A debugging callback which prints out the step, status, and any user data at the current step
/// in an algorithm.
///
/// # Usage:
///
/// ```rust
/// use ganesh::traits::*;
/// use ganesh::algorithms::gradient_free::{NelderMead, NelderMeadConfig};
/// use ganesh::test_functions::Rosenbrock;
/// use ganesh::core::DebugObserver;
///
/// let problem = Rosenbrock { n: 2 };
/// let mut nm = NelderMead::default();
/// let result = nm.process(&problem, &(), NelderMeadConfig::new([2.3, 3.4]), NelderMead::default_callbacks().with_observer(DebugObserver)).unwrap();
/// // ^ This will print debug messages for each step
/// assert!(result.converged);
/// ```
pub struct DebugObserver;
impl<A, P, S, U, E, C> Observer<A, P, S, U, E, C> for DebugObserver
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status + Debug,
    U: Debug,
{
    fn observe(
        &mut self,
        current_step: usize,
        _algorithm: &A,
        _problem: &P,
        status: &S,
        _args: &U,
        _config: &C,
    ) {
        println!("Step: {}\n{:#?}", current_step, status);
    }
}
