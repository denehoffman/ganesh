use std::{convert::Infallible, fmt::Debug, ops::ControlFlow, sync::Arc};

use parking_lot::Mutex;

use crate::traits::{Algorithm, Status};

/// A trait for all kinds of callbacks used in [`Algorithm`]s. These can be implemented for
/// different kinds of [`Algorithm`]s (`A`), problems (`P`), [`Status`]es (`S`), and user data (`U`).
///
/// These are the least restrictive of all callbacks, able to mutate any of their inputs aside from
/// the current step.
pub trait Callback<A, P, S, U = (), E = Infallible> {
    /// A callback method which is called on each step of an [`Algorithm`].
    fn callback(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &mut P,
        status: &mut S,
        user_data: &mut U,
    ) -> ControlFlow<()>;
}

impl<C, A, P, S, U, E> Callback<A, P, S, U, E> for Arc<Mutex<C>>
where
    C: Callback<A, P, S, U, E>,
{
    fn callback(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &mut P,
        status: &mut S,
        user_data: &mut U,
    ) -> ControlFlow<()> {
        self.lock()
            .callback(current_step, algorithm, problem, status, user_data)
    }
}

/// A trait for all kinds of terminators used in [`Algorithm`]s. These can be implemented for
/// different kinds of [`Algorithm`]s (`A`), problems (`P`), [`Status`]es (`S`), and user data (`U`).
///
/// This is the second least-restrictive type of callback, able to mutate both the [`Algorithm`] and its [`Status`].
pub trait Terminator<A, P, S, U, E> {
    /// A termination check which is called on each step of an [`Algorithm`].
    fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        user_data: &U,
    ) -> ControlFlow<()>;
}

impl<T, A, P, S, U, E> Terminator<A, P, S, U, E> for Arc<Mutex<T>>
where
    T: Terminator<A, P, S, U, E>,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        user_data: &U,
    ) -> ControlFlow<()> {
        self.lock()
            .check_for_termination(current_step, algorithm, problem, status, user_data)
    }
}

/// A trait for all kinds of observers used in [`Algorithm`]s. These can be implemented for
/// different kinds of [`Algorithm`]s (`A`), problems (`P`), [`Status`]es (`S`), and user data (`U`).
///
/// This is the most restrictive type of callback and is not able to mutate any of its inputs aside
/// from itself.
pub trait Observer<A, P, S, U, E> {
    /// An observation method which is called on each step of an [`Algorithm`].
    fn observe(
        &mut self,
        current_step: usize,
        algorithm: &A,
        problem: &P,
        status: &S,
        user_data: &U,
    );
}

impl<O, A, P, S, U, E> Observer<A, P, S, U, E> for Arc<Mutex<O>>
where
    O: Observer<A, P, S, U, E>,
{
    fn observe(
        &mut self,
        current_step: usize,
        algorithm: &A,
        problem: &P,
        status: &S,
        user_data: &U,
    ) {
        self.lock()
            .observe(current_step, algorithm, problem, status, user_data)
    }
}

enum CallbackLike<A, P, S, U, E> {
    Callback(Box<dyn Callback<A, P, S, U, E>>),
    Terminator(Box<dyn Terminator<A, P, S, U, E>>),
    Observer(Box<dyn Observer<A, P, S, U, E>>),
}
impl<A, P, S, U, E> CallbackLike<A, P, S, U, E> {
    fn callback(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &mut P,
        status: &mut S,
        user_data: &mut U,
    ) -> ControlFlow<()> {
        match self {
            Self::Callback(callback) => {
                callback.callback(current_step, algorithm, problem, status, user_data)
            }
            Self::Terminator(terminator) => terminator.check_for_termination(
                current_step,
                algorithm,
                problem,
                status,
                user_data,
            ),
            Self::Observer(observer) => {
                observer.observe(current_step, algorithm, problem, status, user_data);
                ControlFlow::Continue(())
            }
        }
    }
}

/// A set of [`Callback`]s which can be used as an input to [`Algorithm::process`].
pub struct Callbacks<A, P, S, U, E>(Vec<CallbackLike<A, P, S, U, E>>);
impl<A, P, S, U, E> Callbacks<A, P, S, U, E> {
    /// Create an empty set of [`Callback`]s.
    pub const fn empty() -> Self {
        Self(Vec::new())
    }
    /// Return the set of [`Callbacks`] with an additional [`Callback`] added.
    pub fn with_callback<C>(mut self, callback: C) -> Self
    where
        C: Callback<A, P, S, U, E> + 'static,
        A: Algorithm<P, S, U, E>,
        S: Status,
    {
        self.0.push(CallbackLike::Callback(Box::new(callback)));
        self
    }

    /// Return the set of [`Callbacks`] with an additional [`Terminator`] added.
    pub fn with_terminator<T>(mut self, terminator: T) -> Self
    where
        T: Terminator<A, P, S, U, E> + 'static,
        A: Algorithm<P, S, U, E>,
        S: Status,
    {
        self.0.push(CallbackLike::Terminator(Box::new(terminator)));
        self
    }

    /// Return the set of [`Callbacks`] with an additional [`Observer`] added.
    pub fn with_observer<O>(mut self, observer: O) -> Self
    where
        O: Observer<A, P, S, U, E> + 'static,
        A: Algorithm<P, S, U, E>,
        S: Status,
    {
        self.0.push(CallbackLike::Observer(Box::new(observer)));
        self
    }
}
impl<A, P, S, U, E> Callback<A, P, S, U, E> for Callbacks<A, P, S, U, E>
where
    A: Algorithm<P, S, U, E>,
    S: Status,
{
    fn callback(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &mut P,
        status: &mut S,
        user_data: &mut U,
    ) -> ControlFlow<()> {
        if self.0.iter_mut().any(|callback| {
            callback
                .callback(current_step, algorithm, problem, status, user_data)
                .is_break()
        }) {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}
impl<A, P, S, U, E> Default for Callbacks<A, P, S, U, E> {
    fn default() -> Self {
        Self(Default::default())
    }
}

/// A [`Callback`] which terminates the algorithm after a number of steps.
pub struct MaxSteps(pub usize);
impl Default for MaxSteps {
    fn default() -> Self {
        Self(4000)
    }
}
impl<A, P, S, U, E> Terminator<A, P, S, U, E> for MaxSteps {
    fn check_for_termination(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        _status: &mut S,
        _user_data: &U,
    ) -> ControlFlow<()> {
        if current_step >= self.0 {
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
///
/// let mut problem = Rosenbrock { n: 2 };
/// let mut nm = NelderMead::default();
/// let result = nm.process(&mut problem, &mut (), NelderMeadConfig::default().with_x0([2.3, 3.4]), NelderMead::default_callbacks().with_observer(DebugObserver)).unwrap();
/// // ^ This will print debug messages for each step
/// assert!(result.converged);
/// ```
pub struct DebugObserver;
impl<A, P, S, U, E> Observer<A, P, S, U, E> for DebugObserver
where
    S: Debug,
    U: Debug,
{
    fn observe(
        &mut self,
        current_step: usize,
        _algorithm: &A,
        _problem: &P,
        status: &S,
        _user_data: &U,
    ) {
        println!("Step: {}\n{:#?}", current_step, status);
    }
}
