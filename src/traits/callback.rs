use std::{convert::Infallible, fmt::Debug, ops::ControlFlow, sync::Arc};

use parking_lot::Mutex;

use crate::traits::{Algorithm, Status};

/// A trait for all kinds of callbacks used in [`Algorithm`]s. These can be implemented for
/// different kinds of [`Algorithm`]s (`A`), problems (`P`), [`Status`]es (`S`), and user data (`U`).
///
/// These are intended to act as terminators, observers, or any other kind of callback, including abort signals.
pub trait Callback<A, P, S, U = (), E = Infallible>
where
    A: Algorithm<P, S, U, E>,
    S: Status,
{
    /// A callback method which is called on each step of an [`Algorithm`].
    fn callback(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &mut P,
        status: &mut S,
        user_data: &mut U,
    ) -> ControlFlow<()>; // TODO: return a break value?
}

impl<T, A, P, S, U, E> Callback<A, P, S, U, E> for Arc<Mutex<T>>
where
    T: Callback<A, P, S, U, E>,
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
        self.lock()
            .callback(current_step, algorithm, problem, status, user_data)
    }
}

/// A set of [`Callback`]s which can be used as an input to [`Algorithm::process`].
pub struct Callbacks<A, P, S, U, E>(Vec<Box<dyn Callback<A, P, S, U, E>>>);
impl<A, P, S, U, E> Callbacks<A, P, S, U, E> {
    /// Create a new set of [`Callback`]s from the given vector.
    pub fn new(callbacks: Vec<Box<dyn Callback<A, P, S, U, E>>>) -> Self {
        Self(callbacks)
    }
    /// Create an empty set of [`Callback`]s.
    pub fn empty() -> Self {
        Self(Vec::new())
    }
    /// Return the set of [`Callback`]s with an additional callback added.
    pub fn with<C>(mut self, callback: C) -> Self
    where
        C: Callback<A, P, S, U, E> + 'static,
        A: Algorithm<P, S, U, E>,
        S: Status,
    {
        self.0.push(Box::new(callback));
        self
    }
    /// View [`Callback`]s as a slice.
    pub fn as_slice(&self) -> &[Box<dyn Callback<A, P, S, U, E>>] {
        self.0.as_slice()
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
impl<A, P, S, U, E> Callback<A, P, S, U, E> for MaxSteps
where
    A: Algorithm<P, S, U, E>,
    S: Status,
{
    fn callback(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &mut P,
        _status: &mut S,
        _user_data: &mut U,
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
/// let result = nm.process(&mut problem, &mut (), NelderMeadConfig::default().with_x0([2.3, 3.4]), NelderMead::default_callbacks().with(DebugObserver)).unwrap();
/// // ^ This will print debug messages for each step
/// assert!(result.converged);
/// ```
pub struct DebugObserver;
impl<A, P, S, U, E> Callback<A, P, S, U, E> for DebugObserver
where
    A: Algorithm<P, S, U, E>,
    S: Status + Debug,
    U: Debug,
{
    fn callback(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &mut P,
        status: &mut S,
        _user_data: &mut U,
    ) -> ControlFlow<()> {
        println!("Step: {}\n{:#?}", current_step, status);
        ControlFlow::Continue(())
    }
}
