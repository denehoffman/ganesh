use std::{convert::Infallible, fmt::Debug, ops::ControlFlow, sync::Arc};

use parking_lot::RwLock;

use crate::traits::{Algorithm, Status};

/// A [`Callback`] wrapped in an [`Arc<RwLock>`].
pub type WrappedCallback<A, P, S, U, E> = Arc<RwLock<dyn Callback<A, P, S, U, E>>>;

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
        problem: &P,
        status: &mut S,
        user_data: &mut U,
    ) -> ControlFlow<()>; // TODO: return a break value?

    /// A shorthand method for turning a [`Callback`] into an [`Arc<RwLock<Callback>>`].
    fn build(self) -> WrappedCallback<A, P, S, U, E>
    where
        Self: Sized + 'static,
    {
        Arc::new(RwLock::new(self))
    }
}

/// A set of [`Callback`]s which can be used as an input to [`Algorithm::process`].
pub struct Callbacks<A, P, S, U, E>(Vec<WrappedCallback<A, P, S, U, E>>);
impl<A, P, S, U, E> Callbacks<A, P, S, U, E> {
    /// Create an empty set of [`Callback`]s.
    pub fn new() -> Self {
        Self::default()
    }
    /// Return the set of [`Callback`]s with an additional callback added.
    pub fn with(mut self, callback: WrappedCallback<A, P, S, U, E>) -> Self {
        self.0.push(callback);
        self
    }
    /// Return the set of [`Callback`]s with additional callbacks added.
    pub fn extend<I>(mut self, callbacks: I) -> Self
    where
        I: IntoIterator<Item = WrappedCallback<A, P, S, U, E>>,
    {
        self.0.extend(callbacks);
        self
    }
    /// View [`Callback`]s as a slice.
    pub fn as_slice(&self) -> &[WrappedCallback<A, P, S, U, E>] {
        self.0.as_slice()
    }
}
impl<A, P, S, U, E> Default for Callbacks<A, P, S, U, E> {
    fn default() -> Self {
        Self(Default::default())
    }
}
impl<A, P, S, U, E> Clone for Callbacks<A, P, S, U, E> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<A, P, S, U, E, const N: usize> From<[WrappedCallback<A, P, S, U, E>; N]>
    for Callbacks<A, P, S, U, E>
{
    fn from(v: [WrappedCallback<A, P, S, U, E>; N]) -> Self {
        Self(v.to_vec())
    }
}
impl<A, P, S, U, E> From<Vec<WrappedCallback<A, P, S, U, E>>> for Callbacks<A, P, S, U, E> {
    fn from(v: Vec<WrappedCallback<A, P, S, U, E>>) -> Self {
        Self(v)
    }
}
impl<A, P, S, U, E> From<WrappedCallback<A, P, S, U, E>> for Callbacks<A, P, S, U, E> {
    fn from(cb: WrappedCallback<A, P, S, U, E>) -> Self {
        Self(vec![cb])
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
        _problem: &P,
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
/// use ganesh::traits::callback::DebugCallback;
///
/// let mut problem = Rosenbrock { n: 2 };
/// let mut nm = NelderMead::default();
/// let obs = DebugCallback.build();
/// let result = nm.process(&mut problem, &mut (), NelderMeadConfig::default().with_x0([2.3, 3.4]), NelderMead::default_callbacks().with(obs)).unwrap();
/// // ^ This will print debug messages for each step
/// assert!(result.converged);
/// ```
pub struct DebugCallback;
impl DebugCallback {
    /// Finalize the [`Observer`] by wrapping it in an [`Arc`] and [`RwLock`]
    pub fn build() -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self))
    }
}
impl<A, P, S: Status + Debug, U: Debug, E> Callback<A, P, S, U, E> for DebugCallback
where
    A: Algorithm<P, S, U, E>,
{
    fn callback(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        status: &mut S,
        _user_data: &mut U,
    ) -> ControlFlow<()> {
        println!("Step: {}\n{:#?}", current_step, status);
        ControlFlow::Continue(())
    }
}
