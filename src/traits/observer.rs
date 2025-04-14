use std::{fmt::Debug, sync::Arc};

use parking_lot::RwLock;

use crate::core::Bound;

use super::Status;

/// A trait which holds a [`callback`](`Observer::callback`) function that can be used to check an
/// [`Solver`](`crate::traits::Solver`)'s [`Status`] during a minimization.
pub trait Observer<S: Status, U> {
    /// A function that is called at every step of a minimization [`Solver`](`crate::traits::Solver`). If it returns
    /// `true`, the [`Minimizer::minimize`](`crate::core::Minimizer::minimize`) method will terminate.
    fn callback(
        &mut self,
        step: usize,
        bounds: Option<&Vec<Bound>>,
        status: &mut S,
        user_data: &mut U,
    ) -> bool;
}

/// A debugging observer which prints out the step, status, and any user data at the current step
/// in an algorithm.
///
/// # Usage:
///
/// ```rust
/// use ganesh::{Minimizer, NopAbortSignal};
/// use ganesh::traits::*;
/// use ganesh::solvers::gradient_free::NelderMead;
/// use ganesh::test_functions::Rosenbrock;
/// use ganesh::observers::DebugObserver;
///
/// let mut problem = Rosenbrock { n: 2 };
/// let nm = NelderMead::default();
/// let obs = DebugObserver::build();
/// let mut m = Minimizer::new(Box::new(nm), 2).setup(|m| m.with_observer(obs).on_status(|s| s.with_x0([2.3, 3.4])));
/// m.minimize(&mut problem).unwrap();
/// // ^ This will print debug messages for each step
/// assert!(m.status.converged);
/// ```
pub struct DebugObserver;
impl DebugObserver {
    /// Finalize the [`Observer`] by wrapping it in an [`Arc`] and [`RwLock`]
    pub fn build() -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self))
    }
}
impl<S: Status + Debug, U: Debug> Observer<S, U> for DebugObserver {
    fn callback(
        &mut self,
        step: usize,
        _bounds: Option<&Vec<Bound>>,
        status: &mut S,
        _user_data: &mut U,
    ) -> bool {
        println!("{step}, {:?}", status);
        false
    }
}
