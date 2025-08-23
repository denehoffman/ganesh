use std::{fmt::Debug, ops::ControlFlow, sync::Arc};

use parking_lot::RwLock;

use super::Status;

/// A trait which holds a [`callback`](`Observer::callback`) function that can be used to check an
/// [`Algorithm`](`crate::traits::Algorithm`)'s [`Status`] during a minimization.
pub trait Observer<S: Status, U> {
    /// A function that is called at every step of a minimization [`Algorithm`](`crate::traits::Algorithm`). If it returns
    /// `true`, the [`Engine::process`](`crate::core::Engine::process`) method will terminate.
    fn callback(&mut self, step: usize, status: &mut S, user_data: &mut U) -> ControlFlow<()>;
}

/// A debugging observer which prints out the step, status, and any user data at the current step
/// in an algorithm.
///
/// # Usage:
///
/// ```rust
/// use ganesh::core::Engine;
/// use ganesh::traits::*;
/// use ganesh::algorithms::gradient_free::NelderMead;
/// use ganesh::test_functions::Rosenbrock;
/// use ganesh::traits::observer::DebugObserver;
///
/// let mut problem = Rosenbrock { n: 2 };
/// let nm = NelderMead::default();
/// let obs = DebugObserver::build();
/// let mut m = Engine::new(nm).setup(|e| e.with_observer(obs.clone()).configure(|c| c.with_x0([2.3, 3.4])));
/// m.process(&mut problem).unwrap();
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
    fn callback(&mut self, step: usize, status: &mut S, _user_data: &mut U) -> ControlFlow<()> {
        println!("Step: {}\n{:#?}", step, status);
        ControlFlow::Continue(())
    }
}
