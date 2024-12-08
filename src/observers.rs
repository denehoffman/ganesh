use std::fmt::Debug;


use crate::{Observer, Status};

/// A debugging observer which prints out the step, status, and any user data at the current step
/// in an algorithm.
///
/// # Usage:
///
/// ```rust
/// use ganesh::prelude::*;
/// use ganesh::algorithms::NelderMead;
/// use ganesh::test_functions::Rosenbrock;
/// use ganesh::observers::DebugObserver;
///
/// let problem = Rosenbrock { n: 2 };
/// let nm = NelderMead::default();
/// let obs = DebugObserver;
/// let mut m = Minimizer::new(&nm, 2).with_observer(obs);
/// m.minimize(&problem, &[2.3, 3.4], &mut ()).unwrap();
/// // ^ This will print debug messages for each step
/// assert!(m.status.converged);
/// ```
pub struct DebugObserver;
impl<U: Debug> Observer<U> for DebugObserver {
    fn callback(&mut self, step: usize, status: &mut Status, user_data: &mut U) -> bool {
        println!("{step}, {:?}, {:?}", status, user_data);
        true
    }
}
