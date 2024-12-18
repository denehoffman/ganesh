use std::fmt::Debug;

use crate::{
    algorithms::mcmc::{Ensemble, MCMCObserver},
    Observer, Status,
};

/// A debugging observer which prints out the step, status, and any user data at the current step
/// in an algorithm.
///
/// # Usage:
///
/// ```rust
/// use ganesh::Minimizer;
/// use ganesh::traits::*;
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

/// A debugging observer which prints out the step, ensemble state, and any user data at the current step
/// in an algorithm.
///
/// # Usage:
///
/// ```rust
/// use ganesh::Sampler;
/// use ganesh::traits::*;
/// use ganesh::mcmc::{ESS, ESSMove};
/// use ganesh::test_functions::NegativeRosenbrock;
/// use ganesh::observers::DebugMCMCObserver;
/// use fastrand::Rng;
/// use nalgebra::DVector;
///
/// let problem = NegativeRosenbrock { n: 2 };
/// let mut rng = Rng::new();
/// let x0 = (0..5).map(|_| DVector::from_fn(2, |_, _| rng.normal(1.0, 4.0))).collect();
/// let ess = ESS::new([ESSMove::gaussian(0.1), ESSMove::differential(0.9)], rng);
/// let obs = DebugMCMCObserver;
/// let mut sampler = Sampler::new(&ess, x0).with_observer(obs);
/// sampler.sample(&problem, &mut (), 10).unwrap();
/// // ^ This will print debug messages for each step
/// assert!(sampler.ensemble.dimension() == (5, 10, 2));
/// ```
pub struct DebugMCMCObserver;
impl<U: Debug> MCMCObserver<U> for DebugMCMCObserver {
    fn callback(&mut self, step: usize, ensemble: &mut Ensemble, user_data: &mut U) -> bool {
        println!("{step}, {:?}, {:?}", ensemble, user_data);
        true
    }
}
