use std::{fmt::Debug, sync::Arc};

use parking_lot::RwLock;

use crate::{Ensemble, Float, MCMCObserver, Observer, Status};

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
/// let mut problem = Rosenbrock { n: 2 };
/// let nm = NelderMead::default();
/// let obs = DebugObserver::build();
/// let mut m = Minimizer::new(Box::new(nm), 2).with_observer(obs);
/// m.minimize(&mut problem, &[2.3, 3.4], &mut ()).unwrap();
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
impl<U: Debug> Observer<U> for DebugObserver {
    fn callback(&mut self, step: usize, status: &mut Status, user_data: &mut U) -> bool {
        println!("{step}, {:?}, {:?}", status, user_data);
        false
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
/// use ganesh::samplers::{ESS, ESSMove};
/// use ganesh::test_functions::NegativeRosenbrock;
/// use ganesh::observers::DebugMCMCObserver;
/// use fastrand::Rng;
/// use nalgebra::DVector;
///
/// let problem = NegativeRosenbrock { n: 2 };
/// let mut rng = Rng::new();
/// let x0 = (0..5).map(|_| DVector::from_fn(2, |_, _| rng.normal(1.0, 4.0))).collect();
/// let ess = ESS::new([ESSMove::gaussian(0.1), ESSMove::differential(0.9)], rng);
/// let obs = DebugMCMCObserver::build();
/// let mut sampler = Sampler::new(Box::new(ess), x0).with_observer(obs);
/// sampler.sample(&problem, &mut (), 10).unwrap();
/// // ^ This will print debug messages for each step
/// assert!(sampler.ensemble.dimension() == (5, 10, 2));
/// ```
pub struct DebugMCMCObserver;
impl DebugMCMCObserver {
    /// Finalize the [`MCMCObserver`] by wrapping it in an [`Arc`] and [`RwLock`]
    pub fn build() -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(Self))
    }
}
impl<U: Debug> MCMCObserver<U> for DebugMCMCObserver {
    fn callback(&mut self, step: usize, ensemble: &mut Ensemble, user_data: &mut U) -> bool {
        println!("{step}, {:?}, {:?}", ensemble, user_data);
        false
    }
}

/// An obsever which can check the integrated autocorrelation time of the ensemble and
/// terminate if convergence conditions are met
///
/// After getting the IAT for each parameter, the mean value is taken and multiplied by
/// `n_taus_threshold` and compared to the current step. If the ensemble has passed the required
/// number of steps and the change in the mean IAT is less than the given `dtau_threshold`, the
/// observer terminates the sampler (if `terminate` is `true`).
///
/// # Usage:
///
/// ```rust
/// use ganesh::Sampler;
/// use ganesh::traits::*;
/// use ganesh::samplers::{ESS, ESSMove};
/// use ganesh::test_functions::NegativeRosenbrock;
/// use ganesh::observers::AutocorrelationObserver;
/// use fastrand::Rng;
/// use nalgebra::DVector;
///
/// let problem = NegativeRosenbrock { n: 2 };
/// let mut rng = Rng::new();
/// let x0 = (0..5).map(|_| DVector::from_fn(2, |_, _| rng.normal(1.0, 4.0))).collect();
/// let ess = ESS::new([ESSMove::gaussian(0.1), ESSMove::differential(0.9)], rng);
/// let obs = AutocorrelationObserver::default().with_n_check(20).build();
/// let mut sampler = Sampler::new(Box::new(ess), x0).with_observer(obs);
/// sampler.sample(&problem, &mut (), 100).unwrap();
/// // ^ This will print autocorrelation messages for every 20 steps
/// assert!(sampler.ensemble.dimension() == (5, 100, 2));
/// ```
pub struct AutocorrelationObserver {
    n_check: usize,
    n_taus_threshold: usize,
    dtau_threshold: Float,
    discard: Float,
    terminate: bool,
    c: Option<Float>,
    verbose: bool,
    /// A list of recorded mean IAT values
    pub taus: Vec<Float>,
}

impl AutocorrelationObserver {
    /// Set how often (in number of steps) to check this observer (default: `50`)
    pub const fn with_n_check(mut self, n_check: usize) -> Self {
        self.n_check = n_check;
        self
    }
    /// Set the number of mean integrated autocorrelation times needed to terminate (default: `50`)
    pub const fn with_n_taus_threshold(mut self, n_taus_threshold: usize) -> Self {
        self.n_taus_threshold = n_taus_threshold;
        self
    }
    /// Set the threshold for the absolute change in integrated autocorrelation time Δτ/τ (default: `0.01`)
    pub const fn with_dtau_threshold(mut self, dtau_threshold: Float) -> Self {
        self.dtau_threshold = dtau_threshold;
        self
    }
    /// Set the fraction of steps to discard from the beginning of the chain (default: `0.5`)
    pub const fn with_discard(mut self, discard: Float) -> Self {
        self.discard = discard;
        self
    }
    /// Set to `false` to forego termination even if the chains converge (default: `true`)
    pub const fn with_terminate(mut self, terminate: bool) -> Self {
        self.terminate = terminate;
        self
    }
    /// Set the integrated autocorrelation time window size[^Sokal] (default: `7.0`)
    /// [^Sokal]: Sokal, A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms. In C. DeWitt-Morette, P. Cartier, & A. Folacci (Eds.), Functional Integration: Basics and Applications (pp. 131–192). doi:10.1007/978-1-4899-0319-8_6
    pub const fn with_sokal_window(mut self, c: Float) -> Self {
        self.c = Some(c);
        self
    }
    /// Set to `true` to print out details at each check (default: `false`)
    pub const fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
    /// Finalize the [`MCMCObserver`] by wrapping it in an [`Arc`] and [`RwLock`]
    pub fn build(self) -> Arc<RwLock<Self>> {
        Arc::new(RwLock::new(self))
    }
}

impl Default for AutocorrelationObserver {
    fn default() -> Self {
        Self {
            n_check: 50,
            n_taus_threshold: 50,
            dtau_threshold: 0.01,
            discard: 0.5,
            terminate: true,
            c: None,
            verbose: false,
            taus: Vec::default(),
        }
    }
}

impl<U> MCMCObserver<U> for AutocorrelationObserver {
    fn callback(&mut self, step: usize, ensemble: &mut Ensemble, _user_data: &mut U) -> bool {
        if step % self.n_check == 0 {
            let taus = ensemble.get_integrated_autocorrelation_times(
                self.c,
                Some((step as Float * self.discard) as usize),
                None,
            );
            let tau = taus.mean();
            let enough_steps = tau * (self.n_taus_threshold as Float) < step as Float;
            let (dtau, dtau_met) = if !self.taus.is_empty() {
                let dtau = Float::abs(self.taus.last().unwrap_or(&0.0) - tau) / tau;
                (dtau, dtau < self.dtau_threshold)
            } else {
                (Float::NAN, false)
            };
            let converged = enough_steps && dtau_met;
            if self.verbose {
                println!("Integrated Autocorrelation Analysis:");
                println!("τ = \n{}", taus);
                println!(
                    "Minimum steps to converge = {}",
                    (tau * (self.n_taus_threshold as Float)) as usize
                );
                println!("Steps completed = {}", step);
                println!("Δτ/τ = {} (converges if < {})", dtau, self.dtau_threshold);
                println!("Converged: {}\n", converged);
            }
            self.taus.push(tau);
            return converged && self.terminate;
        }
        false
    }
}
