use crate::{
    core::Point,
    traits::{Algorithm, LogDensity, Terminator},
    DVector, Float,
};
use nalgebra::Complex;
use parking_lot::{Mutex, RwLock};
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::{ops::ControlFlow, sync::Arc};

/// Affine Invariant MCMC Ensemble Sampler
pub mod aies;
pub use aies::{AIESConfig, AIESMove, AIES};

/// Ensemble Slice Sampler
pub mod ess;
pub use ess::{ESSConfig, ESSMove, ESS};

/// The [`EnsembleStatus`] which holds information about the ensemble used by a ensemble sampler
pub mod ensemble_status;
pub use ensemble_status::EnsembleStatus;

/// A MCMC walker containing a history of past samples
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Walker {
    history: Vec<Arc<RwLock<Point<DVector<Float>>>>>,
}
impl Walker {
    /// Create a new [`Walker`] located at `x0`
    pub fn new(x0: DVector<Float>) -> Self {
        let history = vec![Arc::new(RwLock::new(Point::from(x0)))];
        Self { history }
    }
    /// Get the dimension of the [`Walker`] `(n_steps, n_variables)`
    pub fn dimension(&self) -> (usize, usize) {
        let n_steps = self.history.len();
        let n_variables = self.history[0].read().x.len();
        (n_steps, n_variables)
    }
    /// Reset the history of the [`Walker`] (except for its starting position)
    pub fn reset(&mut self) {
        let first = self.history.first();
        if let Some(first) = first {
            self.history = vec![first.clone()];
        } else {
            self.history = Vec::default();
        }
    }
    /// Get the most recent (current) [`Walker`]'s position
    ///
    /// # Panics
    ///
    /// This method panics if the walker has no history.
    pub fn get_latest(&self) -> Arc<RwLock<Point<DVector<Float>>>> {
        assert!(!self.history.is_empty());
        self.history[self.history.len() - 1].clone()
    }
    /// Evaluate the most recent position of the [`Walker`]
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`](`crate::traits::CostFunction::evaluate`) for more information.
    pub fn log_density_latest<U, E>(
        &mut self,
        func: &dyn LogDensity<U, E>,
        args: &U,
    ) -> Result<(), E> {
        self.get_latest().write().log_density(func, args)
    }
    /// Add a new position to the [`Walker`]'s history
    pub fn push(&mut self, position: Arc<RwLock<Point<DVector<Float>>>>) {
        self.history.push(position)
    }
}

/// Calculate the integrated autocorrelation time for each parameter according to Karamanis &
/// Beutler[^Karamanis]
///
/// `samples` should have the shape `(n_walkers, n_steps, n_parameters)`.
///
/// `c` is an optional window size (`7.0` if [`None`] provided), see Sokal[^Sokal].
///
/// This is a standalone function that can be used to bypass the [`EnsembleStatus`] struct and calculate
/// IATs for custom inputs.
///
/// [^Karamanis]: Karamanis, M., & Beutler, F. (2020). Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions. arXiv Preprint arXiv: 2002. 06212.
/// [^Sokal]: Sokal, A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms. In C. DeWitt-Morette, P. Cartier, & A. Folacci (Eds.), Functional Integration: Basics and Applications (pp. 131–192). doi:10.1007/978-1-4899-0319-8_6
pub fn integrated_autocorrelation_times(
    samples: Vec<Vec<DVector<Float>>>,
    c: Option<Float>,
) -> DVector<Float> {
    let c = c.unwrap_or(7.0);
    let n_parameters = samples[0][0].len();
    let samples: Vec<DVector<Float>> = samples.into_iter().flatten().collect();
    let mut n = 1usize;
    while n < samples.len() {
        n <<= 1;
    }
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(2 * n);
    let ifft = planner.plan_fft_inverse(2 * n);
    DVector::from_iterator(
        n_parameters,
        (0..n_parameters).map(|i_parameter| {
            let x: Vec<Float> = samples.iter().map(|sample| sample[i_parameter]).collect();
            let mean = x.iter().sum::<Float>() / x.len() as Float;
            let mut input: Vec<Complex<Float>> =
                x.iter().map(|&val| Complex::new(val - mean, 0.0)).collect();
            input.resize(2 * n, Complex::new(0.0, 0.0));

            fft.process(&mut input);

            for val in input.iter_mut() {
                *val *= val.conj();
            }

            ifft.process(&mut input);

            let mut acf: Vec<Float> = input
                .iter()
                .take(x.len())
                .map(|c| c.re / (4.0 * n as Float))
                .collect();

            if !acf.is_empty() && acf[0] != 0.0 {
                let norm_factor = acf[0];
                acf.iter_mut().for_each(|v| *v /= norm_factor);
            }

            let taus: Vec<Float> = acf
                .iter()
                .scan(0.0, |acc, &x| {
                    *acc += x;
                    Some(*acc)
                })
                .map(|x| Float::mul_add(2.0, x, -1.0))
                .collect();
            let ind = taus
                .iter()
                .enumerate()
                .position(|(idx, &tau)| (idx as Float) >= c * tau)
                .unwrap_or(taus.len() - 1);
            taus[ind]
        }),
    )
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
/// use fastrand::Rng;
/// use ganesh::algorithms::mcmc::AutocorrelationTerminator;
/// use ganesh::algorithms::mcmc::{ESSMove, ESS, ESSConfig};
/// use ganesh::test_functions::Rosenbrock;
/// use ganesh::{core::{utils::SampleFloat, Callbacks}, Float, DVector};
/// use ganesh::traits::*;
/// use approx::assert_relative_eq;
///
/// let problem = Rosenbrock { n: 2 };
/// let mut rng = Rng::new();
/// // Use a seed that will converge in a reasonable amount of time
/// rng.seed(0);
/// let x0: Vec<DVector<Float>> = (0..5)
///     .map(|_| DVector::from_fn(2, |_, _| rng.normal(1.0, 4.0)))
///     .collect();
/// let aco = AutocorrelationTerminator::default()
///     .with_n_check(20)
///     .with_verbose(true)
///     .build();
/// let mut sampler = ESS::new(Some(1));
/// let result = sampler.process(&problem, &(),
/// ESSConfig::new(x0.clone()).with_moves([ESSMove::gaussian(0.1),
/// ESSMove::differential(0.9)]), Callbacks::empty().with_terminator(aco.clone())).unwrap();
///
/// println!(
///     "Walker 0 Final Position: {}",
///     result.chain[0].last().unwrap()
/// );
/// println!(
///     "Autocorrelation Time at Termination: {}",
///     aco.lock().taus.last().unwrap()
/// )
/// ```
pub struct AutocorrelationTerminator {
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

impl AutocorrelationTerminator {
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

    /// Wrap the observer in an [`Arc<Mutex<_>>`].
    pub fn build(self) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(self))
    }
}

impl Default for AutocorrelationTerminator {
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

impl<A, P, U, E, C> Terminator<A, P, EnsembleStatus, U, E, C> for AutocorrelationTerminator
where
    A: Algorithm<P, EnsembleStatus, U, E, Config = C>,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        status: &mut EnsembleStatus,
        _args: &U,
        _config: &C,
    ) -> ControlFlow<()> {
        if current_step % self.n_check == 0 {
            let taus = status.get_integrated_autocorrelation_times(
                self.c,
                Some((current_step as Float * self.discard) as usize),
                None,
            );
            let tau = taus.mean();
            let enough_steps = tau * (self.n_taus_threshold as Float) < current_step as Float;
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
                println!("Steps completed = {}", current_step);
                println!("Δτ/τ = {} (converges if < {})", dtau, self.dtau_threshold);
                println!("Converged: {}\n", converged);
            }
            self.taus.push(tau);
            if converged && self.terminate {
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{utils::SampleFloat, Callbacks},
        test_functions::Rosenbrock,
        DVector,
    };
    use fastrand::Rng;

    #[test]
    fn test_autocorrelation_terminator() {
        let problem = Rosenbrock { n: 2 };
        let mut rng = Rng::new();
        rng.seed(0);
        let x0: Vec<DVector<Float>> = (0..5)
            .map(|_| DVector::from_fn(2, |_, _| rng.normal(1.0, 4.0)))
            .collect();
        let aco = AutocorrelationTerminator::default()
            .with_n_check(20)
            .with_discard(0.55)
            .with_sokal_window(7.1)
            .with_terminate(true)
            .with_dtau_threshold(0.05)
            .with_n_taus_threshold(51)
            .with_verbose(false)
            .build();
        let mut sampler = ESS::new(Some(1));
        let result = sampler
            .process(
                &problem,
                &(),
                ESSConfig::new(x0).with_moves([ESSMove::gaussian(0.1), ESSMove::differential(0.9)]),
                Callbacks::empty().with_terminator(aco.clone()),
            )
            .unwrap();
        println!(
            "Walker 0 Final Position: {}",
            result.chain[0].last().unwrap()
        );
        println!(
            "Autocorrelation Time at Termination: {}",
            aco.lock().taus.last().unwrap()
        )
    }
}
