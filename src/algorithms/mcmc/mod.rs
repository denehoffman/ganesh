use crate::{
    DVector, Float,
    core::Point,
    error::{GaneshError, GaneshResult},
    traits::{Algorithm, LogDensity, Terminator},
};
use nalgebra::Complex;
use parking_lot::Mutex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::{ops::ControlFlow, sync::Arc};

/// Affine Invariant MCMC Ensemble Sampler
pub mod aies;
pub use aies::{AIES, AIESConfig, AIESMove};

/// Ensemble Slice Sampler
pub mod ess;
pub use ess::{ESS, ESSConfig, ESSMove};

/// The [`EnsembleStatus`] which holds information about the ensemble used by a ensemble sampler
pub mod ensemble_status;
pub use ensemble_status::EnsembleStatus;

/// Controls how much MCMC chain history is retained in memory.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
pub enum ChainStorageMode {
    /// Retain the full chain for every walker.
    #[default]
    Full,
    /// Retain only the most recent samples for every walker.
    Rolling {
        /// The maximum number of samples retained per walker.
        window: usize,
    },
    /// Retain only periodic samples for every walker.
    Sampled {
        /// Retain every `keep_every`th sample after the initial point.
        keep_every: usize,
        /// Optionally cap the number of retained samples per walker.
        max_samples: Option<usize>,
    },
}

impl ChainStorageMode {
    pub(crate) const fn history_limit(self) -> Option<usize> {
        match self {
            Self::Full => None,
            Self::Rolling { window } => Some(window),
            Self::Sampled { max_samples, .. } => max_samples,
        }
    }
}

pub(crate) fn validate_weighted_moves(weights: &[Float], family: &str) -> GaneshResult<()> {
    if weights.is_empty() {
        return Err(GaneshError::ConfigError(format!(
            "{family} move weights must not be empty"
        )));
    }
    if weights.iter().any(|&weight| !weight.is_finite() || weight < 0.0) {
        return Err(GaneshError::ConfigError(format!(
            "{family} move weights must be finite and non-negative"
        )));
    }
    let total = weights.iter().sum::<Float>();
    if !total.is_finite() || total <= 0.0 {
        return Err(GaneshError::ConfigError(format!(
            "{family} move weights must sum to a positive finite value"
        )));
    }
    Ok(())
}

pub(crate) fn validate_walker_inputs(
    walkers: &[DVector<Float>],
    family: &str,
    min_walkers: usize,
) -> GaneshResult<()> {
    if walkers.len() < min_walkers {
        return Err(GaneshError::ConfigError(format!(
            "{family} requires at least {min_walkers} walkers"
        )));
    }
    let Some(first) = walkers.first() else {
        return Err(GaneshError::ConfigError(format!(
            "{family} walker list must not be empty"
        )));
    };
    if first.is_empty() {
        return Err(GaneshError::ConfigError(format!(
            "{family} walker dimension must be at least 1"
        )));
    }
    if walkers.iter().any(|walker| walker.len() != first.len()) {
        return Err(GaneshError::ConfigError(format!(
            "{family} walkers must all have the same dimension"
        )));
    }
    Ok(())
}

/// A MCMC walker containing a history of past samples
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Walker {
    initial: Point<DVector<Float>>,
    current: Point<DVector<Float>>,
    history: Vec<Point<DVector<Float>>>,
    chain_storage: ChainStorageMode,
    current_retained: bool,
    total_samples_seen: usize,
}
impl Walker {
    /// Create a new [`Walker`] located at `x0`
    pub fn new(x0: DVector<Float>) -> Self {
        let initial = Point::from(x0);
        let current = initial.clone();
        let history = vec![initial.clone()];
        Self {
            initial,
            current,
            history,
            chain_storage: ChainStorageMode::Full,
            current_retained: true,
            total_samples_seen: 1,
        }
    }
    /// Get the dimension of the [`Walker`] `(n_steps, n_variables)`
    pub fn dimension(&self) -> (usize, usize) {
        let n_steps = self.retained_len();
        let n_variables = self.current.x.len();
        (n_steps, n_variables)
    }
    /// Reset the history of the [`Walker`] (except for its starting position)
    pub fn reset(&mut self) {
        self.current = self.initial.clone();
        self.history = vec![self.initial.clone()];
        self.current_retained = true;
        self.total_samples_seen = 1;
        self.enforce_history_limit();
    }
    /// Get the most recent (current) [`Walker`]'s position
    ///
    /// # Panics
    ///
    /// This method panics if the walker has no history.
    pub fn get_latest(&self) -> &Point<DVector<Float>> {
        &self.current
    }
    /// Get a mutable reference to the most recent (current) [`Walker`]'s position
    ///
    /// # Panics
    ///
    /// This method panics if the walker has no history.
    pub fn get_latest_mut(&mut self) -> &mut Point<DVector<Float>> {
        &mut self.current
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
        self.get_latest_mut().log_density(func, args)
    }
    /// Add a new position to the [`Walker`]'s history
    pub fn push(&mut self, position: Point<DVector<Float>>) {
        self.total_samples_seen += 1;
        self.current = position;
        self.current_retained = self.should_retain_current();
        if self.current_retained {
            self.history.push(self.current.clone());
        }
        self.enforce_history_limit();
    }

    pub(crate) fn set_chain_storage(&mut self, chain_storage: ChainStorageMode) {
        self.chain_storage = chain_storage;
        self.rebuild_retained_history();
        self.enforce_history_limit();
    }

    pub(crate) fn retained_positions(&self) -> Vec<&Point<DVector<Float>>> {
        if self.current_retained {
            self.history.iter().collect()
        } else {
            let mut positions = self.history.iter().collect::<Vec<_>>();
            positions.push(&self.current);
            positions
        }
    }

    fn retained_len(&self) -> usize {
        self.history.len() + usize::from(!self.current_retained)
    }

    fn should_retain_current(&self) -> bool {
        match self.chain_storage {
            ChainStorageMode::Full | ChainStorageMode::Rolling { .. } => true,
            ChainStorageMode::Sampled { keep_every, .. } => {
                keep_every == 0 || (self.total_samples_seen - 1).is_multiple_of(keep_every)
            }
        }
    }

    fn rebuild_retained_history(&mut self) {
        self.history = vec![self.initial.clone()];
        self.current_retained = true;
        if self.total_samples_seen == 1 {
            self.current = self.initial.clone();
            return;
        }
        if self.should_retain_current() {
            self.history.push(self.current.clone());
            self.current_retained = true;
        } else {
            self.current_retained = false;
        }
    }

    fn enforce_history_limit(&mut self) {
        if let Some(limit) = self.chain_storage.history_limit() {
            if self.history.len() > limit {
                let excess = self.history.len() - limit;
                self.history.drain(0..excess);
            }
        }
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
/// ESSConfig::new(x0.clone()).unwrap().with_moves([ESSMove::gaussian(0.1),
/// ESSMove::differential(0.9)]).unwrap(), Callbacks::empty().with_terminator(aco.clone())).unwrap();
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
#[derive(Clone)]
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
        DVector,
        core::{Callbacks, utils::SampleFloat},
        test_functions::Rosenbrock,
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
                ESSConfig::new(x0)
                    .unwrap()
                    .with_moves([ESSMove::gaussian(0.1), ESSMove::differential(0.9)])
                    .unwrap(),
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
