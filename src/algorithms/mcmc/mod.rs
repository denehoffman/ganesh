//! Generic ensemble Markov-chain Monte Carlo algorithms.

use crate::core::{LinearAlgebra, RandomScalar};
use crate::traits::{Algorithm, Terminator};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{ops::ControlFlow, sync::Arc};

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

/// Affine-invariant ensemble sampling.
pub mod aies;
pub use aies::{AIESConfig, AIESInit, AIESMove, EnsembleStatus, AIES};

/// Ensemble slice sampling.
pub mod ess;
pub use ess::{ESSConfig, ESSInit, ESSMove, ESS};

pub use crate::core::mcmc_diagnostics::integrated_autocorrelation_times;

/// Periodically checks ensemble integrated autocorrelation times and optionally terminates.
#[derive(Clone)]
pub struct AutocorrelationTerminator {
    n_check: usize,
    n_taus_threshold: usize,
    dtau_threshold: f64,
    discard: f64,
    terminate: bool,
    c: Option<f64>,
    verbose: bool,
    /// Recorded mean integrated autocorrelation times.
    pub taus: Vec<f64>,
}

impl AutocorrelationTerminator {
    /// Set the number of steps between checks.
    pub const fn with_n_check(mut self, n_check: usize) -> Self {
        self.n_check = n_check;
        self
    }
    /// Set the minimum number of autocorrelation times required.
    pub const fn with_n_taus_threshold(mut self, value: usize) -> Self {
        self.n_taus_threshold = value;
        self
    }
    /// Set the relative autocorrelation-time stability threshold.
    pub const fn with_dtau_threshold(mut self, value: f64) -> Self {
        self.dtau_threshold = value;
        self
    }
    /// Set the fraction of initial samples discarded for the check.
    pub const fn with_discard(mut self, value: f64) -> Self {
        self.discard = value;
        self
    }
    /// Select whether convergence terminates the run.
    pub const fn with_terminate(mut self, value: bool) -> Self {
        self.terminate = value;
        self
    }
    /// Set Sokal's autocorrelation window parameter.
    pub const fn with_sokal_window(mut self, value: f64) -> Self {
        self.c = Some(value);
        self
    }
    /// Select verbose diagnostic output.
    pub const fn with_verbose(mut self, value: bool) -> Self {
        self.verbose = value;
        self
    }
    /// Wrap the terminator for shared inspection after a run.
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
            taus: Vec::new(),
        }
    }
}

impl<T, B, A, P, U, E, C> Terminator<A, P, EnsembleStatus<T, B>, U, E, C>
    for AutocorrelationTerminator
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    A: Algorithm<P, EnsembleStatus<T, B>, U, E, Config = C>,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        status: &mut EnsembleStatus<T, B>,
        _args: &U,
        _config: &C,
    ) -> ControlFlow<()> {
        if current_step % self.n_check != 0 || status.chain.is_empty() {
            return ControlFlow::Continue(());
        }
        let discard = (current_step as f64 * self.discard) as usize;
        let samples = status
            .chain
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .skip(discard.min(chain.len()))
                    .filter_map(|sample| {
                        let values = (0..sample.len())
                            .map(|index| sample.get(index).to_f64())
                            .collect::<Option<Vec<_>>>()?;
                        Some(nalgebra::DVector::from_vec(values))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if samples.iter().any(Vec::is_empty) {
            return ControlFlow::Continue(());
        }
        let taus = integrated_autocorrelation_times(samples, self.c);
        let tau = taus.iter().sum::<f64>() / taus.len() as f64;
        let enough_steps = tau * (self.n_taus_threshold as f64) < current_step as f64;
        let dtau = self
            .taus
            .last()
            .map_or(f64::NAN, |previous| (previous - tau).abs() / tau);
        let converged = enough_steps && dtau < self.dtau_threshold;
        if self.verbose {
            println!("Integrated Autocorrelation Analysis:");
            println!("τ = \n{taus}");
            println!(
                "Minimum steps to converge = {}",
                (tau * self.n_taus_threshold as f64) as usize
            );
            println!("Steps completed = {current_step}");
            println!("Δτ/τ = {dtau} (converges if < {})", self.dtau_threshold);
            println!("Converged: {converged}\n");
        }
        self.taus.push(tau);
        if converged && self.terminate {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}
