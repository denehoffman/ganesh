#![allow(dead_code, unused_variables)]
use crate::{core::Point, traits::CostFunction, Float};
use nalgebra::{Complex, DVector};
use parking_lot::RwLock;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Affine Invariant MCMC Ensemble Sampler
pub mod aies;
pub use aies::{AIESMove, AIES};

/// Ensemble Slice Sampler
pub mod ess;
pub use ess::{ESSMove, ESS};

/// The [`EnsembleStatus`] which holds information about the ensemble used by a ensemble sampler
pub mod ensemble_status;
pub use ensemble_status::EnsembleStatus;

/// A MCMC walker containing a history of past samples
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Walker {
    history: Vec<Arc<RwLock<Point>>>,
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
        let n_variables = self.history[0].read().dimension();
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
    pub fn get_latest(&self) -> Arc<RwLock<Point>> {
        assert!(!self.history.is_empty());
        self.history[self.history.len() - 1].clone()
    }
    /// Evaluate the most recent position of the [`Walker`]
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    pub fn evaluate_latest<U, E>(
        &mut self,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.get_latest().write().evaluate(func, user_data)
    }
    /// Add a new position to the [`Walker`]'s history
    pub fn push(&mut self, position: Arc<RwLock<Point>>) {
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
