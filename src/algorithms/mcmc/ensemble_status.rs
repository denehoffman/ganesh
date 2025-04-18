use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use fastrand::Rng;
use nalgebra::{DMatrix, DVector, RowDVector};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::{
    core::Point,
    traits::{CostFunction, Status},
    Float,
};

use super::{integrated_autocorrelation_times, Walker};

/// A collection of [`Walker`]s
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct EnsembleStatus {
    /// A list of each [`Walker`] in the ensemble
    pub walkers: Vec<Walker>,
}
impl Deref for EnsembleStatus {
    type Target = Vec<Walker>;

    fn deref(&self) -> &Self::Target {
        &self.walkers
    }
}
impl DerefMut for EnsembleStatus {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.walkers
    }
}
impl EnsembleStatus {
    /// Create a new [`EnsembleStatus`] from a set of starting positions `x0` and `max_steps`
    ///
    /// # See Also
    /// [`Walker::new`]
    pub fn new(x0: Vec<DVector<Float>>) -> Self {
        Self {
            walkers: x0.into_iter().map(Walker::new).collect(),
        }
    }
    /// Get the dimension of the [`EnsembleStatus`] `(n_walkers, n_steps, n_variables)`
    pub fn dimension(&self) -> (usize, usize, usize) {
        let n_walkers = self.walkers.len();
        let (n_steps, n_variables) = self.walkers[0].dimension();
        (n_walkers, n_steps, n_variables)
    }
    /// Add a set of positions to the [`EnsembleStatus`], adding each position to the corresponding
    /// [`Walker`] in the given order
    pub fn push(&mut self, positions: Vec<Arc<RwLock<Point>>>) {
        self.walkers
            .iter_mut()
            .zip(positions)
            .for_each(|(walker, position)| {
                walker.push(position);
            });
    }
    /// Evaluate the most recent position of all [`Walker`]s in the [`EnsembleStatus`]
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
        for walker in self.walkers.iter_mut() {
            walker.evaluate_latest(func, user_data)?;
        }
        Ok(())
    }
    /// Randomly draw a [`Walker`] from the [`EnsembleStatus`] other than the one at the provided `index`
    pub fn get_compliment_walker(&self, index: usize, rng: &mut Rng) -> Walker {
        let n_tot = self.walkers.len();
        let r = rng.usize(0..n_tot - 1);
        let j = if r >= index { r + 1 } else { r };
        self.walkers[j].clone()
    }
    /// Randomly draw `n` [`Walker`]s from the [`EnsembleStatus`] other than the one at the provided `index`
    ///
    /// # Panics
    ///
    /// This method will panic if you try to draw more [`Walker`]s than are in the [`EnsembleStatus`]
    /// (aside from the excluded one at the provided `index`).
    pub fn get_compliment_walkers(&self, index: usize, n: usize, rng: &mut Rng) -> Vec<Walker> {
        assert!(n < self.walkers.len());
        let mut indices: Vec<usize> = (0..self.walkers.len()).filter(|&i| i != index).collect();
        rng.shuffle(&mut indices);
        indices[..n]
            .iter()
            .map(|&j| self.walkers[j].clone())
            .collect()
    }
    /// Get the average position of all [`Walker`]s
    pub fn mean(&self) -> DVector<Float> {
        self.walkers
            .iter()
            .map(|walker| walker.get_latest().read().x.clone())
            .sum()
    }
    /// Get the average position of all [`Walker`]s except for the one at the provided `index`
    pub fn mean_compliment(&self, index: usize) -> DVector<Float> {
        self.walkers
            .iter()
            .enumerate()
            .filter_map(|(i, walker)| {
                if i != index {
                    Some(walker.get_latest().read().x.clone())
                } else {
                    None
                }
            })
            .sum::<DVector<Float>>()
            .unscale(self.walkers.len() as Float)
    }
    /// Iterate through all the [`Walker`]s other than the one at the provided `index`
    pub fn iter_compliment(&self, index: usize) -> impl Iterator<Item = Arc<RwLock<Point>>> + '_ {
        self.walkers
            .iter()
            .enumerate()
            .filter_map(move |(i, walker)| {
                if i != index {
                    Some(walker.get_latest())
                } else {
                    None
                }
            })
    }
    /// Get a [`Vec`] containing a [`Vec`] of positions for each [`Walker`] in the ensemble
    ///
    /// If `burn` is [`None`], no burn-in will be performed, otherwise the given number of steps
    /// will be discarded from the beginning of each [`Walker`]'s history.
    ///
    /// If `thin` is [`None`], no thinning will be performed, otherwise every `thin`-th step will
    /// be discarded from the [`Walker`]'s history.
    pub fn get_chain(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<Vec<DVector<Float>>> {
        let burn = burn.unwrap_or(0);
        let thin = thin.unwrap_or(1);
        self.walkers
            .iter()
            .map(|walker| {
                walker
                    .history
                    .iter()
                    .skip(burn)
                    .enumerate()
                    .filter_map(|(i, position)| {
                        if i % thin == 0 {
                            Some(position.read().x.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect()
    }
    /// Get a [`Vec`] containing positions for each [`Walker`] in the ensemble, flattened
    ///
    /// If `burn` is [`None`], no burn-in will be performed, otherwise the given number of steps
    /// will be discarded from the beginning of each [`Walker`]'s history.
    ///
    /// If `thin` is [`None`], no thinning will be performed, otherwise every `thin`-th step will
    /// be discarded from the [`Walker`]'s history.
    pub fn get_flat_chain(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<DVector<Float>> {
        let chain = self.get_chain(burn, thin);
        chain.into_iter().flatten().collect()
    }

    /// Returns a matrix with the latest position of each walker in the ensemble with dimensions
    /// `(n_walkers, n_variables)`
    pub fn get_latest_position_matrix(&self) -> DMatrix<Float> {
        let position: Vec<RowDVector<Float>> = self
            .walkers
            .iter()
            .map(|walker| walker.get_latest().read().x.clone().transpose())
            .collect::<Vec<RowDVector<Float>>>();
        DMatrix::from_rows(position.as_slice())
    }

    /// Calculate the integrated autocorrelation time for each parameter according to Karamanis &
    /// Beutler[^Karamanis]
    ///
    /// `c` is an optional window size (`7.0` if [`None`] provided), see Sokal[^Sokal].
    ///
    /// If `burn` is [`None`], no burn-in will be performed, otherwise the given number of steps
    /// will be discarded from the beginning of each [`Walker`]'s history.
    ///
    /// If `thin` is [`None`], no thinning will be performed, otherwise every `thin`-th step will
    /// be discarded from the [`Walker`]'s history.
    ///
    /// [^Karamanis]: Karamanis, M., & Beutler, F. (2020). Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions. arXiv Preprint arXiv: 2002. 06212.
    /// [^Sokal]: Sokal, A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms. In C. DeWitt-Morette, P. Cartier, & A. Folacci (Eds.), Functional Integration: Basics and Applications (pp. 131–192). doi:10.1007/978-1-4899-0319-8_6
    pub fn get_integrated_autocorrelation_times(
        &self,
        c: Option<Float>,
        burn: Option<usize>,
        thin: Option<usize>,
    ) -> DVector<Float> {
        let samples = self.get_chain(burn, thin);
        integrated_autocorrelation_times(samples, c)
    }
}

impl Status for EnsembleStatus {
    fn reset(&mut self) {
        for walker in self.walkers.iter_mut() {
            walker.reset();
        }
    }

    fn converged(&self) -> bool {
        false
    }

    fn message(&self) -> &str {
        "TODO"
    }

    fn update_message(&mut self, message: &str) {}
}
