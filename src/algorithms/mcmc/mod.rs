#![allow(dead_code, unused_variables)]
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use dyn_clone::DynClone;
use fastrand::Rng;
use nalgebra::{Complex, DVector};
use parking_lot::RwLock;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};

use crate::{init_ctrl_c_handler, is_ctrl_c_pressed, reset_ctrl_c_handler, Bound, Float, Function};

use super::Point;

/// Affine Invariant MCMC Ensemble Sampler
pub mod aies;
pub use aies::{AIESMove, AIES};

/// Ensemble Slice Sampler
pub mod ess;
pub use ess::{ESSMove, ESS};

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
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    pub fn evaluate_latest<U, E>(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.get_latest().write().evaluate(func, user_data)
    }
    /// Evaluate the most recent position of the [`Walker`]
    ///
    /// This function assumes `x` is an internal, unbounded vector, but performs a coordinate transform
    /// to bound `x` when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    pub fn evaluate_latest_bounded<U, E>(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.get_latest()
            .write()
            .evaluate_bounded(func, bounds, user_data)
    }
    /// Add a new position to the [`Walker`]'s history
    pub fn push(&mut self, position: Arc<RwLock<Point>>) {
        self.history.push(position)
    }
}

/// A collection of [`Walker`]s
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ensemble {
    walkers: Vec<Walker>,
}
impl Deref for Ensemble {
    type Target = Vec<Walker>;

    fn deref(&self) -> &Self::Target {
        &self.walkers
    }
}

impl DerefMut for Ensemble {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.walkers
    }
}
impl Ensemble {
    /// Create a new [`Ensemble`] from a set of starting positions `x0` and `max_steps`
    ///
    /// # See Also
    /// [`Walker::new`]
    pub fn new(x0: Vec<DVector<Float>>) -> Self {
        Self {
            walkers: x0.into_iter().map(Walker::new).collect(),
        }
    }
    /// Get the dimension of the Ensemble `(n_walkers, n_steps, n_variables)`
    pub fn dimension(&self) -> (usize, usize, usize) {
        let n_walkers = self.walkers.len();
        let (n_steps, n_variables) = self.walkers[0].dimension();
        (n_walkers, n_steps, n_variables)
    }
    /// Add a set of positions to the [`Ensemble`], adding each position to the corresponding
    /// [`Walker`] in the given order
    pub fn push(&mut self, positions: Vec<Arc<RwLock<Point>>>) {
        self.walkers
            .iter_mut()
            .zip(positions)
            .for_each(|(walker, position)| {
                walker.push(position);
            });
    }
    /// Reset all [`Walker`]s in the [`Ensemble`] (except for their starting position)
    pub fn reset(&mut self) {
        for walker in self.walkers.iter_mut() {
            walker.reset();
        }
    }
    /// Evaluate the most recent position of all [`Walker`]s in the [`Ensemble`]
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    pub fn evaluate_latest<U, E>(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
    ) -> Result<(), E> {
        for walker in self.walkers.iter_mut() {
            walker.evaluate_latest(func, user_data)?;
        }
        Ok(())
    }
    /// Evaluate the most recent position of all [`Walker`]s in the [`Ensemble`]
    ///
    /// This function assumes `x` is an internal, unbounded vector, but performs a coordinate transform
    /// to bound `x` when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    pub fn evaluate_latest_bounded<U, E>(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        for walker in self.walkers.iter_mut() {
            walker.evaluate_latest_bounded(func, bounds, user_data)?;
        }
        Ok(())
    }
    /// Randomly draw a [`Walker`] from the [`Ensemble`] other than the one at the provided `index`
    pub fn get_compliment_walker(&self, index: usize, rng: &mut Rng) -> Walker {
        let n_tot = self.walkers.len();
        let r = rng.usize(0..n_tot - 1);
        let j = if r >= index { r + 1 } else { r };
        self.walkers[j].clone()
    }
    /// Randomly draw `n` [`Walker`]s from the [`Ensemble`] other than the one at the provided `index`
    ///
    /// # Panics
    ///
    /// This method will panic if you try to draw more [`Walker`]s than are in the [`Ensemble`]
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

/// Calculate the integrated autocorrelation time for each parameter according to Karamanis &
/// Beutler[^Karamanis]
///
/// `samples` should have the shape `(n_walkers, n_steps, n_parameters)`.
///
/// `c` is an optional window size (`7.0` if [`None`] provided), see Sokal[^Sokal].
///
/// This is a standalone function that can be used to bypass the [`Ensemble`] struct and calculate
/// IATs for custom inputs.
///
/// [^Karamanis]: Karamanis, M., & Beutler, F. (2020). Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions. arXiv Preprint arXiv: 2002. 06212.
/// [^Sokal]: Sokal, A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms. In C. DeWitt-Morette, P. Cartier, & A. Folacci (Eds.), Functional Integration: Basics and Applications (pp. 131–192). doi:10.1007/978-1-4899-0319-8_6
pub fn integrated_autocorrelation_times(
    samples: Vec<Vec<DVector<Float>>>,
    c: Option<Float>,
) -> DVector<Float> {
    let c = c.unwrap_or(7.0);
    let n_walkers = samples.len();
    let n_steps = samples[0].len();
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

/// A trait representing an MCMC algorithm.
///
/// This trait is implemented for the MCMC algorithms found in the
/// [`algorithms::mcmc`](crate::algorithms::mcmc) module, and contains
/// all the methods needed to be run by a [`Sampler`].
pub trait MCMCAlgorithm<U, E>: DynClone {
    /// Any setup work done before the main steps of the algorithm should be done here.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        ensemble: &mut Ensemble,
    ) -> Result<(), E>;
    /// The main "step" of an algorithm, which is repeated until termination conditions are met or
    /// the max number of steps have been taken.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn step(
        &mut self,
        i_step: usize,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        ensemble: &mut Ensemble,
    ) -> Result<(), E>;
    /// Runs any termination/convergence checks and returns true if the algorithm has converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn check_for_termination(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        ensemble: &mut Ensemble,
    ) -> Result<bool, E>;
    /// Runs any steps needed by the [`MCMCAlgorithm`] after termination.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    #[allow(unused_variables)]
    fn postprocessing(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        ensemble: &mut Ensemble,
    ) -> Result<(), E> {
        Ok(())
    }
}
dyn_clone::clone_trait_object!(<U, E> MCMCAlgorithm<U, E>);

/// A trait which holds a [`callback`](`MCMCObserver::callback`) function that can be used to check an
/// [`MCMCAlgorithm`]'s [`Ensemble`] during sampling.
pub trait MCMCObserver<U> {
    /// A function that is called at every step of a sampling [`MCMCAlgorithm`]. If it returns
    /// `false`, the [`Sampler::sample`] method will terminate.
    fn callback(&mut self, step: usize, ensemble: &mut Ensemble, user_data: &mut U) -> bool;
}

/// The main struct used for running [`MCMCAlgorithm`]s on [`Function`]s.
pub struct Sampler<U, E> {
    /// The chains of walker positions created during sampling
    pub ensemble: Ensemble,
    mcmc_algorithm: Box<dyn MCMCAlgorithm<U, E>>,
    bounds: Option<Vec<Bound>>,
    observers: Vec<Arc<RwLock<dyn MCMCObserver<U>>>>,
}

impl<U, E> Sampler<U, E> {
    /// Creates a new [`Sampler`] with the given [`MCMCAlgorithm`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new<M: MCMCAlgorithm<U, E> + 'static>(mcmc: &M, x0: Vec<DVector<Float>>) -> Self {
        Self {
            ensemble: Ensemble::new(x0),
            mcmc_algorithm: Box::new(dyn_clone::clone(mcmc)),
            bounds: None,
            observers: Vec::default(),
        }
    }
    /// Creates a new [`Sampler`] with the given (boxed) [`MCMCAlgorithm`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new_from_box(
        mcmc_algorithm: Box<dyn MCMCAlgorithm<U, E>>,
        x0: Vec<DVector<Float>>,
    ) -> Self {
        Self {
            ensemble: Ensemble::new(x0),
            mcmc_algorithm,
            bounds: None,
            observers: Vec::default(),
        }
    }
    fn reset(&mut self) {
        self.ensemble.reset();
    }
    /// Set the [`MCMCAlgorithm`] used by the [`Sampler`].
    pub fn with_mcmc_algorithm<M: MCMCAlgorithm<U, E> + 'static>(
        mut self,
        mcmc_algorithm: &M,
    ) -> Self {
        self.mcmc_algorithm = Box::new(dyn_clone::clone(mcmc_algorithm));
        self
    }
    /// Adds a single [`MCMCObserver`] to the [`Sampler`].
    pub fn with_observer(mut self, observer: Arc<RwLock<dyn MCMCObserver<U>>>) -> Self {
        self.observers.push(observer);
        self
    }
    /// Sets all [`Bound`]s of the [`Sampler`]. This can be [`None`] for an unbounded problem, or
    /// [`Some`] [`Vec<(T, T)>`] with length equal to the number of free parameters. Individual
    /// upper or lower bounds can be unbounded by setting them equal to `T::infinity()` or
    /// `T::neg_infinity()` (e.g. `f64::INFINITY` and `f64::NEG_INFINITY`).
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bounds is not equal to the number of free
    /// parameters.
    pub fn with_bounds(mut self, bounds: Option<Vec<(Float, Float)>>) -> Self {
        if let Some(bounds) = bounds {
            assert!(bounds.len() == self.ensemble.dimension().2);
            self.bounds = Some(bounds.into_iter().map(Bound::from).collect());
        } else {
            self.bounds = None
        }
        self
    }
    /// Sets the [`Bound`] of the parameter at the given index.
    pub fn with_bound(mut self, index: usize, bound: Option<(Float, Float)>) -> Self {
        if let Some(bounds) = &mut self.bounds {
            if let Some(bound) = bound {
                bounds[index] = Bound::from(bound);
            } else {
                bounds[index] = Bound::NoBound;
            }
        } else {
            let mut bounds = vec![Bound::default(); self.ensemble.dimension().2];
            if let Some(bound) = bound {
                bounds[index] = Bound::from(bound);
            } else {
                bounds[index] = Bound::NoBound;
            }
            self.bounds = Some(bounds);
        }
        self
    }
    /// Minimize the given [`Function`] starting at the point `x0`.
    ///
    /// This method first runs [`MCMCAlgorithm::initialize`], then runs [`MCMCAlgorithm::step`] in a loop,
    /// terminating if [`MCMCAlgorithm::check_for_termination`] returns `true` or if
    /// the maximum number of allowed steps is exceeded. Each step will be followed by a sequential
    /// call to all given [`MCMCObserver`]s' callback functions. Finally, regardless of convergence,
    /// [`MCMCAlgorithm::postprocessing`] is called.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    pub fn sample(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        n_steps: usize,
    ) -> Result<(), E> {
        init_ctrl_c_handler();
        reset_ctrl_c_handler();
        self.mcmc_algorithm.initialize(
            func,
            self.bounds.as_ref(),
            user_data,
            &mut self.ensemble,
        )?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step < n_steps - 1 // the first step is the initial position
            && !observer_termination
            && !self.mcmc_algorithm.check_for_termination(
                func,
                self.bounds.as_ref(),
                user_data,
                &mut self.ensemble,
            )?
            && !is_ctrl_c_pressed()
        {
            let walker_step = self.ensemble.dimension().1;
            self.mcmc_algorithm.step(
                walker_step + 1,
                func,
                self.bounds.as_ref(),
                user_data,
                &mut self.ensemble,
            )?;
            current_step += 1;
            if !self.observers.is_empty() {
                for observer in self.observers.iter_mut() {
                    observer_termination =
                        observer
                            .write()
                            .callback(walker_step + 1, &mut self.ensemble, user_data)
                            || observer_termination;
                }
            }
        }
        self.mcmc_algorithm.postprocessing(
            func,
            self.bounds.as_ref(),
            user_data,
            &mut self.ensemble,
        )?;
        Ok(())
    }
    /// Get a [`Vec`] containing a [`Vec`] of positions for each [`Walker`] in the ensemble
    ///
    /// If `burn` is [`None`], no burn-in will be performed, otherwise the given number of steps
    /// will be discarded from the beginning of each [`Walker`]'s history.
    ///
    /// If `thin` is [`None`], no thinning will be performed, otherwise every `thin`-th step will
    /// be discarded from the [`Walker`]'s history.
    pub fn get_chains(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<Vec<DVector<Float>>> {
        self.ensemble.get_chain(burn, thin)
    }
    /// Get a [`Vec`] containing positions for each [`Walker`] in the ensemble, flattened
    ///
    /// If `burn` is [`None`], no burn-in will be performed, otherwise the given number of steps
    /// will be discarded from the beginning of each [`Walker`]'s history.
    ///
    /// If `thin` is [`None`], no thinning will be performed, otherwise every `thin`-th step will
    /// be discarded from the [`Walker`]'s history.
    pub fn get_flat_chain(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<DVector<Float>> {
        self.ensemble.get_flat_chain(burn, thin)
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
        self.ensemble
            .get_integrated_autocorrelation_times(c, burn, thin)
    }
}
