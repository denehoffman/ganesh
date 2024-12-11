#![allow(dead_code, unused_variables)]
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use dyn_clone::DynClone;
use fastrand::Rng;
use nalgebra::DVector;
use parking_lot::RwLock;

use crate::{
    init_ctrl_c_handler, is_ctrl_c_pressed, reset_ctrl_c_handler, Bound, Float, Function,
    SampleFloat,
};

use super::Point;

/// Affine Invariant MCMC Ensemble Sampler
pub mod aimes;

/// Ensemble Slice Sampler
pub mod ess;

#[derive(Clone, Debug)]
pub struct Walker {
    history: Vec<Arc<RwLock<Point>>>,
}

impl Walker {
    pub fn new(x0: DVector<Float>, max_steps: usize) -> Self {
        let mut history = Vec::with_capacity(max_steps);
        history.push(Arc::new(RwLock::new(Point::from(x0))));
        Walker { history }
    }
    pub fn reset(&mut self) {
        self.history = Vec::with_capacity(self.history.capacity());
    }
    pub fn get_latest(&self) -> Arc<RwLock<Point>> {
        assert!(self.history.len() > 0);
        self.history[self.history.len() - 1].clone()
    }
    pub fn evaluate_latest<U, E>(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.get_latest().write().evaluate(func, user_data)
    }
    pub fn push(&mut self, position: Arc<RwLock<Point>>) {
        self.history.push(position)
    }
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
}

/// A collection of walkers for MCMC
#[derive(Clone, Debug)]
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
    pub fn new(x0: Vec<DVector<Float>>, max_steps: usize) -> Self {
        Ensemble {
            walkers: x0
                .into_iter()
                .map(|pos| Walker::new(pos, max_steps))
                .collect(),
        }
    }
    // pub fn get_latest(&self) -> Vec<Arc<RwLock<Point>>> {
    //     self.walkers
    //         .iter()
    //         .map(|walker| walker.get_latest())
    //         .collect()
    // }
    pub fn push(&mut self, positions: Vec<Arc<RwLock<Point>>>) {
        self.walkers
            .iter_mut()
            .zip(positions.into_iter())
            .for_each(|(walker, position)| {
                walker.push(position);
            });
    }
    pub fn reset(&mut self) {
        for walker in self.walkers.iter_mut() {
            walker.reset();
        }
    }
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
    pub fn get_compliment_walker(&self, index: usize, rng: &mut Rng) -> Walker {
        let n_tot = self.walkers.len();
        let r = rng.usize(0..n_tot - 1);
        let j = if r >= index { r + 1 } else { r };
        self.walkers[j].clone()
    }
    pub fn get_compliment_walkers(&self, index: usize, n: usize, rng: &mut Rng) -> Vec<Walker> {
        assert!(n < self.walkers.len());
        let mut indices: Vec<usize> = (0..self.walkers.len()).filter(|&i| i != index).collect();
        rng.shuffle(&mut indices);
        indices[..n]
            .iter()
            .map(|&j| self.walkers[j].clone())
            .collect()
    }
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
    pub fn get_flat_chain(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<DVector<Float>> {
        let chain = self.get_chain(burn, thin);
        chain.into_iter().flatten().collect()
    }
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
/// [`MCMCAlgorithm`]'s [`Chains`] during a minimization.
pub trait MCMCObserver<U> {
    /// A function that is called at every step of a minimization [`Algorithm`]. If it returns
    /// `false`, the [`Minimizer::minimize`] method will terminate.
    fn callback(&mut self, step: usize, ensemble: &mut Ensemble, user_data: &mut U) -> bool;
}

/// The main struct used for running [`MCMCAlgorithm`]s on [`Function`]s.
pub struct Sampler<U, E> {
    /// The chains of walker positions created during sampling
    pub ensemble: Ensemble,
    mcmc_algorithm: Box<dyn MCMCAlgorithm<U, E>>,
    bounds: Option<Vec<Bound>>,
    max_steps: usize,
    observers: Vec<Box<dyn MCMCObserver<U>>>,
    dimension: usize,
}

impl<U, E> Sampler<U, E> {
    const DEFAULT_MAX_STEPS: usize = 4000;
    /// Creates a new [`Sampler`] with the given [`MCMCAlgorithm`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new<M: MCMCAlgorithm<U, E> + 'static>(
        mcmc: &M,
        x0: Vec<DVector<Float>>,
        dimension: usize,
    ) -> Self {
        Self {
            ensemble: Ensemble::new(x0, Self::DEFAULT_MAX_STEPS),
            mcmc_algorithm: Box::new(dyn_clone::clone(mcmc)),
            bounds: None,
            max_steps: Self::DEFAULT_MAX_STEPS,
            observers: Vec::default(),
            dimension,
        }
    }
    /// Creates a new [`Sampler`] with the given (boxed) [`MCMCAlgorithm`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new_from_box(
        mcmc_algorithm: Box<dyn MCMCAlgorithm<U, E>>,
        x0: Vec<DVector<Float>>,
        dimension: usize,
    ) -> Self {
        Self {
            ensemble: Ensemble::new(x0, Self::DEFAULT_MAX_STEPS),
            mcmc_algorithm,
            bounds: None,
            max_steps: Self::DEFAULT_MAX_STEPS,
            observers: Vec::default(),
            dimension,
        }
    }
    fn reset(&mut self) {
        self.ensemble.reset();
    }
    /// Set the [`Algorithm`] used by the [`Minimizer`].
    pub fn with_mcmc_algorithm<M: MCMCAlgorithm<U, E> + 'static>(
        mut self,
        mcmc_algorithm: &M,
    ) -> Self {
        self.mcmc_algorithm = Box::new(dyn_clone::clone(mcmc_algorithm));
        self
    }
    /// Set the maximum number of steps to perform before failure (default: 4000).
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        let old_max = self.max_steps;
        self.max_steps = max_steps;
        if max_steps > old_max {
            let diff = max_steps - old_max;
            for walker in self.ensemble.iter_mut() {
                walker.history.reserve_exact(diff);
            }
        }
        self
    }
    /// Sets the current list of [`MCMCObserver`]s of the [`Sampler`].
    pub fn with_observers(mut self, observers: Vec<Box<dyn MCMCObserver<U>>>) -> Self {
        self.observers = observers;
        self
    }
    /// Adds a single [`MCMCObserver`] to the [`Sampler`].
    pub fn with_observer<O>(mut self, observer: O) -> Self
    where
        O: MCMCObserver<U> + 'static,
    {
        self.observers.push(Box::new(observer));
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
            assert!(bounds.len() == self.dimension);
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
            let mut bounds = vec![Bound::default(); self.dimension];
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
    /// call to all given [`Observer`]s' callback functions. Finally, regardless of convergence,
    /// [`MCMCAlgorithm::postprocessing`] is called.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    ///
    /// # Panics
    ///
    /// This method will panic if the length of `x0` is not equal to the dimension of the problem
    /// (number of free parameters) or if any values of `x0` are outside the [`Bound`]s given to the
    /// [`Sampler`].
    pub fn sample(&mut self, func: &dyn Function<U, E>, user_data: &mut U) -> Result<(), E> {
        // assert!(x0.len() == self.dimension);
        // init_ctrl_c_handler();
        // reset_ctrl_c_handler();
        // if let Some(bounds) = &self.bounds {
        //     for (i, (x_i, bound_i)) in x0.iter().zip(bounds).enumerate() {
        //         assert!(
        //             bound_i.contains(*x_i),
        //             "Parameter #{} = {} is outside of the given bound: {}",
        //             i,
        //             x_i,
        //             bound_i
        //         )
        //     }
        // }
        // self.status.x0 = DVector::from_column_slice(x0);
        self.mcmc_algorithm.initialize(
            func,
            self.bounds.as_ref(),
            user_data,
            &mut self.ensemble,
        )?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step <= self.max_steps
            && !observer_termination
            && !self.mcmc_algorithm.check_for_termination(
                func,
                self.bounds.as_ref(),
                user_data,
                &mut self.ensemble,
            )?
            && !is_ctrl_c_pressed()
        {
            self.mcmc_algorithm.step(
                current_step,
                func,
                self.bounds.as_ref(),
                user_data,
                &mut self.ensemble,
            )?;
            current_step += 1;
            if !self.observers.is_empty() {
                for observer in self.observers.iter_mut() {
                    observer_termination =
                        !observer.callback(current_step, &mut self.ensemble, user_data)
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
        // if is_ctrl_c_pressed() {
        // self.status.update_message("Ctrl-C Pressed");
        // }
        Ok(())
    }
    pub fn get_chain(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<Vec<DVector<Float>>> {
        let burn = burn.unwrap_or(0);
        let thin = thin.unwrap_or(1);
        self.ensemble
            .walkers
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
    pub fn get_flat_chain(&self, burn: Option<usize>, thin: Option<usize>) -> Vec<DVector<Float>> {
        let chain = self.get_chain(burn, thin);
        chain.into_iter().flatten().collect()
    }
}
