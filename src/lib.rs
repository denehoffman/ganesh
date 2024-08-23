//! `ganesh`, (/ɡəˈneɪʃ/) named after the Hindu god of wisdom, provides several common minimization algorithms as well as a straightforward, trait-based interface to create your own extensions. This crate is intended to be as simple as possible. The user needs to implement the [`Function`] trait on some struct which will take a vector of parameters and return a single-valued [`Result`] ($`f(\mathbb{R}^n) \to \mathbb{R}`$). Users can optionally provide a gradient function to speed up some algorithms, but a default central finite-difference implementation is provided so that all algorithms will work out of the box.
//!
//! <div class="warning">
//!
//! This crate is still in an early development phase, and the API is not stable. It can (and likely will) be subject to breaking changes before the 1.0.0 version release (and hopefully not many after that).
//!
//! </div>
//!
//! # Table of Contents
//! - [Key Features](#key-features)
//! - [Quick Start](#quick-start)
//! - [Bounds](#bounds)
//! - [Future Plans](#future-plans)
//!
//! # Key Features
//! * Simple but powerful trait-oriented library which tries to follow the Unix philosophy of "do one thing and do it well".
//! * Generics to allow for different numeric types to be used in the provided algorithms.
//! * Algorithms that are simple to use with sensible defaults.
//! * Traits which make developing future algorithms simple and consistent.
//!
//! # Quick Start
//!
//! This crate provides some common test functions in the [`test_functions`] module. Consider the following implementation of the Rosenbrock function:
//!
//! ```rust
//! use std::convert::Infallible;
//! use ganesh::prelude::*;
//!
//! pub struct Rosenbrock {
//!     pub n: usize,
//! }
//! impl Function<f64, (), Infallible> for Rosenbrock {
//!     fn evaluate(&self, x: &[f64], _user_data: &mut ()) -> Result<f64, Infallible> {
//!         Ok((0..(self.n - 1))
//!             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//!             .sum())
//!     }
//! }
//! ```
//! To minimize this function, we could consider using the Nelder-Mead algorithm:
//! ```rust
//! use ganesh::prelude::*;
//! use ganesh::algorithms::NelderMead;
//! # use std::convert::Infallible;
//!
//! # pub struct Rosenbrock {
//! #     pub n: usize,
//! # }
//! # impl Function<f64, (), Infallible> for Rosenbrock {
//! #     fn evaluate(&self, x: &[f64], _user_data: &mut ()) -> Result<f64, Infallible> {
//! #         Ok((0..(self.n - 1))
//! #             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//! #             .sum())
//! #     }
//! # }
//!
//! let problem = Rosenbrock { n: 2 };
//! let nm = NelderMead::default();
//! let mut m = Minimizer::new(nm, 2);
//! let x0 = &[2.0, 2.0];
//! m.minimize(&problem, x0, &mut ()).unwrap();
//! println!("{}", m.status);
//! ```
//!
//! This should output
//! ```shell
//! MSG:       term_f = STDDEV
//! X:         [0.9999999946231828, 0.9999999884539057]
//! F(X):      0.00000000000000009170942877687133
//! N_EVALS:   160
//! CONVERGED: true
//! ```
//!
//! # Bounds
//! All minimizers in `ganesh` have access to a feature which allows algorithms which usually function in unbounded parameter spaces to only return results inside a bounding box. This is done via a parameter transformation, the same one used by [`LMFIT`](https://lmfit.github.io/lmfit-py/) and [`MINUIT`](https://root.cern.ch/doc/master/classTMinuit.html). While the user inputs parameters within the bounds, unbounded algorithms can (and in practice will) convert those values to a set of unbounded "internal" parameters. When functions are called, however, these internal parameters are converted back into bounded "external" parameters, via the following transformations:
//!
//! Upper and lower bounds:
//! ```math
//! x_\text{int} = \arcsin\left(2\frac{x_\text{ext} - x_\text{min}}{x_\text{max} - x_\text{min}} - 1\right)
//! ```
//! ```math
//! x_\text{ext} = x_\text{min} + \left(\sin(x_\text{int}) + 1\right)\frac{x_\text{max} - x_\text{min}}{2}
//! ```
//! Upper bound only:
//! ```math
//! x_\text{int} = \sqrt{(x_\text{max} - x_\text{ext} + 1)^2 - 1}
//! ```
//! ```math
//! x_\text{ext} = x_\text{max} + 1 - \sqrt{x_\text{int}^2 + 1}
//! ```
//! Lower bound only:
//! ```math
//! x_\text{int} = \sqrt{(x_\text{ext} - x_\text{min} + 1)^2 - 1}
//! ```
//! ```math
//! x_\text{ext} = x_\text{min} - 1 + \sqrt{x_\text{int}^2 + 1}
//! ```
//! As noted in the documentation for both `LMFIT` and `MINUIT`, these bounds should be used with caution. They turn linear problems into nonlinear ones, which can mess with error propagation and even fit convergence, not to mention increase function complexity. Methods which output covariance matrices need to be adjusted if bounded, and `MINUIT` recommends fitting a second time near a minimum without bounds to ensure proper error propagation.
//!
//! # Future Plans
//!
//! * Eventually, I would like to implement BGFS and variants, MCMC algorithms, and some more modern gradient-free optimization techniques.
//! * There are probably many optimizations and algorithm extensions that I'm missing right now because I just wanted to get it working first.
//! * A test suite
#![warn(
    clippy::nursery,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::doc_markdown,
    clippy::doc_link_with_quotes,
    clippy::missing_safety_doc,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::perf,
    clippy::style,
    missing_docs
)]

use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
};

use num::{Float, FromPrimitive};

/// Module containing minimization algorithms
pub mod algorithms;
/// Module containing [`Observer`]s
pub mod observers;
/// Module containing standard functions for testing algorithms
pub mod test_functions;

/// Prelude module containing everything someone should need to use this crate for non-development
/// purposes
pub mod prelude {
    pub use crate::{Algorithm, Bound, Function, Minimizer, Observer, Status};
}

#[macro_export]
/// Convenience macro for converting raw numeric values to a generic
macro_rules! convert {
    ($value:expr, $type:ty) => {{
        #[allow(clippy::unwrap_used)]
        <$type as num::NumCast>::from($value).unwrap()
    }};
}
/// An enum that describes a bound/limit on a parameter in a minimization.
///
/// [`Bound`]s take a generic `T` which represents some scalar numeric value. They can be used by
/// bounded [`Algorithm`]s directly, or by unbounded [`Algorithm`]s using parameter space
/// transformations (experimental).
#[derive(Default, Copy, Clone, Debug)]
pub enum Bound<T> {
    #[default]
    /// `(-inf, +inf)`
    NoBound,
    /// `(min, +inf)`
    LowerBound(T),
    /// `(-inf, max)`
    UpperBound(T),
    /// `(min, max)`
    LowerAndUpperBound(T, T),
}
impl<T> Display for Bound<T>
where
    T: Float + Debug + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.lower(), self.upper())
    }
}
impl<T> From<(T, T)> for Bound<T>
where
    T: Float,
{
    fn from(value: (T, T)) -> Self {
        assert!(value.0 < value.1);
        match (value.0.is_finite(), value.1.is_finite()) {
            (true, true) => Self::LowerAndUpperBound(value.0, value.1),
            (true, false) => Self::LowerBound(value.0),
            (false, true) => Self::UpperBound(value.1),
            (false, false) => Self::NoBound,
        }
    }
}
impl<T> Bound<T>
where
    T: Float + Debug,
{
    /// Checks whether the given `value` is compatible with the bounds.
    pub fn contains(&self, value: &T) -> bool {
        match self {
            Self::NoBound => true,
            Self::LowerBound(lb) => value > lb,
            Self::UpperBound(ub) => value < ub,
            Self::LowerAndUpperBound(lb, ub) => value > lb && value < ub,
        }
    }
    /// Returns the lower bound or `-inf` if there is none.
    pub fn lower(&self) -> T {
        match self {
            Self::NoBound => T::neg_infinity(),
            Self::LowerBound(lb) => *lb,
            Self::UpperBound(_) => T::neg_infinity(),
            Self::LowerAndUpperBound(lb, _) => *lb,
        }
    }
    /// Returns the upper bound or `+inf` if there is none.
    pub fn upper(&self) -> T {
        match self {
            Self::NoBound => T::infinity(),
            Self::LowerBound(_) => T::infinity(),
            Self::UpperBound(ub) => *ub,
            Self::LowerAndUpperBound(_, ub) => *ub,
        }
    }
    /// Converts an unbounded "external" parameter into a bounded "internal" one via the transform:
    ///
    /// Upper and lower bounds:
    /// ```math
    /// x_\text{int} = \arcsin\left(2\frac{x_\text{ext} - x_\text{min}}{x_\text{max} - x_\text{min}} - 1\right)
    /// ```
    /// Upper bound only:
    /// ```math
    /// x_\text{int} = \sqrt{(x_\text{max} - x_\text{ext} + 1)^2 - 1}
    /// ```
    /// Lower bound only:
    /// ```math
    /// x_\text{int} = \sqrt{(x_\text{ext} - x_\text{min} + 1)^2 - 1}
    /// ```
    pub fn to_bounded(values: &[T], bounds: &[Self]) -> Vec<T> {
        values
            .iter()
            .zip(bounds)
            .map(|(val, bound)| bound._to_bounded(*val))
            .collect()
    }
    fn _to_bounded(&self, val: T) -> T {
        match *self {
            Self::LowerBound(lb) => lb - T::one() + T::sqrt(T::powi(val, 2) + T::one()),
            Self::UpperBound(ub) => ub + T::one() - T::sqrt(T::powi(val, 2) + T::one()),
            Self::LowerAndUpperBound(lb, ub) => {
                lb + (T::sin(val) + T::one()) * (ub - lb) / (T::one() + T::one())
            }
            Self::NoBound => val,
        }
    }
    /// Converts a bounded "internal" parameter into an unbounded "external" one via the transform:
    ///
    /// Upper and lower bounds:
    /// ```math
    /// x_\text{ext} = x_\text{min} + \left(\sin(x_\text{int}) + 1\right)\frac{x_\text{max} - x_\text{min}}{2}
    /// ```
    /// Upper bound only:
    /// ```math
    /// x_\text{ext} = x_\text{max} + 1 - \sqrt{x_\text{int}^2 + 1}
    /// ```
    /// Lower bound only:
    /// ```math
    /// x_\text{ext} = x_\text{min} - 1 + \sqrt{x_\text{int}^2 + 1}
    /// ```
    pub fn to_unbounded(values: &[T], bounds: &[Self]) -> Vec<T> {
        values
            .iter()
            .zip(bounds)
            .map(|(val, bound)| bound._to_unbounded(*val))
            .collect()
    }
    fn _to_unbounded(&self, val: T) -> T {
        match *self {
            Self::LowerBound(lb) => T::sqrt(T::powi(val - lb + T::one(), 2) - T::one()),
            Self::UpperBound(ub) => T::sqrt(T::powi(ub - val + T::one(), 2) - T::one()),
            Self::LowerAndUpperBound(lb, ub) =>
            {
                #[allow(clippy::suspicious_operation_groupings)]
                T::asin((T::one() + T::one()) * (val - lb) / (ub - lb) - T::one())
            }
            Self::NoBound => val,
        }
    }
}

/// A trait which describes a function $`f(\mathbb{R}^n) \to \mathbb{R}`$
///
/// Such a function may also take a `user_data: &mut U` field which can be used to pass external
/// arguments to the function during minimization, or can be modified by the function itself.
///
/// The `Function` trait takes a generic `T` which represents a numeric scalar, a generic `U`
/// representing the type of user data/arguments, and a generic `E` representing any possible
/// errors that might be returned during function execution.
///
/// There is also a default implementation of a gradient function which uses a central
/// finite-difference method to evaluate derivatives. If an exact gradient is known, it can be used
/// to speed up gradient-dependent algorithms.
pub trait Function<T, U, E>
where
    T: Float + FromPrimitive,
{
    /// The evaluation of the function at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    fn evaluate(&self, x: &[T], user_data: &mut U) -> Result<T, E>;
    /// The evaluation of the gradient at a point `x` with the given arguments/user data. The
    /// gradient is passed by mutable reference as an input, which saves a bit on memory allocation
    /// and also provides a similar format to some other fitting libraries.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn gradient(&self, x: &[T], grad: &mut [T], user_data: &mut U) -> Result<(), E> {
        let n = x.len();
        let mut x = x.to_vec();
        let eps = T::cbrt(T::epsilon());
        let two_eps = eps * (T::one() + T::one());
        for i in 0..n {
            let xi = x[i];
            x[i] = xi + eps;
            let fm = self.evaluate(&x, user_data)?;
            x[i] = xi - eps;
            let fp = self.evaluate(&x, user_data)?;
            grad[i] = (fp - fm) / (two_eps);
            x[i] = xi;
        }
        Ok(())
    }
}

/// A status message struct containing all information about a minimization result.
#[derive(Debug, Default, Clone)]
pub struct Status<T> {
    /// A [`String`] message that can be set by minimization [`Algorithm`]s.
    pub message: String,
    /// The current position of the minimization.
    pub x: Vec<T>,
    /// The current value of the minimization problem function at [`Status::x`].
    pub fx: T,
    /// The number of function evaluations (approximately, this is left up to individual
    /// [`Algorithm`]s to correctly compute and may not be exact).
    pub n_evals: usize,
    /// Flag that says whether or not the fit is in a converged state.
    pub converged: bool,
}
impl<T> Status<T> {
    /// Updates the [`Status::message`] field.
    pub fn update_message(&mut self, message: &str) {
        self.message = message.to_string();
    }
    /// Updates the [`Status::x`] and [`Status::fx`] fields.
    pub fn update_position(&mut self, pos: (Vec<T>, T)) {
        self.x = pos.0;
        self.fx = pos.1;
    }
    /// Sets [`Status::converged`] to be `true`.
    pub fn set_converged(&mut self) {
        self.converged = true;
    }
    /// Increments [`Status::n_evals`] by `1`.
    pub fn increment_n_evals(&mut self) {
        self.n_evals += 1;
    }
}
impl<T> Display for Status<T>
where
    T: Debug + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MSG:       {}", self.message)?;
        writeln!(f, "X:         {:?}", self.x)?;
        writeln!(f, "F(X):      {}", self.fx)?;
        writeln!(f, "N_EVALS:   {}", self.n_evals)?;
        write!(f, "CONVERGED: {}", self.converged)
    }
}

/// A trait representing a minimization algorithm.
///
/// This trait is implemented for the algorithms found in the [`algorithms`] module, and contains
/// all the methods needed to be run by a [`Minimizer`].
pub trait Algorithm<T, U, E> {
    /// Any setup work done before the main steps of the algorithm should be done here.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn initialize(
        &mut self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
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
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E>;
    /// Runs any termination/convergence checks and returns true if the algorithm has converged.
    /// Developers should also update the internal [`Status`] of the algorithm here if converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn check_for_termination(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<bool, E>;
    /// Returns the internal [`Status`] of the algorithm. This is a field that all [`Algorithm`]s
    /// should probably contain.
    fn get_status(&self) -> &Status<T>;
    /// Runs any steps needed by the [`Algorithm`] after termination or convergence. This will run
    /// regardless of whether the [`Algorithm`] converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    #[allow(unused_variables)]
    fn postprocessing(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        Ok(())
    }
}

/// A trait which holds a [`callback`](`Observer::callback`) function that can be used to check an
/// [`Algorithm`]'s [`Status`] during a minimization.
pub trait Observer<T, U> {
    /// A function that is called at every step of a minimization [`Algorithm`].
    fn callback(&mut self, step: usize, status: &Status<T>, user_data: &mut U);
}

/// The main struct used for running [`Algorithm`]s on [`Function`]s.
pub struct Minimizer<T, U, E, A>
where
    A: Algorithm<T, U, E>,
{
    /// The [`Status`] of the [`Minimizer`], usually read after minimization.
    pub status: Status<T>,
    algorithm: A,
    bounds: Option<Vec<Bound<T>>>,
    max_steps: usize,
    observers: Vec<Box<dyn Observer<T, U>>>,
    dimension: usize,
    _phantom: PhantomData<E>,
}

impl<T, U, E, A: Algorithm<T, U, E>> Minimizer<T, U, E, A>
where
    T: Float + FromPrimitive + Debug + Display + Default,
{
    const DEFAULT_MAX_STEPS: usize = 4000;
    /// Creates a new [`Minimizer`] with the given [`Algorithm`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new(algorithm: A, dimension: usize) -> Self {
        Self {
            status: Status::default(),
            algorithm,
            bounds: None,
            max_steps: Self::DEFAULT_MAX_STEPS,
            observers: Vec::default(),
            dimension,
            _phantom: PhantomData,
        }
    }
    /// Set the [`Algorithm`] used by the [`Minimizer`].
    pub fn with_algorithm(mut self, algorithm: A) -> Self {
        self.algorithm = algorithm;
        self
    }
    /// Set the maximum number of steps to perform before failure (default: 4000).
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
    /// Sets the current list of [`Observer`]s of the [`Minimizer`].
    pub fn with_observers(mut self, observers: Vec<Box<dyn Observer<T, U>>>) -> Self {
        self.observers = observers;
        self
    }
    /// Adds a single [`Observer`] to the [`Minimizer`].
    pub fn with_observer<O>(mut self, observer: O) -> Self
    where
        O: Observer<T, U> + 'static,
    {
        self.observers.push(Box::new(observer));
        self
    }
    /// Sets all [`Bound`]s of the [`Minimizer`]. This can be [`None`] for an unbounded problem, or
    /// [`Some`] [`Vec<(T, T)>`] with length equal to the number of free parameters. Individual
    /// upper or lower bounds can be unbounded by setting them equal to `T::infinity()` or
    /// `T::neg_infinity()` (e.g. `f64::INFINITY` and `f64::NEG_INFINITY`).
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bounds is not equal to the number of free
    /// parameters.
    pub fn with_bounds(mut self, bounds: Option<Vec<(T, T)>>) -> Self {
        if let Some(bounds) = bounds {
            assert!(bounds.len() == self.dimension);
            self.bounds = Some(bounds.into_iter().map(Bound::from).collect());
        } else {
            self.bounds = None
        }
        self
    }
    /// Sets the [`Bound`] of the parameter at the given index.
    pub fn with_bound(mut self, index: usize, bound: Option<(T, T)>) -> Self {
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
    /// This method first runs [`Algorithm::initialize`], then runs [`Algorithm::step`] in a loop,
    /// terminating if [`Algorithm::check_for_termination`] returns `true` or if
    /// the maximum number of allowed steps is exceeded. Each step will be followed by a sequential
    /// call to all given [`Observer`]s' callback functions, which will use the [`Status`] received
    /// from that step's call to [`Algorithm::get_status`]. Finally, regardless of convergence,
    /// [`Algorithm::postprocessing`] is called. If the algorithm did not converge in the given
    /// step limit, the [`Status::message`] will be set to `"MAX EVALS"` at termination.
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
    /// [`Minimizer`].
    pub fn minimize(
        &mut self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        user_data: &mut U,
    ) -> Result<(), E> {
        assert!(x0.len() == self.dimension);
        if let Some(bounds) = &self.bounds {
            for (i, (x_i, bound_i)) in x0.iter().zip(bounds).enumerate() {
                assert!(
                    bound_i.contains(x_i),
                    "Parameter #{} = {} is outside of the given bound: {}",
                    i,
                    x_i,
                    bound_i
                )
            }
        }
        self.algorithm
            .initialize(func, x0, self.bounds.as_ref(), user_data)?;
        let mut current_step = 0;
        while current_step <= self.max_steps
            && !self
                .algorithm
                .check_for_termination(func, self.bounds.as_ref(), user_data)?
        {
            self.algorithm.step(func, self.bounds.as_ref(), user_data)?;
            current_step += 1;
            if !self.observers.is_empty() {
                let status = self.algorithm.get_status();
                for observer in self.observers.iter_mut() {
                    observer.callback(current_step, status, user_data);
                }
            }
        }
        self.algorithm
            .postprocessing(func, self.bounds.as_ref(), user_data)?;
        let mut status = self.algorithm.get_status().clone();
        if current_step == self.max_steps && !status.converged {
            status.update_message("MAX EVALS");
        }
        self.status = status;
        Ok(())
    }
}
