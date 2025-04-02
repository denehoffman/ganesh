//! `ganesh` (/ɡəˈneɪʃ/), named after the Hindu god of wisdom, provides several common minimization algorithms as well as a straightforward, trait-based interface to create your own extensions. This crate is intended to be as simple as possible. The user needs to implement the [`Function`] trait on some struct which will take a vector of parameters and return a single-valued [`Result`] ($`f(\mathbb{R}^n) \to \mathbb{R}`$). Users can optionally provide a gradient function to speed up some algorithms, but a default central finite-difference implementation is provided so that all algorithms will work out of the box.
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
//! - [MCMC](#mcmc)
//! - [Bounds](#bounds)
//! - [Future Plans](#future-plans)
//! - [Citations](#citations)
//!
//! # Key Features
//! * Simple but powerful trait-oriented library which tries to follow the Unix philosophy of "do one thing and do it well".
//! * Generics to allow for different numeric types to be used in the provided algorithms.
//! * Algorithms that are simple to use with sensible defaults.
//! * Traits which make developing future algorithms simple and consistent.
//! * Pressing `Ctrl-C` during a fit will still output a [`Status`], but the fit message will
//!   indicate that the fit was ended by the user.
//!
//! # Quick Start
//!
//! This crate provides some common test functions in the [`test_functions`] module. Consider the following implementation of the Rosenbrock function:
//!
//! ```rust
//! use std::convert::Infallible;
//! use ganesh::{Function, Float};
//!
//! pub struct Rosenbrock {
//!     pub n: usize,
//! }
//! impl Function<(), Infallible> for Rosenbrock {
//!     fn evaluate(&self, x: &[Float], _user_data: &mut ()) -> Result<Float, Infallible> {
//!         Ok((0..(self.n - 1))
//!             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//!             .sum())
//!     }
//! }
//! ```
//! To minimize this function, we could consider using the Nelder-Mead algorithm:
//! ```rust
//! use ganesh::{Function, Float, Minimizer};
//! use ganesh::algorithms::NelderMead;
//! # use std::convert::Infallible;
//!
//! # pub struct Rosenbrock {
//! #     pub n: usize,
//! # }
//! # impl Function<(), Infallible> for Rosenbrock {
//! #     fn evaluate(&self, x: &[Float], _user_data: &mut ()) -> Result<Float, Infallible> {
//! #         Ok((0..(self.n - 1))
//! #             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//! #             .sum())
//! #     }
//! # }
//! fn main() -> Result<(), Infallible> {
//!     let mut problem = Rosenbrock { n: 2 };
//!     let nm = NelderMead::default();
//!     let mut m = Minimizer::new(Box::new(nm), 2);
//!     let x0 = &[2.0, 2.0];
//!     m.minimize(&mut problem, x0, &mut ())?;
//!     println!("{}", m.status);
//!     Ok(())
//! }
//! ```
//!
//! This should output
//! ```shell
//!╒══════════════════════════════════════════════════════════════════════════════════════════════╕
//!│                                         FIT RESULTS                                          │
//!╞════════════════════════════════════════════╤════════════════════╤═════════════╤══════════════╡
//!│ Status: Converged                          │ fval:   +2.281E-16 │ #fcn:    76 │ #grad:    76 │
//!├────────────────────────────────────────────┴────────────────────┴─────────────┴──────────────┤
//!│ Message: term_f = STDDEV                                                                     │
//!├───────╥──────────────┬──────────────╥──────────────┬──────────────┬──────────────┬───────────┤
//!│ Par # ║        Value │  Uncertainty ║      Initial │       -Bound │       +Bound │ At Limit? │
//!├───────╫──────────────┼──────────────╫──────────────┼──────────────┼──────────────┼───────────┤
//!│     0 ║     +1.001E0 │    +8.461E-1 ║     +2.000E0 │         -inf │         +inf │           │
//!│     1 ║     +1.003E0 │     +1.695E0 ║     +2.000E0 │         -inf │         +inf │           │
//!└───────╨──────────────┴──────────────╨──────────────┴──────────────┴──────────────┴───────────┘
//! ```
//! # MCMC
//! Markov Chain Monte Carlo samplers can be found in the `mcmc` module, and an example can be found in `/examples/multivariate_normal_ess`:
//! ```shell
//! cd examples/multivariate_normal_ess
//! pip install -r requirements.txt
//! just
//! ```
//! if [`Just`](https://github.com/casey/just) is installed, or
//! ```shell
//! cd examples/multivariate_normal_ess
//! pip install -r requirements.txt
//! cargo r -r --example multivariate_normal_ess
//! python visualize.py
//! ```
//! to run manually.
//!
//! # Bounds
//! All minimizers in `ganesh` have access to a feature which allows algorithms which usually function in unbounded parameter spaces to only return results inside a bounding box. This is done via a parameter transformation, the same one used by [`LMFIT`](https://lmfit.github.io/lmfit-py/) and [`MINUIT`](https://root.cern.ch/doc/master/classTMinuit.html). This transform is not enacted on algorithms which already have bounded implementations, like [`L-BFGS-B`](`algorithms::lbfgsb`). While the user inputs parameters within the bounds, unbounded algorithms can (and in practice will) convert those values to a set of unbounded "internal" parameters. When functions are called, however, these internal parameters are converted back into bounded "external" parameters, via the following transformations:
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
//! * Eventually, I would like to implement some more modern gradient-free optimization techniques.
//! * There are probably many optimizations and algorithm extensions that I'm missing right now because I just wanted to get it working first.
//! * There should be more tests (as usual).
//!
//! # Citations
//! While this project does not currently have an associated paper, most of the algorithms it implements do, and they should be cited appropriately. Citations are also generally available in the documentation.
//!
//! ### ESS MCMC Sampler
//! ```text
//! @article{karamanis2020ensemble,
//!   title = {Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions},
//!   author = {Karamanis, Minas and Beutler, Florian},
//!   journal = {arXiv preprint arXiv: 2002.06212},
//!   year = {2020}
//! }
//! ```
//!
//! ### AIES MCMC Sampler
//! ```text
//! @article{Goodman2010,
//!   title = {Ensemble samplers with affine invariance},
//!   volume = {5},
//!   ISSN = {1559-3940},
//!   url = {http://dx.doi.org/10.2140/camcos.2010.5.65},
//!   DOI = {10.2140/camcos.2010.5.65},
//!   number = {1},
//!   journal = {Communications in Applied Mathematics and Computational Science},
//!   publisher = {Mathematical Sciences Publishers},
//!   author = {Goodman,  Jonathan and Weare,  Jonathan},
//!   year = {2010},
//!   month = jan,
//!   pages = {65–80}
//! }
//! ```
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
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Once,
    },
};

use fastrand::Rng;
use fastrand_contrib::RngExt;
use lazy_static::lazy_static;
use nalgebra::{Complex, DMatrix, DVector};
use parking_lot::RwLock;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};
use traits::{MCMCObserver, Observer};

/// Module containing minimization algorithms
pub mod algorithms;
/// Module containing [`Observer`]s
pub mod observers;
/// Module containing standard functions for testing algorithms
pub mod test_functions;

/// Module containing MCMC sampling algorithms
pub mod samplers;

/// Module containing useful traits
pub mod traits {
    pub use crate::observers::{MCMCObserver, Observer};
    pub use crate::{Algorithm, Function, MCMCAlgorithm, SampleFloat};
}

lazy_static! {
    pub(crate) static ref CTRL_C_PRESSED: AtomicBool = AtomicBool::new(false);
}

static INIT: Once = Once::new();

pub(crate) fn init_ctrl_c_handler() {
    INIT.call_once(|| {
        #[allow(clippy::expect_used)]
        ctrlc::set_handler(move || CTRL_C_PRESSED.store(true, Ordering::SeqCst))
            .expect("Error setting Ctrl-C handler");
    });
}

pub(crate) fn reset_ctrl_c_handler() {
    CTRL_C_PRESSED.store(false, Ordering::SeqCst)
}

pub(crate) fn is_ctrl_c_pressed() -> bool {
    CTRL_C_PRESSED.load(Ordering::SeqCst)
}

/// A floating-point number type (defaults to [`f64`], see `f32` feature).
#[cfg(not(feature = "f32"))]
pub type Float = f64;

/// A floating-point number type (defaults to [`f64`], see `f32` feature).
#[cfg(feature = "f32")]
pub type Float = f32;

/// A helper trait to get feature-gated floating-point random values
pub trait SampleFloat {
    /// Get a random value in a range
    fn range(&mut self, lower: Float, upper: Float) -> Float;
    /// Get a random value in the range [0, 1]
    fn float(&mut self) -> Float;
    /// Get a random Normal value
    fn normal(&mut self, mu: Float, sigma: Float) -> Float;
}
impl SampleFloat for Rng {
    #[cfg(not(feature = "f32"))]
    fn range(&mut self, lower: Float, upper: Float) -> Float {
        self.f64_range(lower..upper)
    }
    #[cfg(feature = "f32")]
    fn range(&mut self, lower: Float, upper: Float) -> Float {
        self.f32_range(lower..upper)
    }
    #[cfg(not(feature = "f32"))]
    fn float(&mut self) -> Float {
        self.f64()
    }
    #[cfg(feature = "f32")]
    fn float(&mut self) -> Float {
        self.f32()
    }
    #[cfg(not(feature = "f32"))]
    fn normal(&mut self, mu: Float, sigma: Float) -> Float {
        self.f64_normal(mu, sigma)
    }
    #[cfg(feature = "f32")]
    fn normal(&mut self, mu: Float, sigma: Float) -> Float {
        self.f32_normal(mu, sigma)
    }
}

/// A helper trait to provide a weighted random choice method
pub trait RandChoice {
    /// Return an random index sampled with the given weights
    fn choice_weighted(&mut self, weights: &[Float]) -> Option<usize>;
}

impl RandChoice for Rng {
    fn choice_weighted(&mut self, weights: &[Float]) -> Option<usize> {
        let total_weight = weights.iter().sum();
        let u: Float = self.range(0.0, total_weight);
        let mut cumulative_weight = 0.0;
        for (index, &weight) in weights.iter().enumerate() {
            cumulative_weight += weight;
            if u <= cumulative_weight {
                return Some(index);
            }
        }
        None
    }
}

/// Describes a point in parameter space that can be used in [`Algorithm`](`crate::Algorithm`)s.
#[derive(PartialEq, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Point {
    x: DVector<Float>,
    fx: Float,
}
impl Point {
    /// Get the dimension of the underlying space.
    #[allow(clippy::len_without_is_empty)]
    pub fn dimension(&self) -> usize {
        self.x.len()
    }
    /// Convert the [`Point`] into a [`Vec`]-`Float` tuple.
    pub fn into_vec_val(self) -> (Vec<Float>, Float) {
        let fx = self.get_fx_checked();
        (self.x.data.into(), fx)
    }
    /// Evaluate the given function at the point's coordinate and set the `fx` value to the result.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn evaluate<U, E>(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
    ) -> Result<(), E> {
        if self.fx.is_nan() {
            self.fx = func.evaluate(self.x.as_slice(), user_data)?;
        }
        Ok(())
    }
    /// Evaluate the given function at the point's coordinate and set the `fx` value to the result.
    /// This function assumes `x` is an internal, unbounded vector, but performs a coordinate transform
    /// to bound `x` when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn evaluate_bounded<U, E>(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        if self.fx.is_nan() {
            self.fx = func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?;
        }
        Ok(())
    }
    /// Compare two points by their `fx` value.
    pub fn total_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fx.total_cmp(&other.fx)
    }
    /// Move the point to a new position, resetting the evaluation of the point
    pub fn set_position(&mut self, x: DVector<Float>) {
        self.x = x;
        self.fx = Float::NAN;
    }
    /// Get the current evaluation of the point, if it has been evaluated
    ///
    /// # Panics
    ///
    /// This method will panic if the point is unevaluated.
    pub fn get_fx_checked(&self) -> Float {
        assert!(!self.fx.is_nan(), "Point value requested before evaluation");
        self.fx
    }
}

impl Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "fx: {}", self.fx)?;
        writeln!(f, "{}", self.x)
    }
}

impl From<DVector<Float>> for Point {
    fn from(value: DVector<Float>) -> Self {
        Self {
            x: value,
            fx: Float::NAN,
        }
    }
}
impl From<Vec<Float>> for Point {
    fn from(value: Vec<Float>) -> Self {
        Self {
            x: DVector::from_vec(value),
            fx: Float::NAN,
        }
    }
}
impl<'a> From<&'a Point> for &'a Vec<Float> {
    fn from(value: &'a Point) -> Self {
        value.x.data.as_vec()
    }
}
impl From<&[Float]> for Point {
    fn from(value: &[Float]) -> Self {
        Self {
            x: DVector::from_column_slice(value),
            fx: Float::NAN,
        }
    }
}
impl<'a> From<&'a Point> for &'a [Float] {
    fn from(value: &'a Point) -> Self {
        value.x.data.as_slice()
    }
}
impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.fx.partial_cmp(&other.fx)
    }
}

/// An enum that describes a bound/limit on a parameter in a minimization.
///
/// [`Bound`]s take a generic `T` which represents some scalar numeric value. They can be used by
/// bounded [`Algorithm`]s directly, or by unbounded [`Algorithm`]s using parameter space
/// transformations (experimental).
#[derive(Default, Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Bound {
    #[default]
    /// `(-inf, +inf)`
    NoBound,
    /// `(min, +inf)`
    LowerBound(Float),
    /// `(-inf, max)`
    UpperBound(Float),
    /// `(min, max)`
    LowerAndUpperBound(Float, Float),
}
impl Display for Bound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.lower(), self.upper())
    }
}
impl From<(Float, Float)> for Bound {
    fn from(value: (Float, Float)) -> Self {
        assert!(value.0 < value.1);
        match (value.0.is_finite(), value.1.is_finite()) {
            (true, true) => Self::LowerAndUpperBound(value.0, value.1),
            (true, false) => Self::LowerBound(value.0),
            (false, true) => Self::UpperBound(value.1),
            (false, false) => Self::NoBound,
        }
    }
}
impl From<(Option<Float>, Option<Float>)> for Bound {
    fn from(value: (Option<Float>, Option<Float>)) -> Self {
        assert!(value.0 < value.1);
        match (value.0, value.1) {
            (Some(lb), Some(ub)) => Self::LowerAndUpperBound(lb, ub),
            (Some(lb), None) => Self::LowerBound(lb),
            (None, Some(ub)) => Self::UpperBound(ub),
            (None, None) => Self::NoBound,
        }
    }
}
impl Bound {
    /// Get a value in the uniform distribution between `lower` and `upper`.
    #[cfg(not(feature = "f32"))]
    pub fn get_uniform(&self, rng: &mut Rng) -> Float {
        rng.f64_range(self.lower()..self.upper()) as Float
    }
    /// Get a value in the uniform distribution between `lower` and `upper`.
    #[cfg(feature = "f32")]
    pub fn get_uniform(&self, rng: &mut Rng) -> Float {
        rng.f32_range(self.lower()..self.upper()) as Float
    }
    /// Checks whether the given `value` is compatible with the bounds.
    pub fn contains(&self, value: Float) -> bool {
        match self {
            Self::NoBound => true,
            Self::LowerBound(lb) => value >= *lb,
            Self::UpperBound(ub) => value <= *ub,
            Self::LowerAndUpperBound(lb, ub) => value >= *lb && value <= *ub,
        }
    }
    /// Returns the lower bound or `-inf` if there is none.
    pub const fn lower(&self) -> Float {
        match self {
            Self::NoBound => Float::NEG_INFINITY,
            Self::LowerBound(lb) => *lb,
            Self::UpperBound(_) => Float::NEG_INFINITY,
            Self::LowerAndUpperBound(lb, _) => *lb,
        }
    }
    /// Returns the upper bound or `+inf` if there is none.
    pub const fn upper(&self) -> Float {
        match self {
            Self::NoBound => Float::INFINITY,
            Self::LowerBound(_) => Float::INFINITY,
            Self::UpperBound(ub) => *ub,
            Self::LowerAndUpperBound(_, ub) => *ub,
        }
    }
    /// Checks if the given value is equal to one of the bounds.
    ///
    /// TODO: his just does equality comparison right now, which probably needs to be improved
    /// to something with an epsilon (significant but not critical to most fits right now).
    pub fn at_bound(&self, value: Float) -> bool {
        match self {
            Self::NoBound => false,
            Self::LowerBound(lb) => value == *lb,
            Self::UpperBound(ub) => value == *ub,
            Self::LowerAndUpperBound(lb, ub) => value == *lb || value == *ub,
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
    pub fn to_bounded(values: &[Float], bounds: Option<&Vec<Self>>) -> DVector<Float> {
        bounds
            .map_or_else(
                || values.to_vec(),
                |bounds| {
                    values
                        .iter()
                        .zip(bounds)
                        .map(|(val, bound)| bound._to_bounded(*val))
                        .collect()
                },
            )
            .into()
    }
    fn _to_bounded(&self, val: Float) -> Float {
        match *self {
            Self::LowerBound(lb) => lb - 1.0 + Float::sqrt(Float::powi(val, 2) + 1.0),
            Self::UpperBound(ub) => ub + 1.0 - Float::sqrt(Float::powi(val, 2) + 1.0),
            Self::LowerAndUpperBound(lb, ub) => lb + (Float::sin(val) + 1.0) * (ub - lb) / 2.0,
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
    pub fn to_unbounded(values: &[Float], bounds: Option<&Vec<Self>>) -> DVector<Float> {
        bounds
            .map_or_else(
                || values.to_vec(),
                |bounds| {
                    values
                        .iter()
                        .zip(bounds)
                        .map(|(val, bound)| bound._to_unbounded(*val))
                        .collect()
                },
            )
            .into()
    }
    fn _to_unbounded(&self, val: Float) -> Float {
        match *self {
            Self::LowerBound(lb) => Float::sqrt(Float::powi(val - lb + 1.0, 2) - 1.0),
            Self::UpperBound(ub) => Float::sqrt(Float::powi(ub - val + 1.0, 2) - 1.0),
            Self::LowerAndUpperBound(lb, ub) => Float::asin(2.0 * (val - lb) / (ub - lb) - 1.0),
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
pub trait Function<U, E> {
    /// The evaluation of the function at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    fn evaluate(&self, x: &[Float], user_data: &mut U) -> Result<Float, E>;
    /// The evaluation of the function at a point `x` with the given arguments/user data. This
    /// function assumes `x` is an internal, unbounded vector, but performs a coordinate transform
    /// to bound `x` when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    fn evaluate_bounded(
        &self,
        x: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
    ) -> Result<Float, E> {
        self.evaluate(Bound::to_bounded(x, bounds).as_slice(), user_data)
    }
    /// The evaluation of the gradient at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn gradient(&self, x: &[Float], user_data: &mut U) -> Result<DVector<Float>, E> {
        let n = x.len();
        let x = DVector::from_column_slice(x);
        let mut grad = DVector::zeros(n);
        // This is technically the best step size for the gradient, cbrt(eps) * x_i (or just
        // cbrt(eps) if x_i = 0)
        let h: DVector<Float> = x
            .iter()
            .map(|&xi| Float::cbrt(Float::EPSILON) * (xi.abs() + 1.0))
            .collect::<Vec<_>>()
            .into();
        for i in 0..n {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h[i];
            x_minus[i] -= h[i];
            let f_plus = self.evaluate(x_plus.as_slice(), user_data)?;
            let f_minus = self.evaluate(x_minus.as_slice(), user_data)?;
            grad[i] = (f_plus - f_minus) / (2.0 * h[i]);
        }
        Ok(grad)
    }
    /// The evaluation of the gradient at a point `x` with the given arguments/user data. This
    /// function assumes `x` is an internal, unbounded vector, but performs a coordinate transform
    /// to bound `x` when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn gradient_bounded(
        &self,
        x: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
    ) -> Result<DVector<Float>, E> {
        self.gradient(Bound::to_bounded(x, bounds).as_slice(), user_data)
    }

    /// The evaluation of the hessian at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn hessian(&self, x: &[Float], user_data: &mut U) -> Result<DMatrix<Float>, E> {
        let x = DVector::from_column_slice(x);
        let h: DVector<Float> = x
            .iter()
            .map(|&xi| Float::cbrt(Float::EPSILON) * (xi.abs() + 1.0))
            .collect::<Vec<_>>()
            .into();
        let mut res = DMatrix::zeros(x.len(), x.len());
        let mut g_plus = DMatrix::zeros(x.len(), x.len());
        let mut g_minus = DMatrix::zeros(x.len(), x.len());
        // g+ and g- are such that
        // g+[(i, j)] = g[i](x + h_je_j) and
        // g-[(i, j)] = g[i](x - h_je_j)
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h[i];
            x_minus[i] -= h[i];
            g_plus.set_column(i, &self.gradient(x_plus.as_slice(), user_data)?);
            g_minus.set_column(i, &self.gradient(x_minus.as_slice(), user_data)?);
            for j in 0..=i {
                if i == j {
                    res[(i, j)] = (g_plus[(i, j)] - g_minus[(i, j)]) / (2.0 * h[i]);
                } else {
                    res[(i, j)] = ((g_plus[(i, j)] - g_minus[(i, j)]) / (4.0 * h[j]))
                        + ((g_plus[(j, i)] - g_minus[(j, i)]) / (4.0 * h[i]));
                    res[(j, i)] = res[(i, j)];
                }
            }
        }
        Ok(res)
    }
    /// The evaluation of the hessian at a point `x` with the given arguments/user data. This
    /// function assumes `x` is an internal, unbounded vector, but performs a coordinate transform
    /// to bound `x` when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn hessian_bounded(
        &self,
        x: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
    ) -> Result<DMatrix<Float>, E> {
        self.hessian(Bound::to_bounded(x, bounds).as_slice(), user_data)
    }
}

/// A status message struct containing all information about a minimization result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Status {
    /// A [`String`] message that can be set by minimization [`Algorithm`]s.
    pub message: String,
    /// The current position of the minimization.
    pub x: DVector<Float>,
    /// The initial position of the minimization.
    pub x0: DVector<Float>,
    /// The bounds used for the minimization.
    pub bounds: Option<Vec<Bound>>,
    /// The current value of the minimization problem function at [`Status::x`].
    pub fx: Float,
    /// The number of function evaluations (approximately, this is left up to individual
    /// [`Algorithm`]s to correctly compute and may not be exact).
    pub n_f_evals: usize,
    /// The number of gradient evaluations (approximately, this is left up to individual
    /// [`Algorithm`]s to correctly compute and may not be exact).
    pub n_g_evals: usize,
    /// Flag that says whether or not the fit is in a converged state.
    pub converged: bool,
    /// The Hessian matrix at the end of the fit ([`None`] if not computed yet)
    pub hess: Option<DMatrix<Float>>,
    /// Covariance matrix at the end of the fit ([`None`] if not computed yet)
    pub cov: Option<DMatrix<Float>>,
    /// Errors on parameters at the end of the fit ([`None`] if not computed yet)
    pub err: Option<DVector<Float>>,
    /// Optional parameter names
    pub parameters: Option<Vec<String>>,
}

impl Status {
    /// Updates the [`Status::message`] field.
    pub fn update_message(&mut self, message: &str) {
        self.message = message.to_string();
    }
    /// Updates the [`Status::x`] and [`Status::fx`] fields.
    pub fn update_position(&mut self, pos: (DVector<Float>, Float)) {
        self.x = pos.0;
        self.fx = pos.1;
    }
    /// Sets [`Status::converged`] to be `true`.
    pub fn set_converged(&mut self) {
        self.converged = true;
    }
    /// Increments [`Status::n_f_evals`] by `1`.
    pub fn inc_n_f_evals(&mut self) {
        self.n_f_evals += 1;
    }
    /// Increments [`Status::n_g_evals`] by `1`.
    pub fn inc_n_g_evals(&mut self) {
        self.n_g_evals += 1;
    }
    /// Sets parameter names.
    pub fn set_parameter_names<L: AsRef<str>>(&mut self, names: &[L]) {
        self.parameters = Some(names.iter().map(|name| name.as_ref().to_string()).collect());
    }
    /// Sets the covariance matrix and updates parameter errors.
    pub fn set_cov(&mut self, covariance: Option<DMatrix<Float>>) {
        if let Some(cov_mat) = &covariance {
            self.err = Some(cov_mat.diagonal().map(Float::sqrt));
        }
        self.cov = covariance;
    }
    /// Sets the Hessian matrix, computes the covariance matrix, and updates parameter errors.
    pub fn set_hess(&mut self, hessian: &DMatrix<Float>) {
        self.hess = Some(hessian.clone());
        let mut covariance = hessian.clone().try_inverse();
        if covariance.is_none() {
            covariance = hessian
                .clone()
                .pseudo_inverse(Float::cbrt(Float::EPSILON))
                .ok();
        }
        if let Some(cov_mat) = &covariance {
            self.err = Some(cov_mat.diagonal().map(Float::sqrt));
        }
        self.cov = covariance;
    }
}
impl Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let title = format!(
            "╒══════════════════════════════════════════════════════════════════════════════════════════════╕
│{:^94}│",
            "FIT RESULTS",
        );
        let status = format!(
            "╞════════════════════════════════════════════╤════════════════════╤═════════════╤══════════════╡
│ Status: {}                    │ fval: {:+12.3E} │ #fcn: {:>5} │ #grad: {:>5} │",
            if self.converged {
                "Converged      "
            } else {
                "Invalid Minimum"
            },
            self.fx,
            self.n_f_evals,
            self.n_f_evals,
        );
        let message = format!(
            "├────────────────────────────────────────────┴────────────────────┴─────────────┴──────────────┤
│ Message: {:<83} │",
            self.message,
        );
        let header = "├───────╥──────────────┬──────────────╥──────────────┬──────────────┬──────────────┬───────────┤
│ Par # ║        Value │  Uncertainty ║      Initial │       -Bound │       +Bound │ At Limit? │
├───────╫──────────────┼──────────────╫──────────────┼──────────────┼──────────────┼───────────┤"
            .to_string();
        let mut res_list: Vec<String> = vec![];
        let errs = self
            .err
            .clone()
            .unwrap_or_else(|| DVector::from_element(self.x.len(), Float::NAN));
        let bounds = self
            .bounds
            .clone()
            .unwrap_or_else(|| vec![Bound::NoBound; self.x.len()]);
        for i in 0..self.x.len() {
            let row =
                format!(
                "│ {:>5} ║ {:>+12.3E} │ {:>+12.3E} ║ {:>+12.3E} │ {:>+12.3E} │ {:>+12.3E} │ {:^9} │",
                i,
                self.x[i],
                errs[i],
                self.x0[i],
                bounds[i].lower(),
                bounds[i].upper(),
                if bounds[i].at_bound(self.x[i]) { "yes" } else { "" }
            );
            res_list.push(row);
        }
        let bottom = "└───────╨──────────────┴──────────────╨──────────────┴──────────────┴──────────────┴───────────┘".to_string();
        let out = [title, status, message, header, res_list.join("\n"), bottom].join("\n");
        write!(f, "{}", out)
    }
}

/// A trait representing a minimization algorithm.
///
/// This trait is implemented for the algorithms found in the [`algorithms`] module, and contains
/// all the methods needed to be run by a [`Minimizer`].
pub trait Algorithm<U, E> {
    /// Any setup work done before the main steps of the algorithm should be done here.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
        x0: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
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
        user_data: &mut U,
        status: &mut Status,
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
        func: &dyn Function<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<bool, E>;
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
        func: &dyn Function<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        Ok(())
    }
}

/// The main struct used for running [`Algorithm`]s on [`Function`]s.
pub struct Minimizer<U, E> {
    /// The [`Status`] of the [`Minimizer`], usually read after minimization.
    pub status: Status,
    algorithm: Box<dyn Algorithm<U, E>>,
    max_steps: usize,
    observers: Vec<Arc<RwLock<dyn Observer<U>>>>,
    dimension: usize,
    bounds: Option<Vec<Bound>>,
}
impl<U, E> Display for Minimizer<U, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.status)
    }
}
impl<U, E> Minimizer<U, E> {
    const DEFAULT_MAX_STEPS: usize = 4000;
    /// Creates a new [`Minimizer`] with the given (boxed) [`Algorithm`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new(algorithm: Box<dyn Algorithm<U, E>>, dimension: usize) -> Self {
        Self {
            status: Status::default(),
            algorithm,
            max_steps: Self::DEFAULT_MAX_STEPS,
            observers: Vec::default(),
            dimension,
            bounds: None,
        }
    }
    fn reset_status(&mut self) {
        let new_status = Status {
            bounds: self.status.bounds.clone(),
            ..Default::default()
        };
        self.status = new_status;
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
    pub fn with_bounds<I: IntoIterator<Item = B>, B: Into<Bound>>(mut self, bounds: I) -> Self {
        let bounds = bounds.into_iter().map(Into::into).collect::<Vec<Bound>>();
        assert!(bounds.len() == self.dimension);
        self.bounds = Some(bounds);
        self
    }

    /// Set the maximum number of steps to perform before failure (default: 4000).
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
    /// Adds a single [`Observer`] to the [`Minimizer`].
    pub fn with_observer(mut self, observer: Arc<RwLock<dyn Observer<U>>>) -> Self {
        self.observers.push(observer);
        self
    }
    /// Minimize the given [`Function`] starting at the point `x0`.
    ///
    /// This method first runs [`Algorithm::initialize`], then runs [`Algorithm::step`] in a loop,
    /// terminating if [`Algorithm::check_for_termination`] returns `true` or if
    /// the maximum number of allowed steps is exceeded. Each step will be followed by a sequential
    /// call to all given [`Observer`]s' callback functions. Finally, regardless of convergence,
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
        func: &dyn Function<U, E>,
        x0: &[Float],
        user_data: &mut U,
    ) -> Result<(), E> {
        assert!(x0.len() == self.dimension);
        init_ctrl_c_handler();
        reset_ctrl_c_handler();
        self.reset_status();
        if let Some(bounds) = &self.bounds {
            for (i, (x_i, bound_i)) in x0.iter().zip(bounds).enumerate() {
                assert!(
                    bound_i.contains(*x_i),
                    "Parameter #{} = {} is outside of the given bound: {}",
                    i,
                    x_i,
                    bound_i
                )
            }
        }
        self.status.x0 = DVector::from_column_slice(x0);
        self.algorithm
            .initialize(func, x0, self.bounds.as_ref(), user_data, &mut self.status)?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step <= self.max_steps
            && !observer_termination
            && !self
                .algorithm
                .check_for_termination(func, user_data, &mut self.status)?
            && !is_ctrl_c_pressed()
        {
            self.algorithm
                .step(current_step, func, user_data, &mut self.status)?;
            current_step += 1;
            if !self.observers.is_empty() {
                for observer in self.observers.iter_mut() {
                    observer_termination =
                        observer
                            .write()
                            .callback(current_step, &mut self.status, user_data)
                            || observer_termination;
                }
            }
        }
        self.algorithm
            .postprocessing(func, user_data, &mut self.status)?;
        if current_step > self.max_steps && !self.status.converged {
            self.status.update_message("MAX EVALS");
        }
        if is_ctrl_c_pressed() {
            self.status.update_message("Ctrl-C Pressed");
        }
        Ok(())
    }
}

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
pub trait MCMCAlgorithm<U, E> {
    /// Any setup work done before the main steps of the algorithm should be done here.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
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
        user_data: &mut U,
        ensemble: &mut Ensemble,
    ) -> Result<(), E> {
        Ok(())
    }
}

/// The main struct used for running [`MCMCAlgorithm`]s on [`Function`]s.
pub struct Sampler<U, E> {
    /// The chains of walker positions created during sampling
    pub ensemble: Ensemble,
    mcmc_algorithm: Box<dyn MCMCAlgorithm<U, E>>,
    observers: Vec<Arc<RwLock<dyn MCMCObserver<U>>>>,
}

impl<U, E> Sampler<U, E> {
    /// Creates a new [`Sampler`] with the given (boxed) [`MCMCAlgorithm`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new(mcmc_algorithm: Box<dyn MCMCAlgorithm<U, E>>, x0: Vec<DVector<Float>>) -> Self {
        Self {
            ensemble: Ensemble::new(x0),
            mcmc_algorithm,
            observers: Vec::default(),
        }
    }
    /// Reset the ensemble (except for its starting position)
    pub fn reset(&mut self) {
        self.ensemble.reset();
    }
    /// Adds a single [`MCMCObserver`] to the [`Sampler`].
    pub fn with_observer(mut self, observer: Arc<RwLock<dyn MCMCObserver<U>>>) -> Self {
        self.observers.push(observer);
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
        self.mcmc_algorithm
            .initialize(func, user_data, &mut self.ensemble)?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step < n_steps - 1 // the first step is the initial position
            && !observer_termination
            && !self.mcmc_algorithm.check_for_termination(
                func,
                user_data,
                &mut self.ensemble,
            )?
            && !is_ctrl_c_pressed()
        {
            let walker_step = self.ensemble.dimension().1;
            self.mcmc_algorithm
                .step(walker_step + 1, func, user_data, &mut self.ensemble)?;
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
        self.mcmc_algorithm
            .postprocessing(func, user_data, &mut self.ensemble)?;
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

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use crate::{algorithms::LBFGSB, Minimizer};

    #[test]
    #[allow(unused_variables)]
    fn test_minimizer_constructor() {
        #[allow(clippy::box_default)]
        let algo: LBFGSB<(), Infallible> = LBFGSB::default();
        let minimizer = Minimizer::new(Box::new(algo), 5);
    }
}
