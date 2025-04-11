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
//! use ganesh::{Function, Float, Minimizer, NopAbortSignal};
//! use ganesh::algorithms::NelderMead;
//! use ganesh::traits::*;
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
//!     m.minimize(&mut problem, x0, &mut (), NopAbortSignal::new().boxed())?;
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
//! ### scikit-learn (used in constructing a Bayesian Mixture Model in the Global ESS step)
//! ```text
//! @article{scikit-learn,
//!   title={Scikit-learn: Machine Learning in {P}ython},
//!   author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
//!           and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
//!           and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
//!           Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
//!   journal={Journal of Machine Learning Research},
//!   volume={12},
//!   pages={2825--2830},
//!   year={2011}
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

use std::fmt::{Debug, Display};

use fastrand::Rng;
use fastrand_contrib::RngExt;
use nalgebra::{Cholesky, DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Module containing minimization algorithms
pub mod algorithms;
pub use algorithms::{Minimizer, Status};

/// Module containing MCMC sampling algorithms
pub mod samplers;
pub use samplers::{Ensemble, Sampler};

/// Module containing swarm algorithms
pub mod swarms;
pub use swarms::{Swarm, SwarmMinimizer};

/// Module containing various kinds of observers
pub mod observers;
/// Module containing standard functions for testing algorithms
pub mod test_functions;

/// Module containing a trait for aborting algorithms
pub mod abort_signal;
pub use abort_signal::NopAbortSignal;

/// Module containing useful traits
pub mod traits {
    pub use crate::abort_signal::AbortSignal;
    pub use crate::algorithms::Algorithm;
    pub use crate::observers::{MCMCObserver, Observer};
    pub use crate::samplers::MCMCAlgorithm;
    pub use crate::swarms::SwarmAlgorithm;
    pub use crate::{Function, SampleFloat};
}

/// A floating-point number type (defaults to [`f64`], see `f32` feature).
#[cfg(not(feature = "f32"))]
pub type Float = f64;

/// A floating-point number type (defaults to [`f64`], see `f32` feature).
#[cfg(feature = "f32")]
pub type Float = f32;

/// The mathematical constant $`\pi`$.
#[cfg(not(feature = "f32"))]
pub const PI: Float = std::f64::consts::PI;

/// The mathematical constant $`\pi`$.
#[cfg(feature = "f32")]
pub const PI: Float = std::f32::consts::PI;

/// A helper trait to get feature-gated floating-point random values
pub trait SampleFloat {
    /// Get a random value in a range
    fn range(&mut self, lower: Float, upper: Float) -> Float;
    /// Get a random value in the range [0, 1]
    fn float(&mut self) -> Float;
    /// Get a random Normal value
    fn normal(&mut self, mu: Float, sigma: Float) -> Float;
    /// Get a random value from a multivariate Normal distribution
    #[allow(clippy::expect_used)]
    fn mv_normal(&mut self, mu: &DVector<Float>, cov: &DMatrix<Float>) -> DVector<Float> {
        let cholesky = Cholesky::new(cov.clone()).expect("Covariance matrix not positive definite");
        let a = cholesky.l();
        let z = DVector::from_iterator(mu.len(), (0..mu.len()).map(|_| self.normal(0.0, 1.0)));
        mu + a * z
    }
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

/// Describes a point in parameter space that can be used in [`Algorithm`](`crate::traits::Algorithm`)s.
#[derive(PartialEq, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Point {
    x: DVector<Float>,
    fx: Float,
}
impl Point {
    /// Getter method for the point's position
    pub const fn get_x(&self) -> &DVector<Float> {
        &self.x
    }
    /// Getter method for the point's evaluation
    pub const fn get_fx(&self) -> Float {
        self.fx
    }
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
    /// Converts the point's `x` from an unbounded space to a bounded one.
    pub fn to_bounded(&self, bounds: Option<&Vec<Bound>>) -> Self {
        Self {
            x: Bound::to_bounded(self.x.as_slice(), bounds),
            fx: self.fx,
        }
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
/// bounded algorithms directly, or by some unbounded algorithms using parameter space
/// transformations.
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
    /// Checks whether the given [`DVector`] is compatible with the list of bounds in each
    /// coordinate.
    pub fn contains_vec(bounds: &[Self], vec: &DVector<Float>) -> bool {
        for (bound, value) in bounds.iter().zip(vec) {
            if !bound.contains(*value) {
                return false;
            }
        }
        true
    }
    /// Checks whether the given `value` is compatible with the bound and returns `0.0` if it is,
    /// and the distance to the bound otherwise signed by whether the bound is a lower (`-`) or
    /// upper (`+`) bound.
    pub fn bound_excess(&self, value: Float) -> Float {
        match self {
            Self::NoBound => 0.0,
            Self::LowerBound(lb) => {
                if value >= *lb {
                    0.0
                } else {
                    value - lb
                }
            }
            Self::UpperBound(ub) => {
                if value <= *ub {
                    0.0
                } else {
                    value - ub
                }
            }
            Self::LowerAndUpperBound(lb, ub) => {
                if value < *lb {
                    value - lb
                } else if value > *ub {
                    value - ub
                } else {
                    0.0
                }
            }
        }
    }
    /// Checks whether each of the given [`DVector`]'s coordinates are compatible with the bounds
    /// and returns a [`DVector`] containing the result of [`Bound::bound_excess`] at each
    /// coordinate.
    pub fn bounds_excess(bounds: &[Self], vec: &DVector<Float>) -> DVector<Float> {
        bounds
            .iter()
            .zip(vec)
            .map(|(b, v)| b.bound_excess(*v))
            .collect::<Vec<Float>>()
            .into()
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

pub(crate) fn generate_random_vector(
    dimension: usize,
    lb: Float,
    ub: Float,
    rng: &mut Rng,
) -> DVector<Float> {
    DVector::from_vec((0..dimension).map(|_| rng.range(lb, ub)).collect())
}
pub(crate) fn generate_random_vector_in_limits(
    limits: &[(Float, Float)],
    rng: &mut Rng,
) -> DVector<Float> {
    DVector::from_vec(
        (0..limits.len())
            .map(|i| rng.range(limits[i].0, limits[i].1))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use crate::algorithms::{Minimizer, LBFGSB};

    #[test]
    #[allow(unused_variables)]
    fn test_minimizer_constructor() {
        #[allow(clippy::box_default)]
        let algo: LBFGSB<(), Infallible> = LBFGSB::default();
        let minimizer = Minimizer::new(Box::new(algo), 5);
    }
}
