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
//! - [Bounds](#bounds)
//! - [Future Plans](#future-plans)
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
//! fn main() -> Result<(), Infallible> {
//!     let problem = Rosenbrock { n: 2 };
//!     let nm = NelderMead::default();
//!     let mut m = Minimizer::new(&nm, 2);
//!     let x0 = &[2.0, 2.0];
//!     m.minimize(&problem, x0, &mut ())?;
//!     println!("{}", m.status);
//!     Ok(())
//! }
//! ```
//!
//! This should output
//! ```shell
//! MSG:       term_f = STDDEV
//! X:         +1.000 ± 0.707
//!            +1.000 ± 1.416
//! F(X):      +0.000
//! N_F_EVALS: 159
//! N_G_EVALS: 0
//! CONVERGED: true
//! COV:       
//!   ┌             ┐
//!   │ 0.500 1.000 │
//!   │ 1.000 2.005 │
//!   └             ┘
//! ```
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
//! * Eventually, I would like to implement MCMC algorithms and some more modern gradient-free optimization techniques.
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
    fmt::{Debug, Display, UpperExp},
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, Ordering},
        Once,
    },
};

use dyn_clone::DynClone;
use lazy_static::lazy_static;
use nalgebra::{DMatrix, DVector, RealField, Scalar};
use num::{traits::NumAssign, Float};

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
    T: Scalar + Float + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.lower(), self.upper())
    }
}
impl<T> From<(T, T)> for Bound<T>
where
    T: Scalar + Float,
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
    T: Float + Scalar + Debug,
{
    /// Checks whether the given `value` is compatible with the bounds.
    pub fn contains(&self, value: &T) -> bool {
        match self {
            Self::NoBound => true,
            Self::LowerBound(lb) => value >= lb,
            Self::UpperBound(ub) => value <= ub,
            Self::LowerAndUpperBound(lb, ub) => value >= lb && value <= ub,
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
    /// Checks if the given value is equal to one of the bounds.
    ///
    /// TODO: his just does equality comparison right now, which probably needs to be improved
    /// to something with an epsilon (significant but not critical to most fits right now).
    pub fn at_bound(&self, value: T) -> bool {
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
    pub fn to_bounded(values: &[T], bounds: Option<&Vec<Self>>) -> DVector<T> {
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
    pub fn to_unbounded(values: &[T], bounds: Option<&Vec<Self>>) -> DVector<T> {
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
    T: Float + Scalar + NumAssign,
{
    /// The evaluation of the function at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    fn evaluate(&self, x: &[T], user_data: &mut U) -> Result<T, E>;

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
        x: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<T, E> {
        self.evaluate(Bound::to_bounded(x, bounds).as_slice(), user_data)
    }
    /// The evaluation of the gradient at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn gradient(&self, x: &[T], user_data: &mut U) -> Result<DVector<T>, E> {
        let n = x.len();
        let x = DVector::from_column_slice(x);
        let mut grad = DVector::zeros(n);
        // This is technically the best step size for the gradient, cbrt(eps) * x_i (or just
        // cbrt(eps) if x_i = 0)
        let h: DVector<T> = x
            .iter()
            .map(|&xi| T::cbrt(T::epsilon()) * (xi.abs() + T::one()))
            .collect::<Vec<_>>()
            .into();
        for i in 0..n {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h[i];
            x_minus[i] -= h[i];
            let f_plus = self.evaluate(x_plus.as_slice(), user_data)?;
            let f_minus = self.evaluate(x_minus.as_slice(), user_data)?;
            grad[i] = (f_plus - f_minus) / (convert!(2.0, T) * h[i]);
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
        x: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<DVector<T>, E> {
        self.gradient(Bound::to_bounded(x, bounds).as_slice(), user_data)
    }

    /// The evaluation of the hessian at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn hessian(&self, x: &[T], user_data: &mut U) -> Result<DMatrix<T>, E> {
        let x = DVector::from_column_slice(x);
        let h: DVector<T> = x
            .iter()
            .map(|&xi| T::cbrt(T::epsilon()) * (xi.abs() + T::one()))
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
                    res[(i, j)] = (g_plus[(i, j)] - g_minus[(i, j)]) / (convert!(2, T) * h[i]);
                } else {
                    res[(i, j)] = ((g_plus[(i, j)] - g_minus[(i, j)]) / (convert!(4, T) * h[j]))
                        + ((g_plus[(j, i)] - g_minus[(j, i)]) / (convert!(4, T) * h[i]));
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
        x: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<DMatrix<T>, E> {
        self.hessian(Bound::to_bounded(x, bounds).as_slice(), user_data)
    }
}

/// A status message struct containing all information about a minimization result.
#[derive(Debug, Clone, Default)]
pub struct Status<T: Scalar> {
    /// A [`String`] message that can be set by minimization [`Algorithm`]s.
    pub message: String,
    /// The current position of the minimization.
    pub x: DVector<T>,
    /// The initial position of the minimization.
    pub x0: DVector<T>,
    /// The bounds used for the minimization.
    pub bounds: Option<Vec<Bound<T>>>,
    /// The current value of the minimization problem function at [`Status::x`].
    pub fx: T,
    /// The number of function evaluations (approximately, this is left up to individual
    /// [`Algorithm`]s to correctly compute and may not be exact).
    pub n_f_evals: usize,
    /// The number of gradient evaluations (approximately, this is left up to individual
    /// [`Algorithm`]s to correctly compute and may not be exact).
    pub n_g_evals: usize,
    /// Flag that says whether or not the fit is in a converged state.
    pub converged: bool,
    /// The Hessian matrix at the end of the fit ([`None`] if not computed yet)
    pub hess: Option<DMatrix<T>>,
    /// Covariance matrix at the end of the fit ([`None`] if not computed yet)
    pub cov: Option<DMatrix<T>>,
    /// Errors on parameters at the end of the fit ([`None`] if not computed yet)
    pub err: Option<DVector<T>>,
}
impl<T: Scalar> Status<T> {
    /// Updates the [`Status::message`] field.
    pub fn update_message(&mut self, message: &str) {
        self.message = message.to_string();
    }
    /// Updates the [`Status::x`] and [`Status::fx`] fields.
    pub fn update_position(&mut self, pos: (DVector<T>, T)) {
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
}
impl<T: Scalar + Float + RealField> Status<T> {
    /// Sets the covariance matrix and updates parameter errors.
    pub fn set_cov(&mut self, covariance: Option<DMatrix<T>>) {
        if let Some(cov_mat) = &covariance {
            self.err = Some(cov_mat.diagonal().map(|v| Float::sqrt(v)));
        }
        self.cov = covariance;
    }
    /// Sets the Hessian matrix, computes the covariance matrix, and updates parameter errors.
    pub fn set_hess(&mut self, hessian: &DMatrix<T>) {
        self.hess = Some(hessian.clone());
        let mut covariance = hessian.clone().try_inverse();
        if covariance.is_none() {
            covariance = hessian
                .clone()
                .pseudo_inverse(Float::cbrt(T::epsilon()))
                .ok();
        }
        if let Some(cov_mat) = &covariance {
            self.err = Some(cov_mat.diagonal().map(|v| Float::sqrt(v)));
        }
        self.cov = covariance;
    }
}
impl<T> Display for Status<T>
where
    T: Float + Scalar + Display + UpperExp,
{
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
            .unwrap_or_else(|| DVector::from_element(self.x.len(), T::nan()));
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
pub trait Algorithm<T: Scalar, U, E>: DynClone {
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
        status: &mut Status<T>,
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
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
        status: &mut Status<T>,
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
        status: &mut Status<T>,
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
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
        status: &mut Status<T>,
    ) -> Result<(), E> {
        Ok(())
    }
}
dyn_clone::clone_trait_object!(<T, U, E> Algorithm<T, U, E> where T: Scalar);

/// A trait which holds a [`callback`](`Observer::callback`) function that can be used to check an
/// [`Algorithm`]'s [`Status`] during a minimization.
pub trait Observer<T: Scalar, U> {
    /// A function that is called at every step of a minimization [`Algorithm`]. If it returns
    /// `false`, the [`Minimizer::minimize`] method will terminate.
    fn callback(&mut self, step: usize, status: &mut Status<T>, user_data: &mut U) -> bool;
}

/// The main struct used for running [`Algorithm`]s on [`Function`]s.
pub struct Minimizer<T, U, E, A>
where
    A: Algorithm<T, U, E>,
    T: Scalar,
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

impl<T, U, E, A> Display for Minimizer<T, U, E, A>
where
    A: Algorithm<T, U, E>,
    T: Scalar + Display + Float + UpperExp,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.status)
    }
}

impl<T, U, E, A: Algorithm<T, U, E>> Minimizer<T, U, E, A>
where
    T: Float + Scalar + Default + Display,
{
    const DEFAULT_MAX_STEPS: usize = 4000;
    /// Creates a new [`Minimizer`] with the given [`Algorithm`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new(algorithm: &A, dimension: usize) -> Self {
        Self {
            status: Status::default(),
            algorithm: dyn_clone::clone(algorithm),
            bounds: None,
            max_steps: Self::DEFAULT_MAX_STEPS,
            observers: Vec::default(),
            dimension,
            _phantom: PhantomData,
        }
    }
    fn reset_status(&mut self) {
        let new_status = Status {
            bounds: self.status.bounds.clone(),
            ..Default::default()
        };
        self.status = new_status;
    }
    /// Set the [`Algorithm`] used by the [`Minimizer`].
    pub fn with_algorithm(mut self, algorithm: &A) -> Self {
        self.algorithm = dyn_clone::clone(algorithm);
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
        self.status.bounds = self.bounds.clone();
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
        self.status.bounds = self.bounds.clone();
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
        init_ctrl_c_handler();
        reset_ctrl_c_handler();
        self.reset_status();
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
        self.status.x0 = DVector::from_column_slice(x0);
        self.algorithm
            .initialize(func, x0, self.bounds.as_ref(), user_data, &mut self.status)?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step <= self.max_steps
            && !observer_termination
            && !self.algorithm.check_for_termination(
                func,
                self.bounds.as_ref(),
                user_data,
                &mut self.status,
            )?
            && !is_ctrl_c_pressed()
        {
            self.algorithm.step(
                current_step,
                func,
                self.bounds.as_ref(),
                user_data,
                &mut self.status,
            )?;
            current_step += 1;
            if !self.observers.is_empty() {
                for observer in self.observers.iter_mut() {
                    observer_termination =
                        !observer.callback(current_step, &mut self.status, user_data)
                            || observer_termination;
                }
            }
        }
        self.algorithm
            .postprocessing(func, self.bounds.as_ref(), user_data, &mut self.status)?;
        if current_step > self.max_steps && !self.status.converged {
            self.status.update_message("MAX EVALS");
        }
        if is_ctrl_c_pressed() {
            self.status.update_message("Ctrl-C Pressed");
        }
        Ok(())
    }
}
