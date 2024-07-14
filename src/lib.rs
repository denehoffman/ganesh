//! `ganesh`, (/ɡəˈneɪʃ/) named after the Hindu god of wisdom, provides several common minimization algorithms as well as a straightforward, trait-based interface to create your own extensions. This crate is intended to be as simple as possible. The user needs to implement the [`Function`] trait on some struct which will take a vector of parameters and return a single-valued [`Result`] ($`f(\mathbb{R}^n) \to \mathbb{R}`$). Users can optionally provide gradient and Hessian functions to speed up some algorithms, but a default finite-difference implementation is provided so that all algorithms will work out of the box.
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
//! This crate provides some common test functions in the [`test_functions`] module. Consider the following implementation of the [`Rosenbrock`](`test_functions::Rosenbrock`) function:
//!
//! ```ignore
//! use ganesh::prelude::*;
//! use std::convert::Infallible;
//! pub struct Rosenbrock {
//!     /// Number of dimensions (must be at least 2)
//!     pub n: usize,
//! }
//! impl Function<f64, Infallible> for Rosenbrock {
//!     fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
//!         Ok((0..(self.n - 1))
//!             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//!             .sum())
//!     }
//! }
//! ```
//! To minimize this function, we could consider using the [Nelder-Mead](`algorithms::NelderMead`) algorithm:
//! ```ignore
//! use ganesh::prelude::*;
//! use ganesh::algorithms::NelderMead;
//!
//! let func = Rosenbrock { n: 2 };
//! let mut m = NelderMead::new(func, &[-2.3, 3.4], None);
//! let status = minimize!(m).unwrap();
//! let (x_best, fx_best) = m.best();
//! println!("x: {:?}\nf(x): {}", x_best, fx_best);
//! ```
//!
//! This should output
//! ```shell
//! x: [0.9999459113507765, 0.9998977381285472]
//! f(x): 0.000000006421349269800761
//! ```
//!
//! I have ignored the `status` variable here, but in practice, the [`Minimizer::minimize`] method should return the last message sent by the algorithm. This can indicate the status of a fit without explicitly causing an error. This makes it easier to debug, since it can be tedious to have two separate error types, one for the function and one for the algorithm, returned by the minimization (functions can always be failable in this crate). We could also swap the `f64`s for `f32`s (or any type which implements the [`Field`] trait) in the Rosenbrock implementation. Additionally, if we wanted to modify any of the hyperparameters in the fitting algorithm, we could use [`NelderMeadOptions::builder()`](`algorithms::nelder_mead::NelderMeadOptions::builder`) and pass it as the third argument in the [`NelderMead::new`][`algorithms::NelderMead::new`] constructor. Finally, all algorithm implementations are constructed to pass a unique message type to a callback function. For [`NelderMead`](`algorithms::NelderMead`), we could do the following:
//! ```ignore
//! let status = minimize!(m, |message| println!("step: {}\nx: {:?}\nf(x): {}", message.step, message.x, message.fx)).unwrap();
//! ```
//! This will print out the current step, the best position found by the optimizer at that step, and the function's evaluation at that position for each step in the algorithm. You can use the step number to limit printing (print only steps divisible by 100, for example).
//!
//! The `minimize!` macro exists to simplify the [`Minimizer::minimize<Callback: Fn(M)>(&mut self, callback: Callback)`](`Minimizer::minimize`) call, which looks [a bit ugly](https://enet4.github.io/rust-tropes/#toilet-closure) if you don't actually want a callback.
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
use nalgebra::{ComplexField, DMatrix, DVector};
use num_traits::Float;

/// A trait that extends [`Float`] and [`ComplexField`] with additional conversion methods.
///
/// This trait is implemented for types that satisfy the following bounds:
/// - `Float`: Provides basic floating-point operations.
/// - `ComplexField`: Allows for complex number operations.
/// - `std::iter::Sum`: Enables summing of collections of this type.
pub trait Field: Float + ComplexField + std::iter::Sum {
    /// Converts an f64 value to Self.
    ///
    /// # Arguments
    ///
    /// * `x` - The f64 value to convert.
    ///
    /// # Returns
    ///
    /// The converted value of type Self.
    fn convert(x: f64) -> Self;

    /// Converts a usize value to Self.
    ///
    /// # Arguments
    ///
    /// * `x` - The usize value to convert.
    ///
    /// # Returns
    ///
    /// The converted value of type Self.
    fn convert_usize(x: usize) -> Self;
}
impl Field for f32 {
    fn convert(x: f64) -> Self {
        x as Self
    }

    fn convert_usize(x: usize) -> Self {
        x as Self
    }
}
impl Field for f64 {
    fn convert(x: f64) -> Self {
        x as Self
    }

    fn convert_usize(x: usize) -> Self {
        x as Self
    }
}

/// Represents a multivariate function that can be evaluated and differentiated.
///
/// This trait is generic over the field type `F` and an error type `E`.
///
/// # Type Parameters
///
/// * `F`: A type that implements [`Field`] and has a `'static` lifetime.
/// * `E`: The error type returned by the function's methods.
pub trait Function<F, A, E>
where
    F: Field + 'static,
{
    /// Evaluates the function at the given point.
    ///
    /// # Arguments
    ///
    /// * `x`: A slice of `F` representing the point at which to evaluate the function.
    ///
    /// # Returns
    ///
    /// The function value at `x` of type `F`.
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if the evaluation fails.
    fn evaluate(&self, x: &[F], args: &Option<A>) -> Result<F, E>;

    /// Computes the gradient of the function at the given point using central finite
    /// differences.
    ///
    /// Overwrite this method if the true gradient function is known.
    ///
    /// # Arguments
    ///
    /// * `x`: A slice of `F` representing the point at which to compute the gradient.
    ///
    /// # Returns
    ///
    /// A [`DVector`] of `F` representing the gradient.
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if [`Function::evaluate`] fails.
    fn gradient(&self, x: &[F], args: &Option<A>) -> Result<DVector<F>, E> {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        let h = ComplexField::cbrt(F::epsilon());

        for i in 0..n {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            x_plus[i] += h;
            x_minus[i] -= h;
            let f_plus = self.evaluate(&x_plus, args)?;
            let f_minus = self.evaluate(&x_minus, args)?;
            grad[i] = (f_plus - f_minus) / (F::convert(2.0) * h);
        }

        Ok(grad)
    }

    /// Computes both the gradient and the Hessian matrix of the function at the given point.
    ///
    /// This method uses central finite differences to approximate both the gradient and the Hessian.
    ///
    /// # Arguments
    ///
    /// * `x`: A slice of `F` representing the point at which to compute the gradient and Hessian.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - The gradient at `x` as a [`DVector`] of `F`
    /// - The Hessian at `x` as a [`DMatrix`] of `F`
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if [`Function::evaluate`] fails.
    fn gradient_and_hessian(
        &self,
        x: &[F],
        args: &Option<A>,
    ) -> Result<(DVector<F>, DMatrix<F>), E> {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        let mut hess = DMatrix::zeros(n, n);
        let h = ComplexField::cbrt(F::epsilon());
        let two = F::convert(2.0);
        let four = two * two;

        // Compute Hessian
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal element using central difference
                    let mut x_plus = x.to_vec();
                    let mut x_minus = x.to_vec();
                    x_plus[i] += h;
                    x_minus[i] -= h;

                    let f_plus = self.evaluate(&x_plus, args)?;
                    let f_minus = self.evaluate(&x_minus, args)?;
                    let f_center = self.evaluate(x, args)?;

                    grad[i] = (f_plus - f_minus) / (two * h);
                    hess[(i, i)] = (f_plus - two * f_center + f_minus) / (h * h);
                } else {
                    // Off-diagonal element
                    let mut x_plus_plus = x.to_vec();
                    let mut x_plus_minus = x.to_vec();
                    let mut x_minus_plus = x.to_vec();
                    let mut x_minus_minus = x.to_vec();

                    x_plus_plus[i] += h;
                    x_plus_plus[j] += h;
                    x_plus_minus[i] += h;
                    x_plus_minus[j] -= h;
                    x_minus_plus[i] -= h;
                    x_minus_plus[j] += h;
                    x_minus_minus[i] -= h;
                    x_minus_minus[j] -= h;

                    let f_plus_plus = self.evaluate(&x_plus_plus, args)?;
                    let f_plus_minus = self.evaluate(&x_plus_minus, args)?;
                    let f_minus_plus = self.evaluate(&x_minus_plus, args)?;
                    let f_minus_minus = self.evaluate(&x_minus_minus, args)?;

                    hess[(i, j)] = (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus)
                        / (four * h * h);
                    hess[(j, i)] = hess[(i, j)];
                }
            }
        }

        Ok((grad, hess))
    }
}

/// Represents an optimization algorithm for minimizing a function.
///
/// This trait is generic over the field type `F`, a message type `M`, and an error type `E`.
///
/// # Type Parameters
///
/// * `F`: A type that implements [`Field`].
/// * `M`: A message type used to pass information from the algorithm to the callback function.
/// * `E`: The error type returned by the minimizer's methods.
pub trait Minimizer<F, A, M, E>
where
    F: Field,
{
    /// Performs a single step of the minimization algorithm.
    ///
    /// # Returns
    ///
    /// A message of type `M` containing information about the current state of the optimization.
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if the step fails.
    fn step(&mut self, i: usize, args: &Option<A>) -> Result<M, E>;

    /// Checks if the termination condition for the algorithm has been met.
    ///
    /// # Returns
    ///
    /// `true` if the algorithm should terminate, `false` otherwise.
    fn terminate(&self) -> bool;

    /// Runs the minimization algorithm to completion, calling the provided callback after each step.
    ///
    /// # Type Parameters
    ///
    /// * `Callback`: A function type that takes a message of type `M` as an argument.
    ///
    /// # Arguments
    ///
    /// * `callback`: A function that is called after each step with a message of type `M`. Use
    ///   `minimize(|_| {})` as a pass-through function if you don't need a callback.
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if the minimization process fails.
    fn minimize<Callback: Fn(&M)>(&mut self, args: &Option<A>, callback: Callback) -> Result<M, E>;

    /// Returns the best solution found so far by the minimization algorithm.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A vector of `F` representing the best point found.
    /// - The function value of type `F` at the best point.
    fn best(&self) -> (Vec<F>, F);
}

/// A macro to clean up minimization statements
#[macro_export]
macro_rules! minimize {
    ($minimizer:expr) => {
        $minimizer.minimize(None, |_| {})
    };
    ($minimizer:expr, $callback:expr) => {
        $minimizer.minimize(None, $callback)
    };
}

/// Contains various optimization algorithms.
pub mod algorithms;

/// Provides a set of test functions commonly used in optimization.
pub mod test_functions;

/// A convenient module that re-exports the most commonly used items from this crate.
///
/// This module is designed to be glob-imported (`use crate::prelude::*;`) to quickly
/// bring the core functionality of the crate into scope.
pub mod prelude {
    pub use crate::minimize;
    pub use crate::Field;
    pub use crate::{Function, Minimizer};
}
