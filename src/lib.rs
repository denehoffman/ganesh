//! A crate for function minimization using simple, generic traits.
//!
//! `ganesh`, named after the Hindu god of wisdom, provides several common minimization algorithms
//! as well as a straightforward, trait-based interface to create your own extensions. This crate is
//! intended to be as simple as possible. The user needs to implement the [`Function`] trait on some
//! struct which will take a vector of parameters and return a single-valued result
//! ($`f(\Reals^n) \to \Reals`$). Users can optionally provide gradient and Hessian functions to
//! speed up some algorithms, but a default finite-difference implementation is provided so that all
//! algorithms will work out of the box.
//!
//! # Quick Start
//!
//! This crate provides some common test functions in the [`test_functions`] module. Consider the
//! following implementation of the Rosenbrock function:
//!
//! ```no_run
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
//! To minimize this function, we could consider using the Nelder-Mead algorithm:
//! ```no_run
//! use ganesh::prelude::*;
//! use ganesh::algorithms::nelder_mead::NelderMead;
//!
//! let func = Rosenbrock { n: 2 };
//! let mut m = NelderMead::new(func, &[-2.3, 3.4], None);
//! m.minimize(|_| {}).unwrap();
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
//! We could also swap the `f64`s for `f32`s (or any type which implements the [`Field`] trait) in
//! the Rosenbrock implementation. Additionally, if we wanted to modify any of the hyperparameters
//! in the fitting algorithm, we could use
//! [`NelderMeadOptions::builder()`](`algorithms::NelderMeadOptions::builder`) and
//! pass it as the third argument in the
//! [`NelderMead::new`](`algorithms::NelderMead::new`) constructor. Finally, all
//! algorithm implementations are constructed to pass a unique message type to a callback function.
//! For [`NelderMead`](`algorithms::NelderMead`), we could do the following:
//! ```no_run
//! m.minimize(|m| println!("step: {}\nx: {:?}\nf(x): {}", m.step, m.x, m.fx)).unwrap();
//! ```
//! This will print out the current step, the best position found by the optimizer at that step,
//! and the function's evaluation at that position for each step in the algorithm. You can use the
//! step number to limit printing (print only steps divisible by 100, for example).
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
pub trait Function<F, E>
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
    fn evaluate(&self, x: &[F]) -> Result<F, E>;

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
    fn gradient(&self, x: &[F]) -> Result<DVector<F>, E> {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        let h = ComplexField::cbrt(F::epsilon());

        for i in 0..n {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            x_plus[i] += h;
            x_minus[i] -= h;
            let f_plus = self.evaluate(&x_plus)?;
            let f_minus = self.evaluate(&x_minus)?;
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
    fn gradient_and_hessian(&self, x: &[F]) -> Result<(DVector<F>, DMatrix<F>), E> {
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

                    let f_plus = self.evaluate(&x_plus)?;
                    let f_minus = self.evaluate(&x_minus)?;
                    let f_center = self.evaluate(x)?;

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

                    let f_plus_plus = self.evaluate(&x_plus_plus)?;
                    let f_plus_minus = self.evaluate(&x_plus_minus)?;
                    let f_minus_plus = self.evaluate(&x_minus_plus)?;
                    let f_minus_minus = self.evaluate(&x_minus_minus)?;

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
pub trait Minimizer<F, M, E>
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
    fn step(&mut self, i: usize) -> Result<M, E>;

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
    fn minimize<Callback: Fn(&M)>(&mut self, callback: Callback) -> Result<M, E>;

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
        $minimizer.minimize(|_| {})
    };
    ($minimizer:expr, $callback:expr) => {
        $minimizer.minimize($callback)
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
