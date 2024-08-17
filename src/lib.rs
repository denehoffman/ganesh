//! `ganesh`, (/ɡəˈneɪʃ/) named after the Hindu god of wisdom, provides several common minimization algorithms as well as a straightforward, trait-based interface to create your own extensions. This crate is intended to be as simple as possible. The user needs to implement the [`Function`](`core::Function`) trait on some struct which will take a vector of parameters and return a single-valued [`Result`] ($`f(\mathbb{R}^n) \to \mathbb{R}`$). Users can optionally provide gradient and Hessian functions to speed up some algorithms, but a default finite-difference implementation is provided so that all algorithms will work out of the box.
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
//! ```
//! use ganesh::prelude::*;
//! use std::convert::Infallible;
//! pub struct Rosenbrock {
//!     /// Number of dimensions (must be at least 2)
//!     pub n: usize,
//! }
//! impl Function<f64, (), Infallible> for Rosenbrock {
//!     fn evaluate(&self, x: &DVector<f64>, _args: Option<&()>) -> Result<f64, Infallible> {
//!         Ok((0..(self.n - 1))
//!             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//!             .sum())
//!     }
//! }
//! ```
//! To minimize this function, we could consider using the [Nelder-Mead](`algorithms::NelderMead`) algorithm:
//! ```
//! # use std::convert::Infallible;
//! # pub struct Rosenbrock {
//! #     /// Number of dimensions (must be at least 2)
//! #     pub n: usize,
//! # }
//! # impl Function<f64, (), Infallible> for Rosenbrock {
//! #     fn evaluate(&self, x: &DVector<f64>, _args: Option<&()>) -> Result<f64, Infallible> {
//! #         Ok((0..(self.n - 1))
//! #             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//! #             .sum())
//! #     }
//! # }
//! use ganesh::prelude::*;
//! use ganesh::algorithms::NelderMead;
//!
//! let func = Rosenbrock { n: 2 };
//! let mut m = NelderMead::new(func, &[-2.3, 3.4], None);
//! let status = minimize!(m, 1000).unwrap(); // Run at most 1000 algorithm steps
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
//! I have ignored the `status` variable here, but in practice, the [`Minimizer::minimize`](`core::Minimizer::minimize`) method should return the last message sent by the algorithm. This can indicate the status of a fit without explicitly causing an error. This makes it easier to debug, since it can be tedious to have two separate error types, one for the function and one for the algorithm, returned by the minimization (functions can always be failable in this crate). We could also swap the `f64`s for `f32`s (or any type which implements the [`Field`](`core::Field`) trait) in the Rosenbrock implementation. Additionally, if we wanted to modify any of the hyperparameters in the fitting algorithm, we could use [`NelderMeadOptions::builder()`](`algorithms::nelder_mead::NelderMeadOptions::builder`) and pass it as the third argument in the [`NelderMead::new`][`algorithms::NelderMead::new`] constructor. Finally, all algorithm implementations are constructed to pass a unique message type to a callback function. For [`NelderMead`](`algorithms::NelderMead`), we could do the following:
//! ```ignore
//! let status = minimize!(m, 1000, |message| println!("step: {}\nx: {:?}\nf(x): {}", message.step, message.x, message.fx)).unwrap();
//! ```
//! This will print out the current step, the best position found by the optimizer at that step, and the function's evaluation at that position for each step in the algorithm. You can use the step number to limit printing (print only steps divisible by 100, for example).
//!
//! The [`minimize!`](`crate::minimize!`) macro exists to simplify the [`Minimizer<F, A, E>::minimize<Callback: Fn(&Self)>(&mut self, args: Option<&A>, steps: usize, callback: Callback) -> Result<(), E>`](`core::Minimizer::minimize`) call if you don't actually want a callback or arguments.
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
    clippy::dbg_macro,
    missing_docs
)]
#![allow(clippy::too_many_arguments)]

/// Contains core functionality and traits.
pub mod core;

/// Contains various optimization algorithms.
pub mod algorithms;

/// Provides a set of test functions commonly used in optimization.
pub mod test_functions;

/// A convenient module that re-exports the most commonly used items from this crate.
///
/// This module is designed to be glob-imported (`use crate::prelude::*;`) to quickly
/// bring the core functionality of the crate into scope.
pub mod prelude {
    pub use crate::core::{Field, Function, LineSearch, Minimizer};
    pub use crate::minimize;
    pub use nalgebra::DVector;
}
