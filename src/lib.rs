//! `ganesh` (/ɡəˈneɪʃ/), named after the Hindu god of wisdom, provides several common minimization algorithms as well as a straightforward, trait-based interface to create your own extensions. This crate is intended to be as simple as possible. The user needs to implement the [`CostFunction`](crate::traits::CostFunction) trait on some struct which will take a vector of parameters and return a single-valued [`Result`] ($`f(\mathbb{R}^n) \to \mathbb{R}`$). Users can optionally provide a gradient function to speed up some algorithms, but a default central finite-difference implementation is provided so that all algorithms will work out of the box.
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
//! ## Key Features
//! * Algorithms that are simple to use with sensible defaults.
//! * Traits which make developing future algorithms simple and consistent.
//! * A simple interface that lets new users get started quickly.
//!
//! ## Quick Start
//!
//! This crate provides some common test functions in the [`test_functions`] module. Consider the following implementation of the Rosenbrock function:
//!
//! ```rust
//! use ganesh::traits::*;
//! use ganesh::{core::Engine, Float};
//! use std::convert::Infallible;
//!
//! pub struct Rosenbrock {
//!     pub n: usize,
//! }
//! impl CostFunction<(), Infallible> for Rosenbrock {
//!     fn evaluate(&self, x: &[Float], _user_data: &mut ()) -> Result<Float, Infallible> {
//!         Ok((0..(self.n - 1))
//!             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//!             .sum())
//!     }
//! }
//! ```
//! To minimize this function, we could consider using the Nelder-Mead algorithm:
//! ```rust
//! use ganesh::algorithms::gradient_free::NelderMead;
//! use ganesh::traits::*;
//! use ganesh::{core::Engine, Float};
//! use std::convert::Infallible;
//!
//! # pub struct Rosenbrock {
//! #     pub n: usize,
//! # }
//! # impl CostFunction<(), Infallible> for Rosenbrock {
//! #     fn evaluate(&self, x: &[Float], _user_data: &mut ()) -> Result<Float, Infallible> {
//! #         Ok((0..(self.n - 1))
//! #             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//! #             .sum())
//! #     }
//! # }
//! fn main() -> Result<(), Infallible> {
//!     let problem = Rosenbrock { n: 2 };
//!     let nm = NelderMead::default();
//!     let mut m = Engine::new(nm);
//!     m.on_status(|s| s.with_x0([2.0, 2.0]));
//!     m.process(&problem)?;
//!     println!("{}", m.result);
//!     Ok(())
//! }
//! ```
//!
//! This should output
//! ```shell
//! ╭──────────────────────────────────────────────────────────────────╮
//! │                                                                  │
//! │                           FIT RESULTS                            │
//! │                                                                  │
//! ├───────────┬───────────────────┬────────────────┬─────────────────┤
//! │ Status    │ f(x)              │ #f(x)          │ #∇f(x)          │
//! ├───────────┼───────────────────┼────────────────┼─────────────────┤
//! │ Converged │ 0.00023           │ 76             │ 0               │
//! ├───────────┼───────────────────┴────────────────┴─────────────────┤
//! │           │                                                      │
//! │ Message   │ term_f = STDDEV                                      │
//! │           │                                                      │
//! ├───────────┴─────────────────────────────┬────────────┬───────────┤
//! │ Parameter                               │ Bound      │ At Limit? │
//! ├───────────┬─────────┬─────────┬─────────┼──────┬─────┼───────────┤
//! │           │ =       │ σ       │ 0       │ -    │ +   │           │
//! ├───────────┼─────────┼─────────┼─────────┼──────┼─────┼───────────┤
//! │ x_0       │ 1.00081 │ 0.84615 │ 2.00000 │ -inf │ inf │ No        │
//! │ x_1       │ 1.00313 │ 1.69515 │ 2.00000 │ -inf │ inf │ No        │
//! ╰───────────┴─────────┴─────────┴─────────┴──────┴─────┴───────────╯
//! ```
//! ## MCMC
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
//! ## Bounds
//! All minimizers in `ganesh` have access to a feature which allows algorithms which usually function in unbounded parameter spaces to only return results inside a bounding box. This is done via a parameter transformation, the same one used by [`LMFIT`](https://lmfit.github.io/lmfit-py/) and [`MINUIT`](https://root.cern.ch/doc/master/classTMinuit.html). This transform is not enacted on algorithms which already have bounded implementations, like [`L-BFGS-B`](`algorithms::gradient::lbfgsb`). While the user inputs parameters within the bounds, unbounded algorithms can (and in practice will) convert those values to a set of unbounded "internal" parameters. When functions are called, however, these internal parameters are converted back into bounded "external" parameters, via the following transformations:
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
//! ## Future Plans
//!
//! * Eventually, I would like to implement some more modern gradient-free optimization techniques.
//! * There are probably many optimizations and algorithm extensions that I'm missing right now because I just wanted to get it working first.
//! * There should be more tests and documentation (as usual).
//!
//! ## Citations
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

/// Module containing minimization algorithms
pub mod core;

/// Module containing all traits
pub mod traits;

/// Module containing various minimization algorithms
pub mod algorithms;

/// Module containing standard functions for testing algorithms
pub mod test_functions;

/// Module containing various utilities
pub mod utils;

use std::sync::atomic::{AtomicBool, Ordering};

/// Re-export the `nalgebra` crate
pub use nalgebra;
use parking_lot::Once;

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

static WARNINGS_ENABLED: AtomicBool = AtomicBool::new(true);
static WARNINGS_SET_BY_ENV: AtomicBool = AtomicBool::new(false);
static WARNINGS_OVERRIDE: AtomicBool = AtomicBool::new(false);
static INIT: Once = Once::new();

fn init_env_override() {
    INIT.call_once(|| {
        if let Ok(val) = std::env::var("GANESH_WARNINGS") {
            if val == "0" {
                WARNINGS_SET_BY_ENV.store(true, Ordering::Relaxed);
                WARNINGS_ENABLED.store(false, Ordering::Relaxed);
            }
            if val == "1" {
                WARNINGS_SET_BY_ENV.store(true, Ordering::Relaxed);
                WARNINGS_ENABLED.store(true, Ordering::Relaxed);
            }
        }
    });
}

fn try_set_warnings_override(value: bool) {
    init_env_override();
    if WARNINGS_SET_BY_ENV.load(Ordering::Relaxed) {
        return;
    }
    let already_set = WARNINGS_OVERRIDE.swap(true, Ordering::Relaxed);
    if !already_set {
        WARNINGS_ENABLED.store(value, Ordering::Relaxed);
    }
}

/// A method which can force-enable warnings which may be disabled by dependencies.
///
/// This method will still not enable warnings if the environment variable `GANESH_WARNINGS=0`.
pub fn enable_warnings() {
    try_set_warnings_override(true);
}

/// A method which can force-disable warnings which may be enabled by dependencies.
///
/// This method will still not disable warnings if the environment variable `GANESH_WARNINGS=1`.
pub fn disable_warnings() {
    try_set_warnings_override(false);
}

/// Returns `true` if warnings are enabled.
///
/// Warnings are enabled by default and can be disabled either by setting the environment variable
/// `GANESH_WARNINGS=0` or by calling [`disable_warnings`] first. The first call of
/// [`enable_warnings`] will ensure warnings are enabled, overriding any subsequent calls to
/// [`disable_warnings`]. Setting `GANESH_WARNINGS=1` will force-enable warnings regardless of any
/// calls to [`disable_warnings`]. In all cases, the environment variable takes precedence.
pub fn should_warn() -> bool {
    init_env_override();
    WARNINGS_ENABLED.load(Ordering::Relaxed)
}

/// Conditionally warns the user (warns by default).
///
/// See [`should_warn`] for details on how to conditionally enable and disable warnings.
pub fn maybe_warn(msg: &str) {
    if should_warn() {
        eprintln!("Warning: {msg}");
    }
}
