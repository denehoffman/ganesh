//! `ganesh` (/ɡəˈneɪʃ/), named after the Hindu god of wisdom, provides several common minimization algorithms as well as a straightforward, trait-based interface to create your own extensions. This crate is intended to be as simple as possible. For most minimization problems user needs to implement the [`CostFunction`](crate::traits::cost_function::CostFunction) trait on some struct which will take a vector of parameters and return a single-valued `Result` ($`f(\mathbb{R}^n) \to \mathbb{R}`$). Some algorithms require a gradient which can be implemented via the [`Gradient`](crate::traits::cost_function::Gradient) trait. While users may provide an analytic gradient function to speed up some algorithms, this trait comes with a default central finite-difference implementation so that all algorithms will work out of the box as long as the cost function is well-defined.
//!
//! # Table of Contents
//! - [Key Features](#key-features)
//! - [Quick Start](#quick-start)
//! - [Algorithms](#algorithms)
//! - [Examples](#examples)
//! - [Bounds](#bounds)
//! - [Future Plans](#future-plans)
//! - [Citations](#citations)
//!
//! ## Key Features
//! * Algorithms that are simple to use with sensible defaults.
//! * Traits which make developing future algorithms simple and consistent.
//! * A simple interface that lets new users get started quickly.
//! * The first (and possibly only) pure Rust implementation of the [`L-BFGS-B`](crate::algorithms::gradient::lbfgsb::LBFGSB) algorithm.
//!
//! ## Quick Start
//!
//! This crate provides some common test functions in the [`test_functions`](crate::test_functions) module. Consider the following implementation of the Rosenbrock function:
//!
//! ```rust
//! use ganesh::traits::*;
//! use ganesh::{Float, DVector};
//! use std::convert::Infallible;
//!
//! pub struct Rosenbrock {
//!     pub n: usize,
//! }
//! impl CostFunction for Rosenbrock {
//!     fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
//!         Ok((0..(self.n - 1))
//!             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//!             .sum())
//!     }
//! }
//! ```
//! To minimize this function, we could consider using the Nelder-Mead algorithm:
//! ```rust
//! use ganesh::algorithms::gradient_free::{NelderMead, NelderMeadConfig};
//! use ganesh::traits::*;
//! use ganesh::{Float, DVector, minimize_gradient_free};
//! use std::convert::Infallible;
//!
//! # pub struct Rosenbrock {
//! #     pub n: usize,
//! # }
//! # impl CostFunction for Rosenbrock {
//! #     fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
//! #         Ok((0..(self.n - 1))
//! #             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//! #             .sum())
//! #     }
//! # }
//! fn main() -> Result<(), Infallible> {
//!     let problem = Rosenbrock { n: 2 };
//!     let result = minimize_gradient_free(&problem, [2.0, 2.0], &(), None::<Vec<Bound>>)?;
//!     println!("{}", result);
//!     Ok(())
//! }
//! ```
//!
//! We could also use some more verbose syntax if we wanted additional customization:
//! ```rust
//! use ganesh::algorithms::gradient_free::{NelderMead, NelderMeadConfig};
//! use ganesh::traits::*;
//! use ganesh::{Float, DVector};
//! use std::convert::Infallible;
//!
//! # pub struct Rosenbrock {
//! #     pub n: usize,
//! # }
//! # impl CostFunction for Rosenbrock {
//! #     fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
//! #         Ok((0..(self.n - 1))
//! #             .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
//! #             .sum())
//! #     }
//! # }
//! fn main() -> Result<(), Infallible> {
//!     let problem = Rosenbrock { n: 2 };
//!     let mut nm = NelderMead::default();
//!     let result = nm.process(&problem,
//!                             &(),
//!                             NelderMeadConfig::new([2.0, 2.0]),
//!                             NelderMead::default_callbacks())?;
//!     println!("{}", result);
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
//!
//! The `ganesh` crate provides convenience functions for some of the most common minimization/MCMC algorithms, including [`minimize`] (L-BFGS-B), [`minimize_gradient_free`] (Nelder-Mead), and [`sample`] (AIES).
//!
//! ## Algorithms
//!
//! At the moment, `ganesh` contains the following [`Algorithm`](crate::traits::algorithm::Algorithm)s:
//! - Gradient descent/quasi-Newton:
//!   - [`L-BFGS-B`](crate::algorithms::gradient::lbfgsb::LBFGSB)
//!   - [`Adam`](crate::algorithms::gradient::adam::Adam) (for stochastic [`CostFunction`](crate::traits::cost_function::CostFunction)s)
//! - Gradient-free:
//!   - [`Nelder-Mead`](crate::algorithms::gradient_free::nelder_mead::NelderMead)
//!   - [`Simulated Annealing`](crate::algorithms::gradient_free::simulated_annealing::SimulatedAnnealing)
//! - Markov Chain Monte Carlo (MCMC):
//!   - [`AIES`](crate::algorithms::mcmc::aies::AIES)
//!   - [`ESS`](crate::algorithms::mcmc::ess::ESS)
//! - Swarms:
//!   - [`PSO`](crate::algorithms::particles::pso::PSO) (a basic form of particle swarm optimization)
//!
//! All algorithms are written in pure Rust, including [`L-BFGS-B`](crate::algorithms::gradient::lbfgsb::LBFGSB), which is typically a binding to
//! `FORTRAN` code in other crates.
//!
//! ## Examples
//!
//! More examples can be found in the `examples` directory of this project. They all contain a
//! `.justfile` which allows the whole example to be run with the command, [`just`](https://github.com/casey/just).
//! To just run the Rust-side code and skip the Python visualization, any of the examples can be run with
//!
//! ```shell
//! cargo r -r --example <example_name>
//! ```
//!
//! ## Bounds
//! All [`Algorithm`](crate::traits::algorithm::Algorithm)s in `ganesh` can be constructed to have access to a feature which allows algorithms which usually function in unbounded parameter spaces to only return results inside a bounding box. This is done via a parameter transformation, similar to that used by [`LMFIT`](https://lmfit.github.io/lmfit-py/) and [`MINUIT`](https://root.cern.ch/doc/master/classTMinuit.html). This transform is not directly useful with algorithms which already have bounded implementations, like [`L-BFGS-B`](crate::algorithms::gradient::lbfgsb::LBFGSB), but it can be combined with other transformations which may be useful to algorithms with bounds. While the user inputs parameters within the bounds, unbounded algorithms can (and in practice will) convert those values to a set of unbounded "internal" parameters. When functions are called, however, these internal parameters are converted back into bounded "external" parameters, via the following transformations:
//!
//! Upper and lower bounds:
//! ```math
//! x_\text{int} = \frac{u}{\sqrt{1 - u^2}}
//! ```
//! ```math
//! x_\text{ext} = c + w \frac{x_\text{int}}{\sqrt{x_\text{int}^2 + 1}}
//! ```
//! where
//! ```math
//! u = \frac{x_\text{ext} - c}{w},\ c = \frac{x_\text{min} + x_\text{max}}{2},\ w = \frac{x_\text{max} - x_\text{min}}{2}
//! ```
//! Upper bound only:
//! ```math
//! x_\text{int} = \frac{1}{2}\left(\frac{1}{(x_\text{max} - x_\text{ext})} - (x_\text{max} - x_\text{ext}) \right)
//! ```
//! ```math
//! x_\text{ext} = x_\text{max} - (\sqrt{x_\text{int}^2 + 1} - x_\text{int})
//! ```
//! Lower bound only:
//! ```math
//! x_\text{int} = \frac{1}{2}\left((x_\text{ext} - x_\text{min}) - \frac{1}{(x_\text{ext} - x_\text{min})} \right)
//! ```
//! ```math
//! x_\text{ext} = x_\text{min} + (\sqrt{x_\text{int}^2 + 1} + x_\text{int})
//! ```
//! While `MINUIT` and `LMFIT` recommend caution in interpreting covariance matrices obtained from
//! fits with bounds transforms, `ganesh` does not, since it implements higher-order derivatives on
//! these bounds while these other libraries use linear approximations.
//!
//! ## Future Plans
//!
//! * Eventually, I would like to implement some more modern gradient-free optimization techniques.
//! * There are probably many optimizations and algorithm extensions that I'm missing right now.
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

/// Module containing core functionality.
pub mod core;

/// Module containing all traits.
pub mod traits;

/// Module containing various minimization algorithms.
pub mod algorithms;

/// Module containing standard functions for testing algorithms.
pub mod test_functions;

/// Module containing `ganesh`-wide error types
pub mod error;

/// Feature-gated Python / `pyo3` wrapper support for downstream Rust crates with Python bindings.
#[cfg(feature = "python")]
pub mod python;

/// A floating-point number type (defaults to [`f64`], see `f32` feature).
#[cfg(not(feature = "f32"))]
pub type Float = f64;

/// A floating-point number type (defaults to [`f64`], see `f32` feature).
#[cfg(feature = "f32")]
pub type Float = f32;

/// Re-export some useful `nalgebra` types for convenience.
pub use nalgebra;
pub use nalgebra::{DMatrix, DVector};

/// The mathematical constant $`\pi`$.
#[cfg(not(feature = "f32"))]
pub const PI: Float = std::f64::consts::PI;

/// The mathematical constant $`\pi`$.
#[cfg(feature = "f32")]
pub const PI: Float = std::f32::consts::PI;

/// A preset minimization algorithm which uses the [L-BFGS-B](`algorithms::gradient::LBFGSB`) algorithm.
///
/// Using the given starting position and optional bounds, this method will attempt to minimize the
/// given problem which must implement the [Gradient](traits::Gradient) trait.
pub fn minimize<P, I, U, Bounds, B, E>(
    problem: &P,
    x0: I,
    user_data: &U,
    bounds: Option<Bounds>,
) -> Result<core::MinimizationSummary, E>
where
    I: AsRef<[Float]>,
    Bounds: IntoIterator<Item = B>,
    B: Into<traits::Bound>,
    P: traits::Gradient<U, E>,
{
    use algorithms::gradient::{LBFGSBConfig, LBFGSB};
    use traits::{Algorithm, SupportsBounds};

    let mut solver = LBFGSB::default();
    let mut config = LBFGSBConfig::new(x0);
    if let Some(bounds) = bounds {
        config = config.with_bounds(bounds);
    }
    solver.process_default(problem, user_data, config)
}

/// A preset minimization algorithm which uses the [Nelder-Mead](`algorithms::gradient_free::NelderMead`) algorithm.
///
/// Using the given starting position and optional bounds, this method will attempt to minimize the
/// given problem which must implement the [CostFunction](traits::CostFunction) trait. This can be
/// used for functions which do not have easily-defined gradients.
pub fn minimize_gradient_free<P, I, U, Bounds, B, E>(
    problem: &P,
    x0: I,
    user_data: &U,
    bounds: Option<Bounds>,
) -> Result<core::MinimizationSummary, E>
where
    I: AsRef<[Float]>,
    Bounds: IntoIterator<Item = B>,
    B: Into<traits::Bound>,
    P: traits::CostFunction<U, E>,
{
    use algorithms::gradient_free::{NelderMead, NelderMeadConfig};
    use traits::{Algorithm, SupportsBounds};

    let mut solver = NelderMead::default();
    let mut config = NelderMeadConfig::new(x0);
    if let Some(bounds) = bounds {
        config = config.with_bounds(bounds);
    }
    solver.process_default(problem, user_data, config)
}

/// A preset Markov Chain Monte Carlo algorithm which uses the [AIES](`algorithms::mcmc::AIES`) algorithm.
///
/// Using a set of starting positions for each walker, this method will attempt to sample `n_steps`
/// positions for each walker from the target distribution. The problem must implement the
/// [`LogDensity`](traits::LogDensity) trait.
pub fn sample<P, I, U, Bounds, B, E>(
    problem: &P,
    x0: I,
    n_steps: usize,
    user_data: &U,
) -> Result<core::MCMCSummary, E>
where
    I: AsRef<[DVector<Float>]>,
    P: traits::LogDensity<U, E>,
    E: From<error::GaneshError>,
{
    use algorithms::mcmc::{AIESConfig, AIES};
    use core::MaxSteps;
    use traits::Algorithm;

    let mut solver = AIES::default();
    let config = AIESConfig::new(x0.as_ref().to_vec())?;
    solver.process(
        problem,
        user_data,
        config,
        AIES::default_callbacks().with_terminator(MaxSteps(n_steps)),
    )
}

/// Run a multistart minimization workflow and collect all run summaries.
///
/// Each run is created by `restart_factory`, and `restart_policy` decides whether another run
/// should be launched based on the current [`core::MultiStartState`]. This lets callers implement
/// fixed restart counts or more adaptive policies that depend on the number of completed restarts
/// and the minima already found.
pub fn minimize_multistart<P, U, E, A, S, F, R>(
    problem: &P,
    user_data: &U,
    restart_factory: &mut F,
    restart_policy: &mut R,
) -> Result<core::MultiStartSummary, E>
where
    S: traits::Status,
    A: traits::Algorithm<P, S, U, E, Summary = core::MinimizationSummary>,
    F: core::RestartFactory<A, P, S, U, E>,
    R: core::RestartPolicy,
{
    core::minimize_multistart(problem, user_data, restart_factory, restart_policy)
}
