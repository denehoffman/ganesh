//! `ganesh` (/ɡəˈneɪʃ/), named after the Hindu god of wisdom, provides minimization and sampling algorithms through scalar- and linear-algebra-generic Rust interfaces. Most users implement [`CostFunction`](crate::traits::CostFunction), optionally implement [`Gradient`](crate::traits::Gradient), and run an [`Algorithm`](crate::traits::Algorithm). Default finite differences keep analytic derivatives optional.
//!
//! # Table of Contents
//! - [Key Features](#key-features)
//! - [Quick Start](#quick-start)
//! - [Algorithms](#algorithms)
//! - [Examples](#examples)
//! - [Parameter Transforms](#parameter-transforms)
//! - [Future Plans](#future-plans)
//! - [Citations](#citations)
//!
//! ## Key Features
//! * Algorithms that are simple to use with sensible defaults.
//! * Traits which make developing future algorithms simple and consistent.
//! * A simple interface that lets new users get started quickly.
//! * A pure Rust implementation of [`L-BFGS-B`](crate::algorithms::gradient::LBFGSB).
//! * `f64`/nalgebra defaults with native `f32`, optional ndarray, and downstream-extensible scalar
//!   and linear algebra traits.
//! * Composable scaling, bounds, and periodic transforms with derivative propagation.
//!
//! Rust precision is selected through type parameters, so `f32` and `f64` can coexist without
//! feature switching. The crate is entirely Python-agnostic; downstream bindings own their Python
//! types and translate them to ordinary `ganesh` Rust values.
//!
//! The `backend-ndarray` feature deliberately does not choose an `ndarray-linalg` LAPACK source.
//! Applications using it should depend on `ndarray-linalg` directly and enable exactly one of its
//! source features before linking an executable that uses those operations.
//!
//! ## Quick Start
//!
//! This crate provides some common test functions in the [`test_functions`] module. Consider the following implementation of the Rosenbrock function:
//!
//! ```rust
//! use ganesh::traits::*;
//! use ganesh::{LinearAlgebra, RealScalar, Vector};
//! use std::convert::Infallible;
//!
//! pub struct Rosenbrock {
//!     pub n: usize,
//! }
//! impl<T, B> CostFunction<T, B> for Rosenbrock
//! where
//!     T: RealScalar,
//!     B: LinearAlgebra<T>,
//! {
//!     fn evaluate(&self, x: &Vector<T, B>, _args: &()) -> Result<T, Infallible> {
//!         Ok((0..(self.n - 1)).fold(T::zero(), |sum, i| {
//!             sum + T::literal(100.0) * (x.get(i + 1) - x.get(i).powi(2)).powi(2)
//!                 + (T::one() - x.get(i)).powi(2)
//!         }))
//!     }
//! }
//! ```
//! To minimize this function, we could consider using the Nelder-Mead algorithm:
//! ```rust
//! use ganesh::algorithms::gradient_free::NelderMead;
//! use ganesh::traits::*;
//! use ganesh::test_functions::Rosenbrock;
//! use std::convert::Infallible;
//! fn main() -> Result<(), Infallible> {
//!     let problem = Rosenbrock { n: 2 };
//!     let mut nm: NelderMead = Default::default();
//!     let result = nm.process_default(&problem, &(), [2.0, 2.0])?;
//!     println!("{}", result);
//!     Ok(())
//! }
//! ```
//!
//! We could also use some more verbose syntax if we wanted additional customization:
//! ```rust
//! use ganesh::algorithms::gradient_free::{NelderMead, NelderMeadConfig};
//! use ganesh::traits::*;
//! use ganesh::test_functions::Rosenbrock;
//! use std::convert::Infallible;
//! fn main() -> Result<(), Infallible> {
//!     let problem = Rosenbrock { n: 2 };
//!     let mut nm: NelderMead = Default::default();
//!     let config = NelderMeadConfig::default();
//!     let result = nm.process(
//!         &problem,
//!         &(),
//!         vec![2.0, 2.0],
//!         config,
//!         NelderMead::default_callbacks(),
//!     )?;
//!     println!("{}", result);
//!     Ok(())
//! }
//! ```
//!
//! The same program is available as `cargo run --example readme`. It outputs
//! ```shell
//! ╭─────────────────────────────────────────────────────────────╮
//! │                    MINIMIZATION SUMMARY                     │
//! ├───────────┬─────────┬─────────────┬─────────┬───────────────┤
//! │ Status    │ f(x)    │ # f(x)      │ # ∇f(x) │ # H(x)        │
//! ├───────────┼─────────┼─────────────┼─────────┼───────────────┤
//! │ Converged │ 0.00009 │ 80          │ 0       │ 0             │
//! ├───────────┼─────────┴─────────────┴─────────┴───────────────┤
//! │ Message   │ Success: term_f = STDDEV                        │
//! ├───────────┴─────────────────────────────────┬───────────────┤
//! │ Parameters                                  │ Bounds        │
//! ├───────────┬─────────┬─────────────┬─────────┼───────┬───────┤
//! │ Name      │ Value   │ Uncertainty │ Initial │ Lower │ Upper │
//! │ x_0       │ 1.00410 │ NaN         │ 2.00000 │ −∞    │ ∞     │
//! │ x_1       │ 1.00909 │ NaN         │ 2.00000 │ −∞    │ ∞     │
//! ╰───────────┴─────────┴─────────────┴─────────┴───────┴───────╯
//! ```
//!
//! The `ganesh` crate uses algorithm methods such as [`Algorithm::process`](`crate::traits::algorithm::Algorithm::process`),
//! [`Algorithm::process_with_default_callbacks`](`crate::traits::algorithm::Algorithm::process_with_default_callbacks`),
//! and [`Algorithm::process_default`](`crate::traits::algorithm::Algorithm::process_default`) as the primary
//! entrypoints for running optimizers and samplers.
//!
//! ## Algorithms
//!
//! At the moment, `ganesh` contains the following [`Algorithm`](crate::traits::algorithm::Algorithm)s:
//! - Gradient descent/quasi-Newton:
//!   - [`L-BFGS-B`](crate::algorithms::gradient::LBFGSB)
//!   - [`Adam`](crate::algorithms::gradient::Adam) (for stochastic [`CostFunction`](crate::traits::CostFunction)s)
//!   - [`ConjugateGradient`](crate::algorithms::gradient::ConjugateGradient)
//!   - [`TrustRegion`](crate::algorithms::gradient::TrustRegion)
//! - Gradient-free:
//!   - [`Nelder-Mead`](crate::algorithms::gradient_free::NelderMead)
//!   - [`Simulated Annealing`](crate::algorithms::gradient_free::SimulatedAnnealing)
//!   - [`CMAES`](crate::algorithms::gradient_free::CMAES)
//!   - [`DifferentialEvolution`](crate::algorithms::gradient_free::DifferentialEvolution)
//! - Markov Chain Monte Carlo (MCMC):
//!   - [`AIES`](crate::algorithms::mcmc::AIES)
//!   - [`ESS`](crate::algorithms::mcmc::ESS)
//! - Swarms:
//!   - [`PSO`](crate::algorithms::particles::PSO)
//!
//! All algorithms are written in pure Rust, including [`L-BFGS-B`](crate::algorithms::gradient::LBFGSB), which is typically a binding to
//! `FORTRAN` code in other crates.
//!
//! ## Examples
//!
//! The `examples` directory contains a compact generic numeric example and polished showcases for
//! optimization, fitting, multistart minimization, swarms, and ensemble sampling. `periodic_fit`
//! demonstrates mixed scaling, positivity, and cyclic phase coordinates; `multistart` demonstrates
//! deterministic basin discovery. Run any Rust example directly:
//!
//! ```shell
//! cargo run --release --example pso
//! ```
//! Each showcase directory also contains a `.justfile`. For example,
//! `just --justfile examples/pso/.justfile show` runs the Rust code and renders the visualization
//! with a standalone `uv` script whose Python dependencies are pinned inline.
//!
//! ## Parameter Transforms
//! Generic algorithms accept [`Bounds`] as a smooth transform; [`L-BFGS-B`](crate::algorithms::gradient::LBFGSB) instead keeps native projected bounds. While users provide external parameters, transformed algorithms operate on internal coordinates and evaluate problems after converting back to external coordinates:
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
//! Transforms compose in application order with [`Transform::then`].
//! [`PeriodicTransform`](crate::traits::PeriodicTransform) canonicalizes cyclic coordinates into a
//! half-open interval while leaving the optimizer's internal coordinate unbounded:
//! ```math
//! x_\text{ext} = a + (x_\text{int} - a) \mathbin{\operatorname{rem\_euclid}} (b-a)
//! ```
//! Away from the seam its Jacobian is the identity and its component Hessians are zero. Objective
//! values and derivatives must agree across the seam. The repeated lift is suitable for
//! minimization, but not for MCMC because it creates an improper internal target.
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

/// Default floating-point type used by convenience APIs and reporting utilities.
///
/// Generic algorithms select their scalar through type parameters.
pub type Float = f64;

/// Re-export some useful `nalgebra` types for convenience.
pub use nalgebra;
pub use nalgebra::{DMatrix, DVector};

#[cfg(feature = "backend-ndarray")]
pub use core::NdArrayProvider;
/// Re-export crate-owned scalar and linear algebra traits for generic optimizer APIs.
pub use core::{
    Determinant, LinearAlgebra, LinearSolve, Matrix, NalgebraProvider, PseudoInverse, RandomScalar,
    RealScalar, Scalar, SymmetricEigen, Vector,
};
pub use traits::{
    Bounds, IdentityTransform, ScalarBound, ScaleTransform, Transform, TransformedProblem,
};

/// The mathematical constant $`\pi`$.
pub const PI: Float = std::f64::consts::PI;
