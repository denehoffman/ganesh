<p align="center">
  <img
    width="800"
    src="media/wordmark.png"
  />
</p>
<p align="center">
  <h1 align="center">Function Minimization in Rust, Simplified</h1>
</p>
<p align="center">
  <a href="https://github.com/denehoffman/ganesh/releases" alt="Releases">
    <img alt="GitHub Release" src="https://img.shields.io/github/v/release/denehoffman/ganesh?style=for-the-badge&logo=github"></a>
  <a href="https://github.com/denehoffman/ganesh/commits/main/" alt="Latest Commits">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/denehoffman/ganesh?style=for-the-badge&logo=github"></a>
  <a href="https://github.com/denehoffman/ganesh/actions" alt="Build Status">
    <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/denehoffman/ganesh/rust.yml?style=for-the-badge&logo=github"></a>
  <a href="LICENSE-APACHE" alt="License">
    <img alt="GitHub License" src="https://img.shields.io/github/license/denehoffman/ganesh?style=for-the-badge"></a>
  <a href="https://crates.io/crates/ganesh" alt="Ganesh on crates.io">
    <img alt="Crates.io Version" src="https://img.shields.io/crates/v/ganesh?style=for-the-badge&logo=rust&logoColor=red&color=red"></a>
  <a href="https://docs.rs/ganesh" alt="Rustitude documentation on docs.rs">
    <img alt="docs.rs" src="https://img.shields.io/docsrs/ganesh?style=for-the-badge&logo=rust&logoColor=red"></a>
  <a href="https://app.codecov.io/github/denehoffman/ganesh/tree/main/" alt="Codecov coverage report">
    <img alt="Codecov" src="https://img.shields.io/codecov/c/github/denehoffman/ganesh?style=for-the-badge&logo=codecov"></a>
  <a href="https://codspeed.io/denehoffman/ganesh">
    <img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fcodspeed.io%2Fbadge.json&style=for-the-badge" alt="CodSpeed"/></a>
</p>

<!-- cargo-rdme start -->

`ganesh` (/ɡəˈneɪʃ/), named after the Hindu god of wisdom, provides several common minimization algorithms as well as a straightforward, trait-based interface to create your own extensions. This crate is intended to be as simple as possible. The user needs to implement the `CostFunction` trait on some struct which will take a vector of parameters and return a single-valued `Result` ($`f(\mathbb{R}^n) \to \mathbb{R}`$). Users can optionally provide a gradient function to speed up some algorithms, but a default central finite-difference implementation is provided so that all algorithms will work out of the box.

<div class="warning">

This crate is still in an early development phase, and the API is not stable. It can (and likely will) be subject to breaking changes before the 1.0.0 version release (and hopefully not many after that).

</div>

# Table of Contents
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [MCMC](#mcmc)
- [Bounds](#bounds)
- [Future Plans](#future-plans)
- [Citations](#citations)

## Key Features
* Algorithms that are simple to use with sensible defaults.
* Traits which make developing future algorithms simple and consistent.
* A simple interface that lets new users get started quickly.

## Quick Start

This crate provides some common test functions in the [`test_functions`](https://docs.rs/ganesh/latest/ganesh/test_functions/) module. Consider the following implementation of the Rosenbrock function:

```rust
use ganesh::traits::*;
use ganesh::{core::Engine, Float};
use std::convert::Infallible;

pub struct Rosenbrock {
    pub n: usize,
}
impl CostFunction<(), Infallible> for Rosenbrock {
    fn evaluate(&self, x: &[Float], _user_data: &mut ()) -> Result<Float, Infallible> {
        Ok((0..(self.n - 1))
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum())
    }
}
```
To minimize this function, we could consider using the Nelder-Mead algorithm:
```rust
use ganesh::algorithms::gradient_free::NelderMead;
use ganesh::traits::*;
use ganesh::{core::Engine, Float};
use std::convert::Infallible;

fn main() -> Result<(), Infallible> {
    let problem = Rosenbrock { n: 2 };
    let nm = NelderMead::default();
    let mut m = Engine::new(nm);
    m.configure(|c| c.with_x0([2.0, 2.0]));
    m.process(&problem)?;
    println!("{}", m.result);
    Ok(())
}
```

This should output
```shell
╭──────────────────────────────────────────────────────────────────╮
│                                                                  │
│                           FIT RESULTS                            │
│                                                                  │
├───────────┬───────────────────┬────────────────┬─────────────────┤
│ Status    │ f(x)              │ #f(x)          │ #∇f(x)          │
├───────────┼───────────────────┼────────────────┼─────────────────┤
│ Converged │ 0.00023           │ 76             │ 0               │
├───────────┼───────────────────┴────────────────┴─────────────────┤
│           │                                                      │
│ Message   │ term_f = STDDEV                                      │
│           │                                                      │
├───────────┴─────────────────────────────┬────────────┬───────────┤
│ Parameter                               │ Bound      │ At Limit? │
├───────────┬─────────┬─────────┬─────────┼──────┬─────┼───────────┤
│           │ =       │ σ       │ 0       │ -    │ +   │           │
├───────────┼─────────┼─────────┼─────────┼──────┼─────┼───────────┤
│ x_0       │ 1.00081 │ 0.84615 │ 2.00000 │ -inf │ inf │ No        │
│ x_1       │ 1.00313 │ 1.69515 │ 2.00000 │ -inf │ inf │ No        │
╰───────────┴─────────┴─────────┴─────────┴──────┴─────┴───────────╯
```

## Algorithms

At the moment, `ganesh` contains the following `Algorithm`s:
- Gradient descent/quasi-Newton:
  - `LBFGSB`
  - `ConjugateGradient`
  - `Adam` (for stochastic
  `CostFunction`s)
- Gradient-free:
  - `NelderMead`
  - `SimulatedAnnealing`
- Markov Chain Monte Carlo (MCMC):
  - `AIES`
  - `ESS`
- Swarms:
  - `PSO` (a basic form of particle swarm optimization)

All algorithms are written in pure Rust, including `L-BFGS-B` which is typically a binding to
`FORTRAN` code in other crates.

## Examples

More examples can be found in the `examples` directory of this project. They all contain a
`.justfile` which allows the whole example to be run with the command, [`just`](https://github.com/casey/just).
To just run the Rust-side code and skip the Python visualization, any of the examples can be run with

```shell
cargo r -r --example <example_name>
```

## Bounds
All [`Algorithm`]s in `ganesh` can be constructed to have access to a feature which allows algorithms which usually function in unbounded parameter spaces to only return results inside a bounding box. This is done via a parameter transformation, the same one used by [`LMFIT`](https://lmfit.github.io/lmfit-py/) and [`MINUIT`](https://root.cern.ch/doc/master/classTMinuit.html). This transform is not enacted on algorithms which already have bounded implementations, like [`L-BFGS-B`](`algorithms::gradient::lbfgsb`). While the user inputs parameters within the bounds, unbounded algorithms can (and in practice will) convert those values to a set of unbounded "internal" parameters. When functions are called, however, these internal parameters are converted back into bounded "external" parameters, via the following transformations:

Upper and lower bounds:
```math
x_\text{int} = \arcsin\left(2\frac{x_\text{ext} - x_\text{min}}{x_\text{max} - x_\text{min}} - 1\right)
```
```math
x_\text{ext} = x_\text{min} + \left(\sin(x_\text{int}) + 1\right)\frac{x_\text{max} - x_\text{min}}{2}
```
Upper bound only:
```math
x_\text{int} = \sqrt{(x_\text{max} - x_\text{ext} + 1)^2 - 1}
```
```math
x_\text{ext} = x_\text{max} + 1 - \sqrt{x_\text{int}^2 + 1}
```
Lower bound only:
```math
x_\text{int} = \sqrt{(x_\text{ext} - x_\text{min} + 1)^2 - 1}
```
```math
x_\text{ext} = x_\text{min} - 1 + \sqrt{x_\text{int}^2 + 1}
```
As noted in the documentation for both `LMFIT` and `MINUIT`, these bounds should be used with caution. They turn linear problems into nonlinear ones, which can mess with error propagation and even fit convergence, not to mention increase function complexity. Methods which output covariance matrices need to be adjusted if bounded, and `MINUIT` recommends fitting a second time near a minimum without bounds to ensure proper error propagation. Some methods, like `L-BFGS-B`, are written with implicit bounds handling, and these transformations are not performed in such cases.

## Future Plans

* Eventually, I would like to implement some more modern gradient-free optimization techniques.
* There are probably many optimizations and algorithm extensions that I'm missing right now.
* There should be more tests and documentation (as usual).

## Citations
While this project does not currently have an associated paper, most of the algorithms it implements do, and they should be cited appropriately. Citations are also generally available in the documentation.

### ESS MCMC Sampler
```text
@article{karamanis2020ensemble,
  title = {Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions},
  author = {Karamanis, Minas and Beutler, Florian},
  journal = {arXiv preprint arXiv: 2002.06212},
  year = {2020}
}
```

### scikit-learn (used in constructing a Bayesian Mixture Model in the Global ESS step)
```text
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
          and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
          and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
          Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
```

### AIES MCMC Sampler
```text
@article{Goodman2010,
  title = {Ensemble samplers with affine invariance},
  volume = {5},
  ISSN = {1559-3940},
  url = {http://dx.doi.org/10.2140/camcos.2010.5.65},
  DOI = {10.2140/camcos.2010.5.65},
  number = {1},
  journal = {Communications in Applied Mathematics and Computational Science},
  publisher = {Mathematical Sciences Publishers},
  author = {Goodman,  Jonathan and Weare,  Jonathan},
  year = {2010},
  month = jan,
  pages = {65–80}
}
```

<!-- cargo-rdme end -->
