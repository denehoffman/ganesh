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

`ganesh` (/ɡəˈneɪʃ/), named after the Hindu god of wisdom, provides several common minimization algorithms as well as a straightforward, trait-based interface to create your own extensions. This crate is intended to be as simple as possible. For most minimization problems user needs to implement the [`CostFunction`](https://docs.rs/ganesh/latest/ganesh/traits/cost_function/trait.CostFunction.html) trait on some struct which will take a vector of parameters and return a single-valued `Result` ($`f(\mathbb{R}^n) \to \mathbb{R}`$). Some algorithms require a gradient which can be implemented via the [`Gradient`](https://docs.rs/ganesh/latest/ganesh/traits/cost_function/trait.Gradient.html) trait. While users may provide an analytic gradient function to speed up some algorithms, this trait comes with a default central finite-difference implementation so that all algorithms will work out of the box as long as the cost function is well-defined.

# Table of Contents
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Algorithms](#algorithms)
- [Examples](#examples)
- [Bounds](#bounds)
- [Future Plans](#future-plans)
- [Citations](#citations)

## Key Features
* Algorithms that are simple to use with sensible defaults.
* Traits which make developing future algorithms simple and consistent.
* A simple interface that lets new users get started quickly.
* The first (and possibly only) pure Rust implementation of the [`L-BFGS-B`](https://docs.rs/ganesh/latest/ganesh/algorithms/gradient/lbfgsb/struct.LBFGSB.html) algorithm.

## Quick Start

This crate provides some common test functions in the [`test_functions`](https://docs.rs/ganesh/latest/ganesh/test_functions/) module. Consider the following implementation of the Rosenbrock function:

```rust
use ganesh::traits::*;
use ganesh::{Float, DVector};
use std::convert::Infallible;

pub struct Rosenbrock {
    pub n: usize,
}
impl CostFunction for Rosenbrock {
    fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
        Ok((0..(self.n - 1))
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum())
    }
}
```
To minimize this function, we could consider using the Nelder-Mead algorithm:
```rust
use ganesh::algorithms::gradient_free::{NelderMead, NelderMeadConfig};
use ganesh::traits::*;
use ganesh::{Float, DVector};
use std::convert::Infallible;

fn main() -> Result<(), Infallible> {
    let problem = Rosenbrock { n: 2 };
    let mut nm = NelderMead::default();
    let result = nm.process(&problem,
                            &(),
                            NelderMeadConfig::new([2.0, 2.0]),
                            NelderMead::default_callbacks())?;
    println!("{}", result);
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

At the moment, `ganesh` contains the following [`Algorithm`](https://docs.rs/ganesh/latest/ganesh/traits/algorithm/trait.Algorithm.html)s:
- Gradient descent/quasi-Newton:
  - [`L-BFGS-B`](https://docs.rs/ganesh/latest/ganesh/algorithms/gradient/lbfgsb/struct.LBFGSB.html)
  - [`Adam`](https://docs.rs/ganesh/latest/ganesh/algorithms/gradient/adam/struct.Adam.html) (for stochastic [`CostFunction`](https://docs.rs/ganesh/latest/ganesh/traits/cost_function/trait.CostFunction.html)s)
- Gradient-free:
  - [`Nelder-Mead`](https://docs.rs/ganesh/latest/ganesh/algorithms/gradient_free/nelder_mead/struct.NelderMead.html)
  - [`Simulated Annealing`](https://docs.rs/ganesh/latest/ganesh/algorithms/gradient_free/simulated_annealing/struct.SimulatedAnnealing.html)
- Markov Chain Monte Carlo (MCMC):
  - [`AIES`](https://docs.rs/ganesh/latest/ganesh/algorithms/mcmc/aies/struct.AIES.html)
  - [`ESS`](https://docs.rs/ganesh/latest/ganesh/algorithms/mcmc/ess/struct.ESS.html)
- Swarms:
  - [`PSO`](https://docs.rs/ganesh/latest/ganesh/algorithms/particles/pso/struct.PSO.html) (a basic form of particle swarm optimization)

All algorithms are written in pure Rust, including [`L-BFGS-B`](https://docs.rs/ganesh/latest/ganesh/algorithms/gradient/lbfgsb/struct.LBFGSB.html), which is typically a binding to
`FORTRAN` code in other crates.

## Examples

More examples can be found in the `examples` directory of this project. They all contain a
`.justfile` which allows the whole example to be run with the command, [`just`](https://github.com/casey/just).
To just run the Rust-side code and skip the Python visualization, any of the examples can be run with

```shell
cargo r -r --example <example_name>
```

## Bounds
All [`Algorithm`](https://docs.rs/ganesh/latest/ganesh/traits/algorithm/trait.Algorithm.html)s in `ganesh` can be constructed to have access to a feature which allows algorithms which usually function in unbounded parameter spaces to only return results inside a bounding box. This is done via a parameter transformation, similar to that used by [`LMFIT`](https://lmfit.github.io/lmfit-py/) and [`MINUIT`](https://root.cern.ch/doc/master/classTMinuit.html). This transform is not directly useful with algorithms which already have bounded implementations, like [`L-BFGS-B`](https://docs.rs/ganesh/latest/ganesh/algorithms/gradient/lbfgsb/struct.LBFGSB.html), but it can be combined with other transformations which may be useful to algorithms with bounds. While the user inputs parameters within the bounds, unbounded algorithms can (and in practice will) convert those values to a set of unbounded "internal" parameters. When functions are called, however, these internal parameters are converted back into bounded "external" parameters, via the following transformations:

Upper and lower bounds:
```math
x_\text{int} = \frac{u}{\sqrt{1 - u^2}}
```
```math
x_\text{ext} = c + w \frac{x_\text{int}}{\sqrt{x_\text{int}^2 + 1}}
```
where
```math
u = \frac{x_\text{ext} - c}{w},\ c = \frac{x_\text{min} + x_\text{max}}{2},\ w = \frac{x_\text{max} - x_\text{min}}{2}
```
Upper bound only:
```math
x_\text{int} = \frac{1}{2}\left(\frac{1}{(x_\text{max} - x_\text{ext})} - (x_\text{max} - x_\text{ext}) \right)
```
```math
x_\text{ext} = x_\text{max} - (\sqrt{x_\text{int}^2 + 1} - x_\text{int})
```
Lower bound only:
```math
x_\text{int} = \frac{1}{2}\left((x_\text{ext} - x_\text{min}) - \frac{1}{(x_\text{ext} - x_\text{min})} \right)
```
```math
x_\text{ext} = x_\text{min} + (\sqrt{x_\text{int}^2 + 1} + x_\text{int})
```
While `MINUIT` and `LMFIT` recommend caution in interpreting covariance matrices obtained from
fits with bounds transforms, `ganesh` does not, since it implements higher-order derivatives on
these bounds while these other libraries use linear approximations.

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
