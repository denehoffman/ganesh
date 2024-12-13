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

`ganesh` (/ɡəˈneɪʃ/), named after the Hindu god of wisdom, provides several common minimization algorithms as well as a straightforward, trait-based interface to create your own extensions. This crate is intended to be as simple as possible. The user needs to implement the `Function` trait on some struct which will take a vector of parameters and return a single-valued `Result` ($`f(\mathbb{R}^n) \to \mathbb{R}`$). Users can optionally provide a gradient function to speed up some algorithms, but a default central finite-difference implementation is provided so that all algorithms will work out of the box.

> [!CAUTION]
> This crate is still in an early development phase, and the API is not stable. It can (and likely will) be subject to breaking changes before the 1.0.0 version release (and hopefully not many after that).

# Table of Contents
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Bounds](#bounds)
- [Future Plans](#future-plans)
- [Citations](#citations)

# Key Features
* Simple but powerful trait-oriented library which tries to follow the Unix philosophy of "do one thing and do it well".
* Generics to allow for different numeric types to be used in the provided algorithms.
* Algorithms that are simple to use with sensible defaults.
* Traits which make developing future algorithms simple and consistent.
* Pressing `Ctrl-C` during a fit will still output a [`Status`], but the fit message will
  indicate that the fit was ended by the user.

# Quick Start

This crate provides some common test functions in the `test_functions` module. Consider the following implementation of the Rosenbrock function:

```rust
use std::convert::Infallible;
use ganesh::prelude::*;

pub struct Rosenbrock {
    pub n: usize,
}
impl Function<f64, (), Infallible> for Rosenbrock {
    fn evaluate(&self, x: &[f64], _user_data: &mut ()) -> Result<f64, Infallible> {
        Ok((0..(self.n - 1))
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum())
    }
}
```
To minimize this function, we could consider using the Nelder-Mead algorithm:
```rust
use ganesh::prelude::*;
use ganesh::algorithms::NelderMead;

fn main() -> Result<(), Infallible> {
    let problem = Rosenbrock { n: 2 };
    let nm = NelderMead::default();
    let mut m = Minimizer::new(&nm, 2);
    let x0 = &[2.0, 2.0];
    m.minimize(&problem, x0, &mut ())?;
    println!("{}", m.status);
    Ok(())
}
```

This should output
```shell
╒══════════════════════════════════════════════════════════════════════════════════════════════╕
│                                         FIT RESULTS                                          │
╞════════════════════════════════════════════╤════════════════════╤═════════════╤══════════════╡
│ Status: Converged                          │ fval:    +2.281E-4 │ #fcn:    76 │ #grad:    76 │
├────────────────────────────────────────────┴────────────────────┴─────────────┴──────────────┤
│ Message: term_f = STDDEV                                                                     │
├───────╥──────────────┬──────────────╥──────────────┬──────────────┬──────────────┬───────────┤
│ Par # ║        Value │  Uncertainty ║      Initial │       -Bound │       +Bound │ At Limit? │
├───────╫──────────────┼──────────────╫──────────────┼──────────────┼──────────────┼───────────┤
│     0 ║     +1.001E0 │    +8.461E-1 ║     +2.000E0 │         -inf │         +inf │           │
│     1 ║     +1.003E0 │     +1.695E0 ║     +2.000E0 │         -inf │         +inf │           │
└───────╨──────────────┴──────────────╨──────────────┴──────────────┴──────────────┴───────────┘
```

# Bounds
All minimizers in `ganesh` have access to a feature which allows algorithms which usually function in unbounded parameter spaces to only return results inside a bounding box. This is done via a parameter transformation, the same one used by [`LMFIT`](https://lmfit.github.io/lmfit-py/) and [`MINUIT`](https://root.cern.ch/doc/master/classTMinuit.html). This transform is not enacted on algorithms which already have bounded implementations, like `L-BFGS-B`. While the user inputs parameters within the bounds, unbounded algorithms can (and in practice will) convert those values to a set of unbounded "internal" parameters. When functions are called, however, these internal parameters are converted back into bounded "external" parameters, via the following transformations:

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
As noted in the documentation for both `LMFIT` and `MINUIT`, these bounds should be used with caution. They turn linear problems into nonlinear ones, which can mess with error propagation and even fit convergence, not to mention increase function complexity. Methods which output covariance matrices need to be adjusted if bounded, and `MINUIT` recommends fitting a second time near a minimum without bounds to ensure proper error propagation.

# Future Plans

* Eventually, I would like to implement more modern gradient-free optimization techniques.
* There are probably many optimizations and algorithm extensions that I'm missing right now because I just wanted to get it working first.
* There should be more tests (as usual).


# Citations
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
