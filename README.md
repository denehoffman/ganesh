<p align="center">
  <h1 align="center">Ganesh</h1>
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
  <a href="LICENSE" alt="License">
    <img alt="GitHub License" src="https://img.shields.io/github/license/denehoffman/ganesh?style=for-the-badge"></a>
  <a href="https://crates.io/crates/ganesh" alt="Ganesh on crates.io">
    <img alt="Crates.io Version" src="https://img.shields.io/crates/v/ganesh?style=for-the-badge&logo=rust&logoColor=red&color=red"></a>
  <a href="https://docs.rs/ganesh" alt="Rustitude documentation on docs.rs">
    <img alt="docs.rs" src="https://img.shields.io/docsrs/ganesh?style=for-the-badge&logo=rust&logoColor=red"></a>
  <a href="https://app.codecov.io/github/denehoffman/ganesh/tree/main/" alt="Codecov coverage report">
    <img alt="Codecov" src="https://img.shields.io/codecov/c/github/denehoffman/ganesh?style=for-the-badge&logo=codecov"></a>
</p>

`ganesh`, (/ɡəˈneɪʃ/) named after the Hindu god of wisdom, provides several common minimization algorithms as well as a straightforward, trait-based interface to create your own extensions. This crate is intended to be as simple as possible. The user needs to implement the `Function` trait on some struct which will take a vector of parameters and return a single-valued `Result` ($`f(\mathbb{R}^n) \to \mathbb{R}`$). Users can optionally provide gradient and Hessian functions to speed up some algorithms, but a default finite-difference implementation is provided so that all algorithms will work out of the box.

> [!CAUTION]
> This crate is still in an early development phase, and the API is not stable. It can (and likely will) be subject to breaking changes before the 1.0.0 version release (and hopefully not many after that).

# Table of Contents
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Future Plans](#future-plans)

# Key Features
* Simple but powerful trait-oriented library which tries to follow the Unix philosophy of "do one thing and do it well".
* Generics to allow for different numeric types to be used in the provided algorithms.
* Algorithms that are simple to use with sensible defaults.
* Traits which make developing future algorithms simple and consistent.

# Quick Start

This crate provides some common test functions in the `test_functions` module. Consider the following implementation of the Rosenbrock function:

```rust
use ganesh::prelude::*;
use std::convert::Infallible;
pub struct Rosenbrock {
    /// Number of dimensions (must be at least 2)
    pub n: usize,
}
impl Function<f64, Infallible> for Rosenbrock {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
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

let func = Rosenbrock { n: 2 };
let mut m = NelderMead::new(func, &[-2.3, 3.4], None);
let status = minimize!(m).unwrap();
let (x_best, fx_best) = m.best();
println!("x: {:?}\nf(x): {}", x_best, fx_best);
```

This should output
```shell
x: [0.9999459113507765, 0.9998977381285472]
f(x): 0.000000006421349269800761
```

I have ignored the `status` variable here, but in practice, the `Minimizer::minimize` method should return the last message sent by the algorithm. This can indicate the status of a fit without explicitly causing an error. This makes it easier to debug, since it can be tedious to have two separate error types, one for the function and one for the algorithm, returned by the minimization (functions can always be failable in this crate). We could also swap the `f64`s for `f32`s (or any type which implements the `Field` trait) in the Rosenbrock implementation. Additionally, if we wanted to modify any of the hyperparameters in the fitting algorithm, we could use `NelderMeadOptions::builder()` and pass it as the third argument in the `NelderMead::new` constructor. Finally, all algorithm implementations are constructed to pass a unique message type to a callback function. For `NelderMead`, we could do the following:
```rust
let status = minimize!(m, |message| println!("step: {}\nx: {:?}\nf(x): {}", message.step, message.x, message.fx)).unwrap();
```
This will print out the current step, the best position found by the optimizer at that step, and the function's evaluation at that position for each step in the algorithm. You can use the step number to limit printing (print only steps divisible by 100, for example).

The `minimize!` macro exists to simplify the `Minimizer::minimize<Callback: Fn(M)>(&mut self, callback: Callback)` call, which looks [a bit ugly](https://enet4.github.io/rust-tropes/#toilet-closure) if you don't actually want a callback.

# Future Plans

* Eventually, I would like to implement BGFS and variants, MCMC algorithms, and some more modern gradient-free optimization techniques.
* There are probably many optimizations and algorithm extensions that I'm missing right now because I just wanted to get it working first.
* Reduce the amount of `Vec::clone`s in the algorithms.
* A test suite
