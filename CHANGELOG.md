# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.5](https://github.com/denehoffman/ganesh/compare/v0.8.4...v0.8.5) - 2024-09-09

### Added

- reboot L-BFGS-B on invalid line searches

### Fixed

- follow strong Wolfe condition a bit more carefully
- make bounds inclusive
- ensure sufficient decrease is met before marking line search as valid
- make `g_eval` increment gradient evaluations rather than function evaluations
- use  trait to implement ordering on float-like generics

### Other

- remove unused import
- remove comment

## [0.8.4](https://github.com/denehoffman/ganesh/compare/v0.8.3...v0.8.4) - 2024-09-03

### Fixed
- use absolute value for absolute tolerance

### Other
- reverse some dot products with the wrong dimensions

## [0.8.3](https://github.com/denehoffman/ganesh/compare/v0.8.2...v0.8.3) - 2024-09-03

### Fixed
- switch sign on function termination condition

## [0.8.2](https://github.com/denehoffman/ganesh/compare/v0.8.1...v0.8.2) - 2024-09-03

### Added
- add function value terminators for BFGS algorithms
- add function value terminators for BFGS algorithms

## [0.8.1](https://github.com/denehoffman/ganesh/compare/v0.8.0...v0.8.1) - 2024-09-03

### Added
- add gradient tolerance to L-BFGS-B

### Other
- Merge branch 'main' into development
- export BFGS methods in  mod

## [0.8.0](https://github.com/denehoffman/ganesh/compare/v0.7.1...v0.8.0) - 2024-09-03

### Added
- add L-BFGS-B algorithm
- update line search to take a `max_step` optional argument and return a bool flag of validity rather than an `Option`
- add `LineSearch` trait and implementations of BFGS and L-BFGS algorithms
- update `NelderMead` to count gradient evals and use bounded interface
- add bounded evaluation shortcuts to `Function` trait and count gradient evaluations in `Status`

### Fixed
- simplify logic by removing internal `m`
- change to inequality to ensure a proper status message if the max iterations are passed

### Other
- fix brackets in readme and update main lib docs
- update readme
- remove unused collections module

## [0.7.1](https://github.com/denehoffman/ganesh/compare/v0.7.0...v0.7.1) - 2024-08-23

### Other
- fix doctests
- make minimize return `Result<(), E>` and store `Status` in the `Minimizer` struct

## [0.7.0](https://github.com/denehoffman/ganesh/compare/v0.6.0...v0.7.0) - 2024-08-23

### Added
- add useful assert warning for trying to construct a `NelderMead` `Simplex` with fewer than 2 points
- add check to make sure starting position is within bounds
- add display method, methods for getting `lower` and `upper` bounds, and `contains` method for `Bounds`
- add `Debug`s to `NelderMead`
- add preliminary implementation of BFGS algorithm
- add method to return the gradient and inverse of Hessian matrix

### Fixed
- remove tracking `main.rs`, which I use for quick demos
- adaptive Nelder-Mead now requires inputting the dimension
- remove out-of-bounds issue
- step direction should be opposite the gradient
- `p` is `-grad_f` so this was right all along
- allow expect in Hessian inverse function
- update BFGS algorithm to recent changes with ganesh
- change `learning_rate` to an `Option` in gradient descent

### Other
- adds documentation to all parts of crate, additionally makes some `Algorithm` methods return `Result`s now
- fix typo in example
- update dependencies
- update licensing
- switch license to MIT
- add Bounds section to TOC
- correct statements about `Function` trait in readme
- typo in readme
- update README.md
- major rewrite of library, adds experimental bounds to Nelder Mead
- qualify path to `abs` function
- Merge remote-tracking branch 'origin/bfgs' into development
- change slice to `DVector` in documentation
- update docs and fix links/footnotes

## [0.6.0](https://github.com/denehoffman/ganesh/compare/v0.5.0...v0.6.0) - 2024-08-17

### Added
- reduces the `Field` trait to use `num` traits rather than `nalgebra`'s `RealField`

### Fixed
- ensure all methods use the `Field` trait rather than just `Float` for better compatibility
- re-export `nalgebra::DVector`

### Other
- fix some of the documentation to reflect recent changes to the crate

## [0.5.0](https://github.com/denehoffman/ganesh/compare/v0.4.0...v0.5.0) - 2024-08-15

### Added
- change most slice types to `nalgebra::DVector`s to make algorithms more ergonomic, add `LineSearch` algorithms

## [0.4.0](https://github.com/denehoffman/ganesh/compare/v0.3.1...v0.4.0) - 2024-07-30

### Other
- undo changes to previous version, lifetimes make things more difficult to work with for end-users. Removed NelderMeadMessage.

## [0.3.1](https://github.com/denehoffman/ganesh/compare/v0.3.0...v0.3.1) - 2024-07-30

### Added
- change functions to references to avoid cloning any underlying function data

## [0.3.0](https://github.com/denehoffman/ganesh/compare/v0.2.0...v0.3.0) - 2024-07-19

### Added
- switch &Option<args> to Option<&args> and remove messages in favor of extending Minimizer trait
- add Send/Sync to Function

### Fixed
- change callback to no longer be optional, this just required typing None::<fn(&_)> everywhere which is way uglier than |_|{}
- make callback optional to avoid toilet bowl closure

### Other
- Merge branch 'development' of github.com:denehoffman/ganesh into development
- update crate docs to reflect new changes
- add wordmark
- add logo to readme
- remove num::Float trait dependence
- add logo
- Merge branch 'main' into development

## [0.2.0](https://github.com/denehoffman/ganesh/compare/v0.1.0...v0.2.0) - 2024-07-14

### Fixed
- re-implement args that were lost in merge
- move main traits to core module and modify gradient and hessian methods to work better at larger values

### Other
- update docstrings to reflect arguments
- Merge branch 'development' of https://github.com/denehoffman/ganesh into development
- add benchmark
- release

## [0.1.0](https://github.com/denehoffman/ganesh/releases/tag/v0.1.0) - 2024-07-13

### Added
- switch to typed-builder for Result-less builder patterns
- first commit, basic traits, some test functions, and some basic optimization algorithms

### Fixed
- ignore doctests that aren't meant to be tests

### Other
- add metadata to Cargo.toml
- create release.yml
- fix \geq symbol
- update documentation to match README and newer features
- Create LICENSE
- add workflows
- many interconnected changes that I don't care to write individual commits for
- remove BFGS algorithm (for now)
- add KaTeX header
- create README.md
