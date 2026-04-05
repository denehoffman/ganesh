# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.26.2](https://github.com/denehoffman/ganesh/compare/v0.26.1...v0.26.2) (2026-04-05)


### Features

* **python:** Publish type stubs ([8df49b6](https://github.com/denehoffman/ganesh/commit/8df49b6e91303f3fa9a711e59b67e89eb808f7b9))


### Bug Fixes

* **python:** Validate extracted objects structurally ([7c1363f](https://github.com/denehoffman/ganesh/commit/7c1363fc663983785a8e180f1c1a61acc5ffbd16))

## [0.26.1](https://github.com/denehoffman/ganesh/compare/v0.26.0...v0.26.1) (2026-04-04)


### Features

* **python:** Add status wrappers and summary conversions ([fc2b855](https://github.com/denehoffman/ganesh/commit/fc2b8559de8361f69b091ab8cd26ca2dd6a7d198))


### Bug Fixes

* **ci:** Checkout repo before cargo publish ([d07137d](https://github.com/denehoffman/ganesh/commit/d07137d81a584f69161dfdb64c29cc3e4b4b56fd))
* **python:** Configure ty for dynamic package exports ([1e1ab9e](https://github.com/denehoffman/ganesh/commit/1e1ab9e99816ac4eb86c0bfb60dc177940710de9))
* **python:** Stabilize ty checks for dynamic exports ([0810ef7](https://github.com/denehoffman/ganesh/commit/0810ef71126d7087576e98ed1afdabd9a13270ea))

## [0.26.0](https://github.com/denehoffman/ganesh/compare/v0.25.1...v0.26.0) (2026-04-03)


### ⚠ BREAKING CHANGES

* remove top-level rust helpers and split python init/config API
* split algorithm init from config
* add mixed Python package integration
* add sampled MCMC chain retention
* add rolling MCMC chain storage
* simplify MCMC walker storage
* standardize bounds handling modes
* validate sampler walkers and custom simplex shapes
* validate weighted MCMC move inputs
* **Status:** This strongly types status messages and removes convergence flags in favor of matching the message type

### Features

* Add CMA-ES optimizer ([9ad2122](https://github.com/denehoffman/ganesh/commit/9ad2122d724031281fc88d5f679b56317d94f886))
* Add convenience methods for gradient and gradient-free fits and MCMC sampling using reasonable default settings ([66e4561](https://github.com/denehoffman/ganesh/commit/66e456152b55c854a5a7e0f0565ab02a9172887a))
* Add debug derives for swarm types ([ccd37eb](https://github.com/denehoffman/ganesh/commit/ccd37eb02d28aaf44f5efd8878929665fa48f9f2))
* Add differential evolution optimizer ([166181e](https://github.com/denehoffman/ganesh/commit/166181ee35bc9d876bfdf99c0b8b06ce3b5657a6))
* Add generic error type and use Results rather than panics in algorithm configuration ([8dc1e6a](https://github.com/denehoffman/ganesh/commit/8dc1e6a6d0a6d946d81f395cc259beb7394cf86a))
* Add interval progress observer ([def4775](https://github.com/denehoffman/ganesh/commit/def477552d1e4523326e5bee27a40e9b14f767bc))
* Add MCMC diagnostics helpers ([3a145ed](https://github.com/denehoffman/ganesh/commit/3a145ede40be6e2b1b4a481195bc0b7385fe5a30))
* Add mixed Python package integration ([9050360](https://github.com/denehoffman/ganesh/commit/90503606962a3f47657247a63a04138e1a39adaf))
* Add multistart minimization orchestration ([7b4553e](https://github.com/denehoffman/ganesh/commit/7b4553ea3ad2afcd8ef97db824cc84a3ff085a58))
* Add nonlinear conjugate gradient optimizer ([550d493](https://github.com/denehoffman/ganesh/commit/550d493dff3724730e51a61b335c80dc768b469e))
* Add numpy-backed python conversions ([7da54cb](https://github.com/denehoffman/ganesh/commit/7da54cbeb9b32467c5444352c3908b86323799a8))
* Add python config schema exports ([eb92f4f](https://github.com/denehoffman/ganesh/commit/eb92f4fc46a70937fa6e5eca7499ee3d68dadca7))
* Add python feature scaffolding ([d072c43](https://github.com/denehoffman/ganesh/commit/d072c437450121d35d2618a4e215511b34fdbe50))
* Add python LBFGSB config wrapper ([756aaf3](https://github.com/denehoffman/ganesh/commit/756aaf322e0e95836cf68a7dbb999229ab845ca8))
* Add python mcmc summary dict export ([9ee8b6c](https://github.com/denehoffman/ganesh/commit/9ee8b6c188dcfc75099f9cf3ff1014087d42acc0))
* Add python mcmc summary wrapper ([d8790e0](https://github.com/denehoffman/ganesh/commit/d8790e05ac364a645685fc87003061b0e4078212))
* Add python minimization summary dict export ([b8bcabe](https://github.com/denehoffman/ganesh/commit/b8bcabe31b6b5ab1f6e7524138a2e5569f8d09b2))
* Add python minimization summary wrapper ([693af90](https://github.com/denehoffman/ganesh/commit/693af9099e00720ddd6329eeaf8223df2b1feb71))
* Add python multistart summary wrapper ([cb1ef76](https://github.com/denehoffman/ganesh/commit/cb1ef7681a1c036ab00d7cc04b331209f95eaffc))
* Add python run options wrappers ([beb34b4](https://github.com/denehoffman/ganesh/commit/beb34b4f45b7971b2c6dec58e01435e06fc65de8))
* Add python simulated annealing config wrapper ([e722b04](https://github.com/denehoffman/ganesh/commit/e722b04205e140aa94641b1a4107f42e9e521a6a))
* Add python simulated annealing summary exports ([d78232b](https://github.com/denehoffman/ganesh/commit/d78232bf8c2080825b9e13eca01b7a179f4d05da))
* Add python wrappers for cmaes and differential evolution configs ([128cebf](https://github.com/denehoffman/ganesh/commit/128cebf0a123c9cbc58620f40715f50ccf2c7906))
* Add python wrappers for core optimizer configs ([629c8b4](https://github.com/denehoffman/ganesh/commit/629c8b4a6d65ee1ce8e50dbf098cfa8bba0b2c90))
* Add recoverable numerical helper errors ([769da30](https://github.com/denehoffman/ganesh/commit/769da304045f87644a0c7bfb111691d6fd664613))
* Add rolling MCMC chain storage ([76f0dc3](https://github.com/denehoffman/ganesh/commit/76f0dc391dbc4ed7852260bcf73188631d48f380))
* Add sampled MCMC chain retention ([ac6d174](https://github.com/denehoffman/ganesh/commit/ac6d174909b1bd2443d47f50a5624480bf6731ae))
* Add scale transform helpers ([72dba5c](https://github.com/denehoffman/ganesh/commit/72dba5c20a5b5a7b9fe01e39c7a386d161e4eb5a))
* Add step-boundary checkpoint and resume support ([1bce7f2](https://github.com/denehoffman/ganesh/commit/1bce7f2ce0ba911a687a0962062d778bd935e24d))
* Add summary export helpers ([fe6a292](https://github.com/denehoffman/ganesh/commit/fe6a2927179852e268f0f9f964d2c811639fefc7))
* Add trust-region optimizer ([23b611f](https://github.com/denehoffman/ganesh/commit/23b611fe4f8385add6ad49fa951eeebdea22c384))
* Add typed python error mapping ([2fdd5b9](https://github.com/denehoffman/ganesh/commit/2fdd5b937c7bd598e133fa7199889f86f2798de5))
* Propagate parameter names from config ([54cc088](https://github.com/denehoffman/ganesh/commit/54cc0885e88ba03c7b821afd8046bd706f1ba022))
* Standardize bounds handling modes ([6f97c00](https://github.com/denehoffman/ganesh/commit/6f97c004acc0c02d3ef1d52c1f36db22e256d32b))
* **Status:** Replace status String with a StatusMessage type and remove convergence fields in favor of StatusMessage::success() ([a28a3a3](https://github.com/denehoffman/ganesh/commit/a28a3a33523d1146188f234260236d3dd77282c3))


### Bug Fixes

* **ci:** Restore release publishing workflows ([9e24122](https://github.com/denehoffman/ganesh/commit/9e241223f9cc6a160eb54bebe5b2e7d5525418e3))
* Ensure all Status-like structs are correctly and fully reset on reset() calls ([29ee51c](https://github.com/denehoffman/ganesh/commit/29ee51c0ecfa92d5782e33d40e9e827cd7341667))
* Faithfully record function evaluations in NM init, MCMC, and PSO init ([e14cf24](https://github.com/denehoffman/ganesh/commit/e14cf24d6d040e9cc6cd595a664bb59692ee26b4))
* Guard ESS adaptive scaling against zero updates ([0a13ad4](https://github.com/denehoffman/ganesh/commit/0a13ad4c047a0de97784f37ee0f8d9a4f0796edb))
* Harden bound containment and infinite sampling ([4aa3dd8](https://github.com/denehoffman/ganesh/commit/4aa3dd80d3647f14472136fff76de2b122243f14))
* Properly compute free indices in L-BFGS-B ([589cfde](https://github.com/denehoffman/ganesh/commit/589cfdeabc8bf38725a1f1265b05ec416bf89f8c))
* Scale volume by delta^n rather than delta^{n+1} in shrink steps for Nelder-Mead ([cb40999](https://github.com/denehoffman/ganesh/commit/cb40999d5a452812811ad603a0c913774aad950e))
* Use best neighbor rather than worst in PSO ring topology ([32e49dd](https://github.com/denehoffman/ganesh/commit/32e49ddb82f575ccc8390e142f275a6668490aaa))
* Use best simplex point instead of worst in Diameter and Higham x-terminator methods of Nelder-Mead ([34d9d19](https://github.com/denehoffman/ganesh/commit/34d9d199045f9b0935e3a3fa7754e13b5f6a6b68))
* Use correct random variable draw range in PSO ([0effd12](https://github.com/denehoffman/ganesh/commit/0effd1262b9e96ec7cd08aa0de726ddea2540cf0))
* Use correct update step in simulated annealing ([08201fe](https://github.com/denehoffman/ganesh/commit/08201fe3728af0dfaaa79b9a923e97d2745086bd))
* Use proper sign in stretch move proposal ([bcdd5b4](https://github.com/denehoffman/ganesh/commit/bcdd5b47c688b136d507eb7eecbb9050b2014186))
* Validate sampler walkers and custom simplex shapes ([d6b9243](https://github.com/denehoffman/ganesh/commit/d6b9243bbb3fffcb2a4d374ebdeb985aae81e2b1))
* Validate weighted MCMC move inputs ([6ca7561](https://github.com/denehoffman/ganesh/commit/6ca75616687db714012a9bad5387e029abadcf52))


### Performance Improvements

* Add lightweight benchmark matrix ([7929116](https://github.com/denehoffman/ganesh/commit/79291162fdb1dffce61f71ed37feaa4a847a44b5))
* Avoid walker clones in ensemble proposals ([ecdbfb3](https://github.com/denehoffman/ganesh/commit/ecdbfb3e468cb2222606f1b2691656bb700ee0fd))
* Cache resolved Nelder-Mead bounds state ([8da188e](https://github.com/denehoffman/ganesh/commit/8da188e48aa47d0b31f53ed150facbaecba4c87e))
* Reduce status formatting overhead in progress paths ([0462dd7](https://github.com/denehoffman/ganesh/commit/0462dd7889ecfb21e2773254e4830ac8602533a9))
* Simplify MCMC walker storage ([7a7cdba](https://github.com/denehoffman/ganesh/commit/7a7cdba9619f4fa74ad09016f3645c5bb8c71ffe))
* Use fused objective and derivative evaluations ([7bf83c1](https://github.com/denehoffman/ganesh/commit/7bf83c149a0e03a8e11b85577e4004ab942fb37b))


### Code Refactoring

* Remove top-level rust helpers and split python init/config API ([3f5959c](https://github.com/denehoffman/ganesh/commit/3f5959c459b7e44f330f1ab377213e411991e4b5))
* Split algorithm init from config ([2c82cf3](https://github.com/denehoffman/ganesh/commit/2c82cf32cc0b590458e9665cc22983f2caea7f22))

## [Unreleased]

## [0.25.1](https://github.com/denehoffman/ganesh/compare/v0.25.0...v0.25.1) - 2025-10-02

### Fixed

- Bounds::upper() was returning -inf instead of +inf for intervals without an upper bound
- Make sure the first Adam iteration acts on internal parameters

## [0.25.0](https://github.com/denehoffman/ganesh/compare/v0.24.0...v0.25.0) - 2025-09-26

### Other

- Development Updates ([#82](https://github.com/denehoffman/ganesh/pull/82))

## [0.24.0](https://github.com/denehoffman/ganesh/compare/v0.23.1...v0.24.0) - 2025-09-21

### Added

- add DynClone to callbacks to make Callbacks clone-able

## [0.23.1](https://github.com/denehoffman/ganesh/compare/v0.23.0...v0.23.1) - 2025-09-21

### Other

- update README and lib.rs docs with correct links

## [0.23.0](https://github.com/denehoffman/ganesh/compare/v0.22.0...v0.23.0) - 2025-09-21

### Added

- Major refactor of `ganesh` crate which includes large changes to the user-facing API
- Large performance improvements across main algorithms like Nelder-Mead and L-BFGS-B
- Unified interface for `Transform`s like bounds transformations
- Unified interface for `Terminator`s and `Observer`s
- `Engine`-less processing of all algorithms under singe trait interface
- Many other bugfixes and improvements

## [0.22.0](https://github.com/denehoffman/ganesh/compare/v0.21.1...v0.22.0) - 2025-04-11

### Added

- add `NopAbortSignal` to the main namespace
- Add AbortSignal to handle abortion of calculations
- Add AbortSignal to handle abortion of calculations
- Add AbortSignal to handle abortion of calculations

### Fixed

- add defaults to abort signals and make some `new` methods `const`

### Other

- Merge pull request #71 from estriv/main
- add abort signals to doctests

## [0.21.1](https://github.com/denehoffman/ganesh/compare/v0.21.0...v0.21.1) - 2025-04-11

### Added

- add Clone derive to TrackingSwarmObserver
- make the TrackingSwarmObserver fields pub

## [0.21.0](https://github.com/denehoffman/ganesh/compare/v0.20.1...v0.21.0) - 2025-04-11

### Added

- change the way swarm algorithms are initialized to simplify swarm construction

## [0.20.1](https://github.com/denehoffman/ganesh/compare/v0.20.0...v0.20.1) - 2025-04-10

### Added

- add Default impl for Particle

### Fixed

- update some pub visibility on Point and remove a python file committed by mistake

## [0.20.0](https://github.com/denehoffman/ganesh/compare/v0.19.0...v0.20.0) - 2025-04-10

### Added

- finalize PSO implementation and overhaul organization
- add Global step to ESS

### Fixed

- remove `kmeans` dependency and implement the algorithm by hand
- move vector generating utilities to root module

## [0.19.0](https://github.com/denehoffman/ganesh/compare/v0.18.0...v0.19.0) - 2025-04-02

### Other

- reverse a bit of the previous overhaul, keeping bounds in the initialization method and in the minimizer is a good idea, but they are not required elsewhere

## [0.18.0](https://github.com/denehoffman/ganesh/compare/v0.17.1...v0.18.0) - 2025-04-01

### Added

- reorganize, make Bounds a part of functions and individual algorithms

### Other

- remove some spaces
- move Observer traits to `observer` module

## [0.17.1](https://github.com/denehoffman/ganesh/compare/v0.17.0...v0.17.1) - 2025-04-01

### Added

- allow users to implement a tuning step at the function level

### Fixed

- correct behavior in observer doctest

## [0.17.0](https://github.com/denehoffman/ganesh/compare/v0.16.0...v0.17.0) - 2025-01-28

### Added

- allow users to skip Hessian calculation in L-BFGS-B algorithm

## [0.16.0](https://github.com/denehoffman/ganesh/compare/v0.15.2...v0.16.0) - 2025-01-04

### Other

- remove coverage tests for  feature, since it doesn't work all the time yet
- remove  from  and  since it requires  and  generics to implement , which is sometimes difficult. Also made  the default  method and removed

## [0.15.2](https://github.com/denehoffman/ganesh/compare/v0.15.1...v0.15.2) - 2024-12-20

### Other

- *(observers)* fix doctests

## [0.15.1](https://github.com/denehoffman/ganesh/compare/v0.15.0...v0.15.1) - 2024-12-20

### Fixed

- change signature of `with_observer` methods to be more flexible

## [0.15.0](https://github.com/denehoffman/ganesh/compare/v0.14.3...v0.15.0) - 2024-12-20

### Added

- Add autocorrelation `MCMCObserver`
- add `build` methods to observers to make `Arc<RwLock<Self>>`s out of them

### Fixed

- change `build` signature for standalone structs
- correct autocorrelation calculation

### Other

- add usage doctest for `AutocorrelationObserver`
- update example with IAT observer and proper burn-in, as well as an IAT plot
- remove methods to set all observers and change the signature of `with_observer` to accept an `Arc<RwLock<Observer>>`
- switch `Observer`/`MCMCObserver` callback notation to terminate if true is returned

## [0.14.3](https://github.com/denehoffman/ganesh/compare/v0.14.2...v0.14.3) - 2024-12-14

### Fixed

- make  custom constructors public

## [0.14.2](https://github.com/denehoffman/ganesh/compare/v0.14.1...v0.14.2) - 2024-12-14

### Added

- add serde to  and members

## [0.14.1](https://github.com/denehoffman/ganesh/compare/v0.14.0...v0.14.1) - 2024-12-14

### Fixed

- ensure `f32` feature works by making a few type-agnostic calls

## [0.14.0](https://github.com/denehoffman/ganesh/compare/v0.13.1...v0.14.0) - 2024-12-13

### Added

- add interface for setting Sokal window size
- add interface to update `ESS` hyperparameter settings
- finishing touches on `mcmc` module
- add integrated autocorrelation time
- add ctrl-c back to MCMC sampler
- add Multivariate normal ESS example
- add initial MCMC interface
- use `mul_add` where applicable
- add benchmarks for each algorithm
- move `Point` to be usable by other algorithms and correct the way Nelder-Mead functions are evaluated with bounds

### Other

- fix Just -> just
- remove trace plots
- update README.md and crate-level documnetation with MCMC section
- ignore .pkl files
- rename `Point::len` to `Point::dimension`
- rename i -> step
- add docstrings to sampler
- rename AIMES -> AIES
- get rid of generics

## [0.13.1](https://github.com/denehoffman/ganesh/compare/v0.13.0...v0.13.1) - 2024-11-08

### Added

- use unzip to support 1.69.0 as MSRV

### Fixed

- clippy lints

### Other

- add codspeed link to readme
- change indents

## [0.13.0](https://github.com/denehoffman/ganesh/compare/v0.12.2...v0.13.0) - 2024-11-05

### Added

- add optional parameter names field to `Status`
- add `serde` serialization and deserialization to `Status`

### Other

- switch to codspeed for benchmarks

## [0.12.2](https://github.com/denehoffman/ganesh/compare/v0.12.1...v0.12.2) - 2024-10-24

### Fixed

- change default tolerances to be a bit more realistic

## [0.12.1](https://github.com/denehoffman/ganesh/compare/v0.12.0...v0.12.1) - 2024-10-21

### Fixed

- store `Algorithm` in a `Box` in `Minimizer` and add `new_from_box` constructor
- add  trait bound to  constructor

### Other

- add test for  constructors

## [0.12.0](https://github.com/denehoffman/ganesh/compare/v0.11.3...v0.12.0) - 2024-10-21

### Added

- use `dyn-clone` to pass `Algorithm`s to `Minimizer`s by reference

### Other

- add criterion benchmark comparison for PRs
- update documentation to reflect the output of pretty-printing a `Status`

## [0.11.3](https://github.com/denehoffman/ganesh/compare/v0.11.2...v0.11.3) - 2024-10-19

### Added

- add a feature to make Ctrl-C gracefully stop the algorithm and return the fit result with a message

## [0.11.2](https://github.com/denehoffman/ganesh/compare/v0.11.1...v0.11.2) - 2024-10-17

### Added

- allow `Observer`s to modify fit `Status` and terminate minimization

## [0.11.1](https://github.com/denehoffman/ganesh/compare/v0.11.0...v0.11.1) - 2024-10-17

### Other

- relocate html docs header
- fix link in README
- Merge branch 'main' of https://github.com/denehoffman/ganesh
- correct default in docstring
- add comma

## [0.11.0](https://github.com/denehoffman/ganesh/compare/v0.10.0...v0.11.0) - 2024-09-12

### Added

- add `Display` to `Minimizer`
- add initial value and bounds to `Status` and improve `Display` output
- add (sketchy) function to check if a parameter is at its bounds

### Fixed

- add `UpperExp` trait bound to `Minimizer` `Display`
- make sure `Status` is reset on a new run of the same minimizer

### Other

- move all `Status` structs out of `Algorithm`s and into `Minimizer` and `Algorithm` method signatures

## [0.10.0](https://github.com/denehoffman/ganesh/compare/v0.9.1...v0.10.0) - 2024-09-10

### Added

- move Hessian inversion to `Status` and add `hess` field

### Fixed

- use cargo-llvm-cov (messed up git history on previous attempt)
- use correct internal/external bounded/regular calls in all algorithms
- change finite difference delta

### Other

- Merge pull request [#31](https://github.com/denehoffman/ganesh/pull/31) from denehoffman/hotfixes
- change implementation of Hessian to use gradients
- fix link

## [0.9.1](https://github.com/denehoffman/ganesh/compare/v0.9.0...v0.9.1) - 2024-09-09

### Fixed

- ensure all algorithms reset completely when their initialize method is called
- use set_cov to also calculate errors, change method to take an `Option`
- update BFGS and L-BFGS methods to be closer to the implementation for L-BFGS-B and fix errors in L-BFGS causing incorrect convergence

### Other

- update readme and main doc
- add basic convergence tests to all algorithms
- add leading signs to `Status` `Display` method
- improve pretty printing for `Status`

## [0.9.0](https://github.com/denehoffman/ganesh/compare/v0.8.5...v0.9.0) - 2024-09-09

### Added

- add errors to all algorithms
- add Hessian evaluation to `Function` trait
- kickstart BFGS with H-inverse scaling
- switch to using nalgebra data structures by default

### Fixed

- simplify L-BFGS algorithm and ensure the first few steps are computed correctly
- left and right matrices were switched by accident
- make terminator epsilon fields public for BFGS methods, set default f tolerance to epsilon rather than its cube root

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
