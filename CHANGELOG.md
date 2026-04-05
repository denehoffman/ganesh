# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.27.0](https://github.com/denehoffman/ganesh/compare/v0.26.2...v0.27.0) (2026-04-05)


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

* Add (sketchy) function to check if a parameter is at its bounds ([a0d6eca](https://github.com/denehoffman/ganesh/commit/a0d6eca61e5c5a645a8747634764299f99155c03))
* Add `build` methods to observers to make `Arc<RwLock<Self>>`s out of them ([bd749d3](https://github.com/denehoffman/ganesh/commit/bd749d3567776246db4dd52caab510434f8339cc))
* Add `Debug`s to `NelderMead` ([2523379](https://github.com/denehoffman/ganesh/commit/2523379a6f41dc5d662a018f6a86158112ccb59e))
* Add `Display` to `Minimizer` ([b10929e](https://github.com/denehoffman/ganesh/commit/b10929e39b82defbb82afebbceb9ee59c15c0828))
* Add `LineSearch` trait and implementations of BFGS and L-BFGS algorithms ([73c107c](https://github.com/denehoffman/ganesh/commit/73c107ca78569a8751b465160728efefa01b70ec))
* Add `LogDensity` trait to differentiate MCMC functions from minimization functions ([2052224](https://github.com/denehoffman/ganesh/commit/20522249711466546700a90abac946cc0f44df1b))
* Add `n_f_evals` and `n_g_evals` to `EnsembleStatus` (currently unused) ([f1e24ae](https://github.com/denehoffman/ganesh/commit/f1e24aead7e06659f8a097199a094cb58859a1a1))
* Add `NopAbortSignal` to the main namespace ([e2867b3](https://github.com/denehoffman/ganesh/commit/e2867b3d17e5b1796866e2d34633d2d2c7e712e6))
* Add `process_default` method to `Algorithm` trait ([f304350](https://github.com/denehoffman/ganesh/commit/f304350839c6d016d4244960dcb689d8b4adbd43))
* Add `serde` serialization and deserialization to `Status` ([f084746](https://github.com/denehoffman/ganesh/commit/f08474602382695e6ff6f84a9983d441149e95c9))
* Add a `SimulatedAnnealingSummary` ([49178f8](https://github.com/denehoffman/ganesh/commit/49178f859486ab96702f64f2a5da8b32f5b3e814))
* Add a feature to make Ctrl-C gracefully stop the algorithm and return the fit result with a message ([ceb94a0](https://github.com/denehoffman/ganesh/commit/ceb94a0015c8e3d82a5fb70933033bdc36ce6782))
* Add a feature to make Ctrl-C gracefully stop the algorithm and return the fit result with a message ([60603c0](https://github.com/denehoffman/ganesh/commit/60603c055da07e97fd48f82bc493a24be0b09400))
* Add AbortSignal to handle abortion of calculations ([813fbc0](https://github.com/denehoffman/ganesh/commit/813fbc010238225b3c25562fe044ce955ad85e49))
* Add AbortSignal to handle abortion of calculations ([0c319d1](https://github.com/denehoffman/ganesh/commit/0c319d152855f05e5e27383a8fe4ec7fff2b735f))
* Add AbortSignal to handle abortion of calculations ([dc16061](https://github.com/denehoffman/ganesh/commit/dc16061a88892d9a4fd7a2bdc44fc569f0077e55))
* Add associated type to `Algorithm` to allow for custom structs for `Summary` ([531ea57](https://github.com/denehoffman/ganesh/commit/531ea578664e7c2e19d1b085a26b285deeb31b42))
* Add autocorrelation `MCMCObserver` ([61c604d](https://github.com/denehoffman/ganesh/commit/61c604d123027eca6707bd2fbec5a6b59ad528aa))
* Add benchmarks for each algorithm ([765cc93](https://github.com/denehoffman/ganesh/commit/765cc9304219b901f5eb943bc6f1d86712926717))
* Add bounded evaluation shortcuts to `Function` trait and count gradient evaluations in `Status` ([d6c3f12](https://github.com/denehoffman/ganesh/commit/d6c3f12d53d0ea92a004903c3edce98ce9e131ba))
* Add check to make sure starting position is within bounds ([53d498b](https://github.com/denehoffman/ganesh/commit/53d498bbda412b3aa9f1fff40bbc31fa530f36c4))
* Add Clone derive to TrackingSwarmObserver ([3aafa67](https://github.com/denehoffman/ganesh/commit/3aafa67a71777f87e9424fb446b6c61a4ae4bdf5))
* Add CMA-ES optimizer ([9ad2122](https://github.com/denehoffman/ganesh/commit/9ad2122d724031281fc88d5f679b56317d94f886))
* Add convenience methods for gradient and gradient-free fits and MCMC sampling using reasonable default settings ([66e4561](https://github.com/denehoffman/ganesh/commit/66e456152b55c854a5a7e0f0565ab02a9172887a))
* Add ctrl-c back to MCMC sampler ([a712f11](https://github.com/denehoffman/ganesh/commit/a712f11e3bbeee736e3a14811dd8bb75ff047e68))
* Add debug derives for swarm types ([ccd37eb](https://github.com/denehoffman/ganesh/commit/ccd37eb02d28aaf44f5efd8878929665fa48f9f2))
* Add Default impl for Particle ([9a3deb6](https://github.com/denehoffman/ganesh/commit/9a3deb624a476bd1e581540876449fafa37cd7c9))
* Add Default implementation for `SimulatedAnnealingConfig` ([fc38f71](https://github.com/denehoffman/ganesh/commit/fc38f71d3d5984d8860ad2834e687edb57da3832))
* Add differential evolution optimizer ([166181e](https://github.com/denehoffman/ganesh/commit/166181ee35bc9d876bfdf99c0b8b06ce3b5657a6))
* Add display method, methods for getting `lower` and `upper` bounds, and `contains` method for `Bounds` ([f0f6bc9](https://github.com/denehoffman/ganesh/commit/f0f6bc967399e9a4cba8826c0fa359f2011750e8))
* Add DynClone to callbacks to make Callbacks clone-able ([8655c03](https://github.com/denehoffman/ganesh/commit/8655c03f15997f9d1233a23f3b5fec63b5345c79))
* Add errors to all algorithms ([59aebc6](https://github.com/denehoffman/ganesh/commit/59aebc6fdec20a2f635eeb8e8d6deaebe9baad80))
* Add function value terminators for BFGS algorithms ([a8071e9](https://github.com/denehoffman/ganesh/commit/a8071e979e92c7ce6a60899fac6df33f2eb7c9f3))
* Add function value terminators for BFGS algorithms ([c7fe4b2](https://github.com/denehoffman/ganesh/commit/c7fe4b2e77b8adb138d1bc804051400de3b64b2c))
* Add generic error type and use Results rather than panics in algorithm configuration ([8dc1e6a](https://github.com/denehoffman/ganesh/commit/8dc1e6a6d0a6d946d81f395cc259beb7394cf86a))
* Add Global step to ESS ([c99d3e3](https://github.com/denehoffman/ganesh/commit/c99d3e366f536e2de37cdf8b41a440358502b872))
* Add gradient tolerance to L-BFGS-B ([36beebd](https://github.com/denehoffman/ganesh/commit/36beebd6ed09c2ec614a70d951f0c855dbad1e5d))
* Add Hager-Zhang line search ([ac9133b](https://github.com/denehoffman/ganesh/commit/ac9133bb46809f4d293700d96649f18c31ce479e))
* Add Hessian evaluation to `Function` trait ([3d278c5](https://github.com/denehoffman/ganesh/commit/3d278c5fed118577a816d388acac025a6855f28b))
* Add initial MCMC interface ([cff7554](https://github.com/denehoffman/ganesh/commit/cff7554c12d77aabf8868fbed140ec88d18cbb76))
* Add initial value and bounds to `Status` and improve `Display` output ([7711bff](https://github.com/denehoffman/ganesh/commit/7711bff2daada1cc267738ec41a2f12331471104))
* Add integrated autocorrelation time ([e9b052e](https://github.com/denehoffman/ganesh/commit/e9b052eb6c6a2d0d5532db3ddba14abdb600a276))
* Add interface for setting Sokal window size ([67afb7b](https://github.com/denehoffman/ganesh/commit/67afb7b1d2051ddccc01026ae1e48e425d04b728))
* Add interface to update `ESS` hyperparameter settings ([6ecbe3b](https://github.com/denehoffman/ganesh/commit/6ecbe3bf1fd7afb7d676475b8581ec20771a97e0))
* Add interval progress observer ([def4775](https://github.com/denehoffman/ganesh/commit/def477552d1e4523326e5bee27a40e9b14f767bc))
* Add L-BFGS-B algorithm ([9315687](https://github.com/denehoffman/ganesh/commit/9315687ca2e3f1126450c9ffb6511e0af8483aaa))
* Add MCMC diagnostics helpers ([3a145ed](https://github.com/denehoffman/ganesh/commit/3a145ede40be6e2b1b4a481195bc0b7385fe5a30))
* Add message to `EnsembleStatus` ([c3af16a](https://github.com/denehoffman/ganesh/commit/c3af16ae9a2c5747a39ab0ebec320047aa506429))
* Add messages to MCMC samplers to track the current move type ([16744f5](https://github.com/denehoffman/ganesh/commit/16744f5a1ac3d60a91d44546f8cd191250a207db))
* Add method to make `Bound` from `&Bound` ([5ed8f75](https://github.com/denehoffman/ganesh/commit/5ed8f75b12a0839dc076f9e479cde5696c61fc0b))
* Add method to return the gradient and inverse of Hessian matrix ([8c68c09](https://github.com/denehoffman/ganesh/commit/8c68c091214bd7ea5307a792888ebc5e4edd6d4c))
* Add mixed Python package integration ([9050360](https://github.com/denehoffman/ganesh/commit/90503606962a3f47657247a63a04138e1a39adaf))
* Add multistart minimization orchestration ([7b4553e](https://github.com/denehoffman/ganesh/commit/7b4553ea3ad2afcd8ef97db824cc84a3ff085a58))
* Add Multivariate normal ESS example ([48a4e08](https://github.com/denehoffman/ganesh/commit/48a4e08dcf68e32afe5a0dc552515a6b64e2a9d4))
* Add nonlinear conjugate gradient optimizer ([550d493](https://github.com/denehoffman/ganesh/commit/550d493dff3724730e51a61b335c80dc768b469e))
* Add numpy-backed python conversions ([7da54cb](https://github.com/denehoffman/ganesh/commit/7da54cbeb9b32467c5444352c3908b86323799a8))
* Add optional parameter names field to `Status` ([0a3125a](https://github.com/denehoffman/ganesh/commit/0a3125aed54f2a7f24db693d5f24c3a796821cb9))
* Add preliminary implementation of BFGS algorithm ([a42292f](https://github.com/denehoffman/ganesh/commit/a42292f2b2ada937719df8e6370a0ecf119c1931))
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
* Add Send + Sync to traits that support them ([836d1d5](https://github.com/denehoffman/ganesh/commit/836d1d583afc6e72c51a0abfad6a99033c1321c9))
* Add serde to  and members ([7742588](https://github.com/denehoffman/ganesh/commit/774258801ea21980027140d9aa9efe9f7e5f856d))
* Add simultaneous setters for hyperparameters whose bounds depend on each other ([610b433](https://github.com/denehoffman/ganesh/commit/610b4332709f2a55529e042c26ebdaebda75c58b))
* Add step-boundary checkpoint and resume support ([1bce7f2](https://github.com/denehoffman/ganesh/commit/1bce7f2ce0ba911a687a0962062d778bd935e24d))
* Add summary export helpers ([fe6a292](https://github.com/denehoffman/ganesh/commit/fe6a2927179852e268f0f9f964d2c811639fefc7))
* Add the Conjugate Gradient algorithm ([1197da0](https://github.com/denehoffman/ganesh/commit/1197da035521c53e06706bb7840bfecce20e4ad5))
* Add trust-region optimizer ([23b611f](https://github.com/denehoffman/ganesh/commit/23b611fe4f8385add6ad49fa951eeebdea22c384))
* Add typed python error mapping ([2fdd5b9](https://github.com/denehoffman/ganesh/commit/2fdd5b937c7bd598e133fa7199889f86f2798de5))
* Add useful assert warning for trying to construct a `NelderMead` `Simplex` with fewer than 2 points ([5856282](https://github.com/denehoffman/ganesh/commit/5856282f27d2aa331538f990d38541a5d46f218c))
* Add warnings which can be disabled by the user ([e65feed](https://github.com/denehoffman/ganesh/commit/e65feedaacd2cf8c78391d0ec88a6786d229345d))
* Allow `Observer`s to modify fit `Status` and terminate minimization ([f62b7c4](https://github.com/denehoffman/ganesh/commit/f62b7c4bd3ef58a462c5dcad9d274968520cdf9b))
* Allow `Observer`s to modify fit `Status` and terminate minimization ([a1c4d0e](https://github.com/denehoffman/ganesh/commit/a1c4d0e5c141faf55fd29e6ee8a5486035d17a41))
* Allow users to implement a tuning step at the function level ([6c71beb](https://github.com/denehoffman/ganesh/commit/6c71bebad9daf730f94ae5d31daa03775e8f3097))
* Allow users to implement a tuning step at the function level ([98a0c4e](https://github.com/denehoffman/ganesh/commit/98a0c4ef678e8574cd06689049303317ccda9974))
* Allow users to skip Hessian calculation in L-BFGS-B algorithm ([09650b3](https://github.com/denehoffman/ganesh/commit/09650b34030f3208be004ea76b675308fbbcfc28))
* Change most slice types to `nalgebra::DVector`s to make algorithms more ergonomic, add `LineSearch` algorithms ([bdfbc30](https://github.com/denehoffman/ganesh/commit/bdfbc304257a630100e09dc1f6f5bfe8cbe4a270))
* Change the way swarm algorithms are initialized to simplify swarm construction ([e39ee77](https://github.com/denehoffman/ganesh/commit/e39ee771f05510f01e6175082ef7f1b095e175e9))
* Clean up how `Bounds` are implemented ([6a8ace5](https://github.com/denehoffman/ganesh/commit/6a8ace5f67adc7da909b91ca639425325af887a6))
* Decouple configs from algorithms and provide `new` methods to avoid invalid state ([7272e54](https://github.com/denehoffman/ganesh/commit/7272e54bd93cec5773e980b60e5147f6e8f16977))
* Ensure any message added to a `Status` by an `Observer` during termination is not overwritten" ([083f90a](https://github.com/denehoffman/ganesh/commit/083f90ad3cd7a2d0f54d796f065d219a88a508be))
* Ensure Some(Float::INFINITY) and Some(Float::NEG_INFINITY) yield unbounded  intervals ([db313e3](https://github.com/denehoffman/ganesh/commit/db313e3724c767fd6a666e5055c607ed12f816d6))
* Finalize PSO implementation and overhaul organization ([6e5c0c7](https://github.com/denehoffman/ganesh/commit/6e5c0c70079e5f588f6711ef5947434eba4991a2))
* Finishing touches on `mcmc` module ([ee96357](https://github.com/denehoffman/ganesh/commit/ee963579606bf7f421bfffe14e2241622a559827))
* Formalize internal configuration structs in each algorithm. This also moves boundaries and starting positions into those config structs. ([217846f](https://github.com/denehoffman/ganesh/commit/217846f98029cb3de007e1ffdc5fbadde8cb520e))
* Implement Adam algorithm ([b9adf69](https://github.com/denehoffman/ganesh/commit/b9adf69a9d1c8040f690207e0346d04d030db852))
* Kickstart BFGS with H-inverse scaling ([9498786](https://github.com/denehoffman/ganesh/commit/94987868c47ad826e99059cad47c994eff71762c))
* Make `MinimizationSummary` fields `DVectors` and add `n_f_evals` to PSO ([f3aac1f](https://github.com/denehoffman/ganesh/commit/f3aac1fb4ca48ba18a2580377e48b3e0c847536e))
* Make the TrackingSwarmObserver fields pub ([957505a](https://github.com/denehoffman/ganesh/commit/957505a17abb17e009cb3e6f8973bc96ecbe605f))
* Merge unified API into main for release ([c1e6b9b](https://github.com/denehoffman/ganesh/commit/c1e6b9b5a48c328a2efd23dd35ff0c305e55ca80))
* Move `AbortSignal` impls to the module for the trait ([06354a7](https://github.com/denehoffman/ganesh/commit/06354a7973993c6f0d628176cf3b59014fe18808))
* Move `Point` to be usable by other algorithms and correct the way Nelder-Mead functions are evaluated with bounds ([91fdc57](https://github.com/denehoffman/ganesh/commit/91fdc5773f64111a63f6caf9da128cf8ea7a3cde))
* Move Hessian inversion to `Status` and add `hess` field ([d82d1fa](https://github.com/denehoffman/ganesh/commit/d82d1fad78442e1ed9ec6ab591e2d686a508946f))
* Move trait implementations and `utils` into `core`, finalize docs, and fix clippy lints and tests ([263aa8f](https://github.com/denehoffman/ganesh/commit/263aa8f26288c8df6d6c8dd6ab33a83e26933f82))
* Port `AutocorrelationObserver` from legacy code ([d926f5c](https://github.com/denehoffman/ganesh/commit/d926f5ce803c6d3c86ed4102fa3cd14963fa2d67))
* Port `TrackingSwarmObserver` from legacy code ([a5ed8b7](https://github.com/denehoffman/ganesh/commit/a5ed8b7586753cd55afa81fda54dd1cc0b8af16c))
* Preliminary step of moving MCMC methods to unified API ([9a271e5](https://github.com/denehoffman/ganesh/commit/9a271e5a5b8270b3c7c58b52f8262da86e551b49))
* Propagate parameter names from config ([54cc088](https://github.com/denehoffman/ganesh/commit/54cc0885e88ba03c7b821afd8046bd706f1ba022))
* **python:** Add status wrappers and summary conversions ([fc2b855](https://github.com/denehoffman/ganesh/commit/fc2b8559de8361f69b091ab8cd26ca2dd6a7d198))
* **python:** Publish type stubs ([8df49b6](https://github.com/denehoffman/ganesh/commit/8df49b6e91303f3fa9a711e59b67e89eb808f7b9))
* Reboot L-BFGS-B on invalid line searches ([a0a6018](https://github.com/denehoffman/ganesh/commit/a0a60186a6ca5367a6fac2c2533a3eb50837bd11))
* Reduces the `Field` trait to use `num` traits rather than `nalgebra`'s `RealField` ([da3d513](https://github.com/denehoffman/ganesh/commit/da3d513466def564fd01245f21a5846a733904eb))
* Reorganize, make Bounds a part of functions and individual algorithms ([18eea4e](https://github.com/denehoffman/ganesh/commit/18eea4e67f2ee28df83b94b2583fe7bcb2942a68))
* Standardize bounds handling modes ([6f97c00](https://github.com/denehoffman/ganesh/commit/6f97c004acc0c02d3ef1d52c1f36db22e256d32b))
* **Status:** Replace status String with a StatusMessage type and remove convergence fields in favor of StatusMessage::success() ([a28a3a3](https://github.com/denehoffman/ganesh/commit/a28a3a33523d1146188f234260236d3dd77282c3))
* Switch to `uv` for all examples and add inline script requirements ([8e5b238](https://github.com/denehoffman/ganesh/commit/8e5b2389979f1f6cfb34f33590fdf7004dd0ccb5))
* Switch to using nalgebra data structures by default ([d838a72](https://github.com/denehoffman/ganesh/commit/d838a7239a84cfcec1de6a19ef52fdd8d892d567))
* Update `NelderMead` to count gradient evals and use bounded interface ([536d8aa](https://github.com/denehoffman/ganesh/commit/536d8aafc9ad09e9a5d278283223e41e5e730e13))
* Update line search to take a `max_step` optional argument and return a bool flag of validity rather than an `Option` ([56a0296](https://github.com/denehoffman/ganesh/commit/56a02965748a25c04432441dd2720618a6534a79))
* Update with_parameter_names signature ([580e7d4](https://github.com/denehoffman/ganesh/commit/580e7d42f53b7e66662684f1f58e292d8421845e))
* Use `dyn-clone` to pass `Algorithm`s to `Minimizer`s by reference ([1820e91](https://github.com/denehoffman/ganesh/commit/1820e91b122d36b682975f921f8ab0ba26d0d7fc))
* Use `mul_add` where applicable ([9e89331](https://github.com/denehoffman/ganesh/commit/9e893314cdc2558637c5b3cc5b33b101d1b0270a))
* Use unzip to support 1.69.0 as MSRV ([9c4e5d1](https://github.com/denehoffman/ganesh/commit/9c4e5d14d59a92e40b3ba1c1014dc8dd5b02d1b9))


### Bug Fixes

* `p` is `-grad_f` so this was right all along ([04cd04e](https://github.com/denehoffman/ganesh/commit/04cd04e6c2bfeebfff391630205c773b38b73de8))
* A few aesthetic corrections and use of expect rather than unwrap ([9db371f](https://github.com/denehoffman/ganesh/commit/9db371f30ef335a9a0da2de77e6cf4638709e3f8))
* Actually remove legacy code files ([59c6bfb](https://github.com/denehoffman/ganesh/commit/59c6bfba0486db2f8fae1090be54a89f6634d5a9))
* Adaptive Nelder-Mead now requires inputting the dimension ([fcd71bb](https://github.com/denehoffman/ganesh/commit/fcd71bb4d2525bb523d0dd2eebcb903df0a50425))
* Add  trait bound to  constructor ([7302020](https://github.com/denehoffman/ganesh/commit/7302020ac54fc7fa18995a5b47ee031f70d5eb12))
* Add `Bounds` back into line searches ([824c679](https://github.com/denehoffman/ganesh/commit/824c6791f1c7d3abdbe464ebd51183e789e312c4))
* Add `config` argument back in to `process_default` ([9b6840d](https://github.com/denehoffman/ganesh/commit/9b6840d5b5e12655b0091787a76dcc40087fb37d))
* Add `UpperExp` trait bound to `Minimizer` `Display` ([37d1cf4](https://github.com/denehoffman/ganesh/commit/37d1cf4e3ca2f4035ea24a87f7b832ba8b384d36))
* Add a `reset` method to `Algorithm` to make sure any intermediate results are cleared when an algorithm is restarted ([fa75a7a](https://github.com/denehoffman/ganesh/commit/fa75a7a964bdd6ac72d9e043b08f0d71a96154ef))
* Add defaults to abort signals and make some `new` methods `const` ([81101d8](https://github.com/denehoffman/ganesh/commit/81101d87cb5d5e20094163e94795174c3a392b34))
* Allow expect in Hessian inverse function ([0ea41b8](https://github.com/denehoffman/ganesh/commit/0ea41b8cf48dd14c10d302a32877070f056d917c))
* Bounds::upper() was returning -inf instead of +inf for intervals without an upper bound ([e5bca10](https://github.com/denehoffman/ganesh/commit/e5bca1005824dffc34a528d67d503c54f0615a8b))
* Change `build` signature for standalone structs ([4bcc9d6](https://github.com/denehoffman/ganesh/commit/4bcc9d69c680782a5919cb0255ecc1e305ea64f1))
* Change `learning_rate` to an `Option` in gradient descent ([17977f8](https://github.com/denehoffman/ganesh/commit/17977f80ad001e457c37b7a07754e960b07f733f))
* Change default tolerances to be a bit more realistic ([f9bc385](https://github.com/denehoffman/ganesh/commit/f9bc3856c8813a10b2fde271f4b9a1263176a978))
* Change finite difference delta ([08ecb1b](https://github.com/denehoffman/ganesh/commit/08ecb1b40229ef0994bf3ec0d2c372995819fd9a))
* Change finite difference delta ([2a379a5](https://github.com/denehoffman/ganesh/commit/2a379a54e7f1588c6e876ac9fbc4f2a8ac90c8b3))
* Change signature of `with_observer` methods to be more flexible ([3bb2897](https://github.com/denehoffman/ganesh/commit/3bb2897c8c69e48372c823a9e1c6c80e87b2fbed))
* Change to inequality to ensure a proper status message if the max iterations are passed ([85f747a](https://github.com/denehoffman/ganesh/commit/85f747a8e9d3b857d1d8b9ef2d6b03062d8b551e))
* **ci:** Checkout repo before cargo publish ([d07137d](https://github.com/denehoffman/ganesh/commit/d07137d81a584f69161dfdb64c29cc3e4b4b56fd))
* **ci:** Restore release publishing workflows ([9e24122](https://github.com/denehoffman/ganesh/commit/9e241223f9cc6a160eb54bebe5b2e7d5525418e3))
* Clippy lints ([7149595](https://github.com/denehoffman/ganesh/commit/7149595ec68bdf1a4aa86bd9c521400243ebf5db))
* Clippy lints ([cf78369](https://github.com/denehoffman/ganesh/commit/cf783690266824032f1df6a4d66b56f02bbb78f3))
* Correct autocorrelation calculation ([e20a29f](https://github.com/denehoffman/ganesh/commit/e20a29f85d0cd4e9fc4cd97d9361d4ced86de99a))
* Correct behavior in observer doctest ([91a3303](https://github.com/denehoffman/ganesh/commit/91a33033f83e163be4fa2b2e05eb11c4be98d72c))
* Correct ensemble mean and add more ESS tests ([23216d3](https://github.com/denehoffman/ganesh/commit/23216d306ba3efa1235f47e6766b4de86c835c8a))
* Correct import path for NelderMead solver ([c3d3f15](https://github.com/denehoffman/ganesh/commit/c3d3f150803d32c536b77ebccb0dd8eac15dfa39))
* Correct links in docs and apply some clippy lints to L-BFGS-B ([bdb4d3c](https://github.com/denehoffman/ganesh/commit/bdb4d3c19e0f3452c25102ab78f493e2a8396176))
* Ensure `f32` feature works by making a few type-agnostic calls ([8e8df4c](https://github.com/denehoffman/ganesh/commit/8e8df4cd71ae13a3e20a079f162d19b58e4c2db5))
* Ensure all algorithms reset completely when their initialize method is called ([5596101](https://github.com/denehoffman/ganesh/commit/5596101a2119d29fc597678d2142d0d8c67eafb2))
* Ensure all methods use the `Field` trait rather than just `Float` for better compatibility ([40771d0](https://github.com/denehoffman/ganesh/commit/40771d00943a77fa1e461d1e2bf36d4b43c267f9))
* Ensure all Status-like structs are correctly and fully reset on reset() calls ([29ee51c](https://github.com/denehoffman/ganesh/commit/29ee51c0ecfa92d5782e33d40e9e827cd7341667))
* Ensure sufficient decrease is met before marking line search as valid ([1f30dcf](https://github.com/denehoffman/ganesh/commit/1f30dcfa7da701464a5bc32acd759fa5d65ce635))
* Faithfully record function evaluations in NM init, MCMC, and PSO init ([e14cf24](https://github.com/denehoffman/ganesh/commit/e14cf24d6d040e9cc6cd595a664bb59692ee26b4))
* Follow strong Wolfe condition a bit more carefully ([51e1082](https://github.com/denehoffman/ganesh/commit/51e1082680d028aab3edad7351a392520730ded0))
* Get rid of the log in the multimodal example ([099ddeb](https://github.com/denehoffman/ganesh/commit/099ddebf71dc95292d7c05a745f622add5ea6a60))
* Guard against free_indices being empty ([db7b4ec](https://github.com/denehoffman/ganesh/commit/db7b4ecfa4afb8e98d6406ea982c4d606f7cec1d))
* Guard ESS adaptive scaling against zero updates ([0a13ad4](https://github.com/denehoffman/ganesh/commit/0a13ad4c047a0de97784f37ee0f8d9a4f0796edb))
* Harden bound containment and infinite sampling ([4aa3dd8](https://github.com/denehoffman/ganesh/commit/4aa3dd80d3647f14472136fff76de2b122243f14))
* Ignore large files in Cargo.toml packaging ([4b19dbc](https://github.com/denehoffman/ganesh/commit/4b19dbca57e4f3141cf8bd6f180bfc067adc036d))
* Ignore some more files to hopefully get the crate size down ([e097a34](https://github.com/denehoffman/ganesh/commit/e097a34a34b52389f52e0d0a211eb028df478916))
* Improve readability in the method for equation 6.1 by using x_minus_g directly ([e96a96d](https://github.com/denehoffman/ganesh/commit/e96a96d6f62cb4a31ea4901b17e7fc975611c39b))
* Left and right matrices were switched by accident ([e836d37](https://github.com/denehoffman/ganesh/commit/e836d3768473a875af3fd77f4842a2b9ad86a9cc))
* Make  custom constructors public ([1dd415e](https://github.com/denehoffman/ganesh/commit/1dd415e0f59d0e21b280eafb8b3a405fe846a580))
* Make `g_eval` increment gradient evaluations rather than function evaluations ([4123bb4](https://github.com/denehoffman/ganesh/commit/4123bb41834c5f3d8f2e310343d99f6064ed30e4))
* Make bounds inclusive ([1166f06](https://github.com/denehoffman/ganesh/commit/1166f06cd9019dd559e33b95625b244fd544cdae))
* Make sure `Status` is reset on a new run of the same minimizer ([d9e5efa](https://github.com/denehoffman/ganesh/commit/d9e5efae365b490b70c89b73956d6041f6c3986c))
* Make sure the first Adam iteration acts on internal parameters ([47dec2b](https://github.com/denehoffman/ganesh/commit/47dec2b6abd1a78ca647accc7af833a222364a8b))
* Make terminator epsilon fields public for BFGS methods, set default f tolerance to epsilon rather than its cube root ([ff66656](https://github.com/denehoffman/ganesh/commit/ff6665682f2aaeaf57e6405fe9b646928996d6bb))
* Move vector generating utilities to root module ([5a2c376](https://github.com/denehoffman/ganesh/commit/5a2c3761db2ea8ace4f49f8d0d1af47eeeadce37))
* Properly compute free indices in L-BFGS-B ([589cfde](https://github.com/denehoffman/ganesh/commit/589cfdeabc8bf38725a1f1265b05ec416bf89f8c))
* **python:** Configure ty for dynamic package exports ([1e1ab9e](https://github.com/denehoffman/ganesh/commit/1e1ab9e99816ac4eb86c0bfb60dc177940710de9))
* **python:** Stabilize ty checks for dynamic exports ([0810ef7](https://github.com/denehoffman/ganesh/commit/0810ef71126d7087576e98ed1afdabd9a13270ea))
* **python:** Validate extracted objects structurally ([7c1363f](https://github.com/denehoffman/ganesh/commit/7c1363fc663983785a8e180f1c1a61acc5ffbd16))
* Re-export `nalgebra::DVector` ([670dd6b](https://github.com/denehoffman/ganesh/commit/670dd6b09d168c6bd85808f9d136fd5c07e20b66))
* Remove `Default` impl of `Callbacks` to encourage use of `const` `empty` constructor ([b2a6b1b](https://github.com/denehoffman/ganesh/commit/b2a6b1bffabf7ae6dcf21004bcf7bb9811f545d7))
* Remove `kmeans` dependency and implement the algorithm by hand ([c44e7b0](https://github.com/denehoffman/ganesh/commit/c44e7b09d94159a8488e1dedcb91d887228edae4))
* Remove `mut`s in tests, benchmarks, and examples ([de98eed](https://github.com/denehoffman/ganesh/commit/de98eedde16c06f2c37254b2f98e79fa3394037b))
* Remove binary reference in Cargo.toml ([552733b](https://github.com/denehoffman/ganesh/commit/552733b42e9448ae2319b9c47864d1d519aa1a19))
* Remove out-of-bounds issue ([00a173d](https://github.com/denehoffman/ganesh/commit/00a173dcaa8a210c4055b3b9964abd4cfda18b75))
* Remove tracking `main.rs`, which I use for quick demos ([077b54a](https://github.com/denehoffman/ganesh/commit/077b54a7f11b83cf126664393b5dd323070ec68e))
* Remove unnecessary (and unlogged) function evaluation in `LBFGSB` which checked termination conditions ([68c5fec](https://github.com/denehoffman/ganesh/commit/68c5fec6b7dce1999445289ca8d0c9ca4de5531a))
* Remove unused parentheses ([074c531](https://github.com/denehoffman/ganesh/commit/074c53159dc13dba20711dd3a8555417103dc110))
* Rename Adam test ([ba84213](https://github.com/denehoffman/ganesh/commit/ba8421371adce8f8497b653a339c99497085705a))
* Scale volume by delta^n rather than delta^{n+1} in shrink steps for Nelder-Mead ([cb40999](https://github.com/denehoffman/ganesh/commit/cb40999d5a452812811ad603a0c913774aad950e))
* Simplify L-BFGS algorithm and ensure the first few steps are computed correctly ([3b01c3c](https://github.com/denehoffman/ganesh/commit/3b01c3cdb31483f665683446617bc3440ca6b2b1))
* Simplify logic by removing internal `m` ([b0b73c9](https://github.com/denehoffman/ganesh/commit/b0b73c93708a29f04f808e557f2567ee02e2ed15))
* Step direction should be opposite the gradient ([1c39dac](https://github.com/denehoffman/ganesh/commit/1c39dacbbd278d7dcdd44227dd7402c9ee9bdffd))
* Store `Algorithm` in a `Box` in `Minimizer` and add `new_from_box` constructor ([152cf88](https://github.com/denehoffman/ganesh/commit/152cf88992ea10fd63b97200216bf831e0f39189))
* Switch sign on function termination condition ([0db29dd](https://github.com/denehoffman/ganesh/commit/0db29dda4e9b8a9d93231b3509cc9bc0a05a230b))
* Unify signatures in Swarm builder methods ([20c601e](https://github.com/denehoffman/ganesh/commit/20c601ebd36c4bb1adfb3295cad3989fc77454fd))
* Update BFGS algorithm to recent changes with ganesh ([a480270](https://github.com/denehoffman/ganesh/commit/a4802705823739a634df2aeb9f164926fd138646))
* Update BFGS and L-BFGS methods to be closer to the implementation for L-BFGS-B and fix errors in L-BFGS causing incorrect convergence ([fc34d3a](https://github.com/denehoffman/ganesh/commit/fc34d3af2a1c8e3c4e954bbcbafe3eb1a37fd48c))
* Update some pub visibility on Point and remove a python file committed by mistake ([31899b0](https://github.com/denehoffman/ganesh/commit/31899b08acf0053b14b40ea8d8ed188a03cd4014))
* Use  trait to implement ordering on float-like generics ([cfd8451](https://github.com/denehoffman/ganesh/commit/cfd8451f09a4f2a56297b7c53c4d1cb582fc19d7))
* Use absolute value for absolute tolerance ([93cc66e](https://github.com/denehoffman/ganesh/commit/93cc66e180eb244e1a9fec49de5dce8e9135f37a))
* Use best neighbor rather than worst in PSO ring topology ([32e49dd](https://github.com/denehoffman/ganesh/commit/32e49ddb82f575ccc8390e142f275a6668490aaa))
* Use best simplex point instead of worst in Diameter and Higham x-terminator methods of Nelder-Mead ([34d9d19](https://github.com/denehoffman/ganesh/commit/34d9d199045f9b0935e3a3fa7754e13b5f6a6b68))
* Use cargo-llvm-cov (messed up git history on previous attempt) ([4439601](https://github.com/denehoffman/ganesh/commit/44396015b6771b056c3101de178df93f3ee5236b))
* Use correct internal/external bounded/regular calls in all algorithms ([365ab05](https://github.com/denehoffman/ganesh/commit/365ab057db82e680c50a994c1882d6c4529bbdba))
* Use correct random variable draw range in PSO ([0effd12](https://github.com/denehoffman/ganesh/commit/0effd1262b9e96ec7cd08aa0de726ddea2540cf0))
* Use correct update step in simulated annealing ([08201fe](https://github.com/denehoffman/ganesh/commit/08201fe3728af0dfaaa79b9a923e97d2745086bd))
* Use proper sign in stretch move proposal ([bcdd5b4](https://github.com/denehoffman/ganesh/commit/bcdd5b47c688b136d507eb7eecbb9050b2014186))
* Use set_cov to also calculate errors, change method to take an `Option` ([c75448f](https://github.com/denehoffman/ganesh/commit/c75448f2a2f3f623db0db3149f252b3b972ce7cf))
* Validate sampler walkers and custom simplex shapes ([d6b9243](https://github.com/denehoffman/ganesh/commit/d6b9243bbb3fffcb2a4d374ebdeb985aae81e2b1))
* Validate weighted MCMC move inputs ([6ca7561](https://github.com/denehoffman/ganesh/commit/6ca75616687db714012a9bad5387e029abadcf52))


### Performance Improvements

* Add lightweight benchmark matrix ([7929116](https://github.com/denehoffman/ganesh/commit/79291162fdb1dffce61f71ed37feaa4a847a44b5))
* Add profiling build profile and ensure non-criterion benches are not built ([8363603](https://github.com/denehoffman/ganesh/commit/83636035fa6442ce9f953aa6b0a2566bc6f8c8d9))
* Avoid walker clones in ensemble proposals ([ecdbfb3](https://github.com/denehoffman/ganesh/commit/ecdbfb3e468cb2222606f1b2691656bb700ee0fd))
* Cache resolved Nelder-Mead bounds state ([8da188e](https://github.com/denehoffman/ganesh/commit/8da188e48aa47d0b31f53ed150facbaecba4c87e))
* Improve L-BFGS-B performance by trading a matrix inverse for a solve ([c9ed1ba](https://github.com/denehoffman/ganesh/commit/c9ed1ba69c78613293dc61bd88a49c3cbbf526a5))
* Improve Nelder-Mead performance by tracking centroid updates rather than recalculating ([af42049](https://github.com/denehoffman/ganesh/commit/af420497a2cdf6c995322efa6d69221bebc8e648))
* Reduce status formatting overhead in progress paths ([0462dd7](https://github.com/denehoffman/ganesh/commit/0462dd7889ecfb21e2773254e4830ac8602533a9))
* Remove potential dx clone ([651acc5](https://github.com/denehoffman/ganesh/commit/651acc516cacad1736ee77c416989dae31190ac2))
* Replace M inverse method by storing LU decomposition and applying solve where needed ([b8bcf71](https://github.com/denehoffman/ganesh/commit/b8bcf71ac951ffa90f9beae364a5d3919dbecdc3))
* Simplify MCMC walker storage ([7a7cdba](https://github.com/denehoffman/ganesh/commit/7a7cdba9619f4fa74ad09016f3645c5bb8c71ffe))
* Stop constructing z matrix and just do the math manually ([ab03a0a](https://github.com/denehoffman/ganesh/commit/ab03a0a229ed7a2bc007d93dbeea04ccd058fb53))
* Stop reallocating w_mat every step ([929fc1f](https://github.com/denehoffman/ganesh/commit/929fc1f6802e39dd81188fdbcb1f0f674b6d7eb6))
* Use fused objective and derivative evaluations ([7bf83c1](https://github.com/denehoffman/ganesh/commit/7bf83c149a0e03a8e11b85577e4004ab942fb37b))


### Code Refactoring

* Remove top-level rust helpers and split python init/config API ([3f5959c](https://github.com/denehoffman/ganesh/commit/3f5959c459b7e44f330f1ab377213e411991e4b5))
* Split algorithm init from config ([2c82cf3](https://github.com/denehoffman/ganesh/commit/2c82cf32cc0b590458e9665cc22983f2caea7f22))

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
