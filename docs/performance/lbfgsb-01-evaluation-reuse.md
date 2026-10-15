**Item 1 — reuse objective and gradient evaluations in L-BFGS-B and conjugate gradient**

Pre-commit review report, 2026-10-15. Base: `5e05d41f0bbb69d275e9fd49145a2053dece45d6` on `development`. No commit or push has been made. Proposed commit: `perf!: reuse objective and gradient evaluations in line searches`.

This patch implements item 1, including the shared conjugate-gradient caller requested during review. The next item after review and approval is item 3 (an unconstrained fast path). Item 2, final Hessian/covariance policy, is excluded from the implementation series at the user's request.

**Changes**

- Add the borrowed `LineSearchStart` value and make `LineSearch::search_from` the required implementation method. L-BFGS-B and conjugate gradient supply their already evaluated position, objective, and gradient.
- Make `LineSearch::search` a shared default: evaluate the starting point once, count the objective/gradient quantities, and delegate to `search_from`. More–Thuente, Hager–Zhang, backtracking, and strong-Wolfe dispatch implement only `search_from`; their duplicate wrappers are removed.
- Forward `evaluate_with_gradient` through `TransformedProblem`. A transformed evaluation computes external coordinates once and pulls the returned gradient back; an untransformed evaluation delegates directly to the original problem's fused override.
- Preserve the lower endpoint's objective in More–Thuente zoom and reuse a gradient already computed on the final zoom iteration.
- Add regression tests and three repeatable data-fitting benchmarks. The existing Rosenbrock benchmark cases are unchanged.

The line-search acceptance rules, interpolation, initial step, and maximum-step handling are unchanged. They belong to item 7. History matrices, factorizations, and scratch-storage behavior are also outside this patch.

**Interface migration requested in review**

This is a breaking change for custom line-search implementations: implement `search_from` using its supplied `x`, `fx`, and `gradient`, instead of implementing only `search`. Existing callers of `search` continue to work. Requiring the seeded method makes missing implementations a compile-time error and avoids mutually recursive defaults. The proposed commit is marked `perf!` accordingly.

L-BFGS-B and conjugate gradient both use the seeded entry point. Standalone callers inherit the shared unseeded wrapper. Conjugate gradient already stores the required position, objective, and gradient; its coefficient updates, direction restarts, convergence rules, and final Hessian estimation are unchanged. The shared `TransformedProblem` fused-evaluation forwarding also applies to other algorithms using that adapter.

**Before/after results, including the existing algorithm benchmarks**

For item 1, the original implementation and the immediately preceding implementation are the same commit. The before column is therefore both comparisons. A source archive of that commit was built independently, with the identical benchmark definitions copied into it.

| Workload | Original / before | After item 1 | Time change |
|---|---:|---:|---:|
| Existing L-BFGS-B Rosenbrock, 2 variables | 772.87 µs | 755.38 µs | -2.26% |
| Existing L-BFGS-B Rosenbrock, 8 variables | 908.93 µs | 914.49 µs | +0.61% |
| Existing conjugate-gradient Rosenbrock, 2 variables | 47.77 µs | 31.31 µs | -34.47% |
| Existing conjugate-gradient Rosenbrock, 8 variables | 67.41 µs | 43.90 µs | -34.88% |
| Least squares, separate, More–Thuente | 390.69 µs | 219.35 µs | -43.86% |
| Least squares, fused, More–Thuente | 390.47 µs | 213.09 µs | -45.43% |
| Least squares, fused, Hager–Zhang | 283.20 µs | 171.22 µs | -39.54% |

Times are Criterion point estimates (regression slope where available, otherwise mean). Percentages are computed from those estimates; negative means less elapsed time. Criterion's bootstrap mean comparison uses a different estimator and can differ slightly. The 8-variable L-BFGS-B benchmark's change is within Criterion's 1% practical noise threshold. The large gains here are in conjugate gradient and the data-fitting workloads; they should not be generalized to every objective.

95% confidence intervals for those time estimates:

| Workload | Before | After |
|---|---:|---:|
| Existing L-BFGS-B Rosenbrock, 2 variables | 772.14–773.60 µs | 754.38–756.55 µs |
| Existing L-BFGS-B Rosenbrock, 8 variables | 908.23–909.72 µs | 913.36–915.67 µs |
| Existing conjugate-gradient Rosenbrock, 2 variables | 47.55–48.11 µs | 31.28–31.34 µs |
| Existing conjugate-gradient Rosenbrock, 8 variables | 67.16–67.67 µs | 43.81–44.01 µs |
| Least squares, separate, More–Thuente | 390.34–391.12 µs | 218.60–220.67 µs |
| Least squares, fused, More–Thuente | 390.28–390.72 µs | 212.83–213.35 µs |
| Least squares, fused, Hager–Zhang | 283.06–283.38 µs | 170.99–171.48 µs |

**Incremental conjugate-gradient caller change**

The following comparison uses measurements saved immediately before conjugate gradient was switched from `search` to `search_from`. That intermediate version already had the fused-transform forwarding, shared default wrapper, and zoom caching. This separates the additional benefit of reusing conjugate gradient's stored starting evaluation from the full improvement over the original algorithm.

| Workload | Before cached caller | After | Time change |
|---|---:|---:|---:|
| Existing conjugate-gradient Rosenbrock, 2 variables | 32.83 µs | 31.31 µs | -4.65% |
| Existing conjugate-gradient Rosenbrock, 8 variables | 46.99 µs | 43.90 µs | -6.57% |

**Evaluation counts and result quality**

| Workload | Before objective / gradient requests | After | Final objective, identical before and after |
|---|---:|---:|---:|
| Least squares, separate, More–Thuente | 31 / 10 | 16 / 6 | 3.0321980540320027e-19 |
| Least squares, fused, More–Thuente | 31 / 10 | 16 / 6 | 3.0321980540320027e-19 |
| Least squares, fused, Hager–Zhang | 12 / 14 | 9 / 11 | 3.6413438684140782e-15 |

The 14 focused regression tests include one-step quadratic tests that verify exactly two objective and two gradient quantities (initial and accepted points), rather than three of each, for both algorithms and native f32/f64. Conjugate gradient still performs its usual final Hessian request. It also verifies that the fused override is actually invoked. Conjugate-gradient equivalence tests exercise all five coefficient update rules with untransformed, scaled, and nonlinear bounded coordinates. Other tests cover all built-in searches and dispatch variants, repeated and changing zoom endpoints, the zoom iteration limit, native f32, fused errors, scale and nonlinear bounds transforms, custom searches inheriting the shared default, and identical bounded/transformed Rosenbrock fits with cached and explicitly reevaluating searches.

**Method**

- AMD Ryzen 7 3700X, Linux 6.18.56 x86_64; rustc 1.101.0-nightly (822182db4 2026-10-13).
- Release profile, CPU affinity fixed to logical CPU 4, 80 samples, one-second warmup, five-second target measurement time per case. Criterion can extend collection to obtain all samples. No other build/test process ran during the final timings.
- Existing L-BFGS-B and conjugate-gradient benchmarks retain unbounded Rosenbrock at dimensions 2 and 8, 60 steps, and their default final Hessian estimation.
- Added benchmarks fit eight parameters to 4,096 deterministic least-squares rows. They use normal L-BFGS-B convergence callbacks, a 60-step cap, maximum step 1, and `LBFGSBErrorMode::Skip` to isolate iteration/evaluation costs. This is benchmark configuration only; production Hessian defaults are unchanged.
- Each new benchmark runs an untimed convergence assertion and prints its final objective and evaluation counts. Separate and fused variants use the same objective and analytic gradient; the fused implementation shares residual computation.
- Exploratory fixed-step results and runs affected by shared build-artifact reuse were excluded. The final comparison used separate build directories, verified original source contents, matching benchmark definitions, and independently checked evaluation counts.

The final raw Criterion estimates and samples are preserved locally under `target/lbfgsb-performance/item1/final/`, with a machine-readable `summary.json`. Named Criterion baselines `line-search-original` and `line-search-item1-after` are retained for both algorithms and the data-fitting cases; the original L-BFGS-B baseline aliases are also preserved. Each later report should compare its before/after results with the original baseline as requested.

**Reproduction**

Create a source archive of the base commit in `/tmp/ganesh-lbfgsb-item1-baseline` and copy the current `benches/benchmark_matrix.rs` into that archive. Use a different `CARGO_TARGET_DIR` for each source snapshot.

Run the before command from the archived source directory:

```sh
rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item1-before-target CRITERION_HOME=/home/dene/research/ganesh/target/criterion taskset -c 4 cargo bench --offline --bench benchmark_matrix -- 'algorithms/(optimization/(lbfgsb|conjugate-gradient)|lbfgsb-evaluations)' --save-baseline line-search-original --sample-size 80 --warm-up-time 1 --measurement-time 5 --noplot
```

Run the after command from the working repository:

```sh
rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item1-after-target CRITERION_HOME=/home/dene/research/ganesh/target/criterion taskset -c 4 cargo bench --offline --bench benchmark_matrix -- 'algorithms/(optimization/(lbfgsb|conjugate-gradient)|lbfgsb-evaluations)' --baseline line-search-original --sample-size 80 --warm-up-time 1 --measurement-time 5 --noplot
```

**Validation**

- `rtk cargo test --offline`: 113 tests passed.
- `rtk cargo test --offline --features python`: 119 tests passed.
- `rtk cargo check --offline --all-features --all-targets`: passed.
- `rtk cargo clippy --offline --all-targets -- -D warnings`: passed; Cargo still reports its existing redundant homepage metadata warning.
- `rtk cargo fmt -- --check` and `rtk git diff --check`: passed.
- An additional strict all-features Clippy run reports seven `clone_on_copy` diagnostics generated by existing PyO3 `from_py_object` annotations in unchanged Python binding files. Those unrelated diagnostics were left untouched.

The new line-search entry point requires its supplied value and gradient to match the supplied point, problem, and arguments. The existing standalone entry point remains available and evaluates its own starting values.
