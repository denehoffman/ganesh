**L-BFGS-B item 7 — line-search step handling**

Pre-commit review report, 2026-10-15. Branch: `development`; base: approved item 5, `f145b560c831550759a301ab6e7b786d137ba2a6`. Original comparison: `5e05d41f0bbb69d275e9fd49145a2053dece45d6`. Item 6 was rejected and its production changes and review document removed. The L-BFGS-B implementation itself matches the base commit byte for byte. This item is uncommitted. Proposed commit: `fix!: correct More-Thuente line-search step handling`.

**Changes and evidence**

The default Moré–Thuente-style search started at one even when the supplied maximum was smaller. Its initial Armijo test omitted α. Expansion from one jumped 80% of the remaining distance to the ceiling, so a default ceiling of 1e8 produced a trial near 8e7. Exhausting zoom could report success without satisfying curvature, and exhausting bracketing returned an advanced α together with the preceding point's value and gradient.

Three focused tests failed before implementation: `rtk cargo test --offline --test line_search_evaluations more_thuente_`. The cap test observed an evaluation beyond its bound; the expansion test counted 21 objective calls where eight suffice; the zero-budget zoom test reported success despite unmet curvature. The log is retained with the measurements.

Trials now start at `min(1, max_step)` and expand geometrically by a factor of four, capped at the supplied maximum. With no maximum, expansion is permitted above one. Both Armijo checks use `f(α) ≤ f(0) + c1 α g(0)ᵀd`; the rising-value bracket check also applies on the second trial. Zoom stops when its budget is exhausted or its midpoint cannot advance. Failure returns a value and gradient belonging to the returned α. Invalid caps and non-descent directions return the cached starting result without new evaluations; nonfinite trial values cause bracketing rather than acceptance.

Interior accepted steps must satisfy strong-Wolfe curvature as well as Armijo. At an explicit upper limit, the search may also accept a strictly decreasing Armijo step whose directional derivative is still negative. A nearby feasible bound can prevent curvature from becoming small enough; repeatedly testing that endpoint cannot solve the problem. This exception is documented on the public search type and covered by an end-to-end bounded fit. It is not applied to an exhausted interior zoom. The normal strong-Wolfe requirements and the role of a maximum step are described in [SciPy's line-search documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html); the bounded-endpoint acceptance rule here is stated explicitly as part of this implementation's behavior.

The deterministic small quadratic now meets the eight-evaluation limit. The nearby-bound fit reaches projected-gradient convergence after one accepted step and two total f/g requests, while all recorded evaluations remain feasible. A failure-budget test verifies that α, f and g refer to the same point. The previously added endpoint-reuse test now expects failure when its zero zoom budget leaves curvature unmet; its exact evaluation counts remain unchanged.

These repairs are in the shared default line search, so conjugate gradient also uses them. Existing tests cover its five update rules, transforms, f32 behavior and seeded-evaluation equivalence. Hager–Zhang and backtracking implementation code was not changed. No public fields, signatures, checkpoint format, history-size policy, factorization-cache implementation, or final Hessian/covariance policy changed. A caller that wants an explicit unit ceiling can pass `Some(1.0)`. The proposed commit is marked as breaking for these acceptance semantics: `None` no longer imposes the previous unit ceiling, and success at a hard ceiling may mean the documented Armijo endpoint exception rather than strong-Wolfe curvature. Callers needing strict curvature must check it for a returned endpoint or supply their own search.

**Before/after and original comparison**

Times are microseconds. Point estimates use Criterion's regression slope where available and mean otherwise. Negative percentages mean less elapsed time. Final before/after cases were run consecutively, alternating order across cases, with 80 samples, three-second warmups and five-second collection targets.

| Workload | Original | Before: item 5 | After: item 7 | Change vs item 5 | Change vs original |
|---|---:|---:|---:|---:|---:|
| Existing Rosenbrock, 2 variables | 772.87 | 39.37 | 32.98 | -16.24% | -95.73% |
| Existing Rosenbrock, 8 variables | 908.93 | 53.94 | 51.76 | -4.05% | -94.31% |
| Least squares, separate, More–Thuente | 390.69 | 229.40 | 232.48 | +1.34% | -40.50% |
| Least squares, fused, More–Thuente | 390.47 | 224.91 | 225.84 | +0.41% | -42.16% |
| Least squares, fused, Hager–Zhang | 283.20 | 179.18 | 184.89 | +3.19% | -34.71% |
| Unbounded diagonal quadratic, 64 variables | 948.02 | 66.35 | 68.81 | +3.71% | -92.74% |
| Unbounded diagonal quadratic, 512 variables | 4778.00 | 437.98 | 450.82 | +2.93% | -90.56% |
| Explicitly unbounded limits, 64 variables | 955.80 | 79.70 | 78.38 | -1.66% | -91.80% |
| Native bounded control, 64 variables | 791.95 | 682.43 | 690.37 | +1.16% | -12.83% |
| Native bounded control, 512 variables | 4556.30 | 4054.29 | 4274.67 | +5.44% | -6.18% |
| Active bounds: 8 of 512 | 3945.37 | 3655.59 | 3474.54 | -4.95% | -11.93% |
| Active bounds: 384 of 512 | 3227.87 | 2582.72 | 2486.64 | -3.72% | -22.96% |
| Small-scale quadratic, 64 variables | 962.62 | 64.49 | 50.85 | -21.15% | -94.72% |
| Small-scale quadratic, 512 variables | 4840.73 | 393.43 | 349.35 | -11.20% | -92.78% |

95% confidence intervals, in microseconds:

| Workload | Before | After |
|---|---:|---:|
| Existing Rosenbrock, 2 variables | 39.03–39.73 | 32.74–33.21 |
| Existing Rosenbrock, 8 variables | 53.49–54.39 | 51.41–52.10 |
| Least squares, separate, More–Thuente | 227.10–231.99 | 229.61–236.20 |
| Least squares, fused, More–Thuente | 222.38–228.10 | 223.20–229.03 |
| Least squares, fused, Hager–Zhang | 178.13–180.22 | 183.16–186.69 |
| Unbounded diagonal quadratic, 64 variables | 65.81–67.06 | 68.27–69.42 |
| Unbounded diagonal quadratic, 512 variables | 428.69–450.98 | 447.35–454.40 |
| Explicitly unbounded limits, 64 variables | 79.29–80.11 | 77.96–78.84 |
| Native bounded control, 64 variables | 679.64–685.18 | 685.21–695.47 |
| Native bounded control, 512 variables | 4008.38–4110.78 | 4206.92–4346.51 |
| Active bounds: 8 of 512 | 3615.29–3699.32 | 3439.07–3510.87 |
| Active bounds: 384 of 512 | 2555.59–2613.41 | 2471.80–2502.50 |
| Small-scale quadratic, 64 variables | 63.76–65.39 | 50.33–51.36 |
| Small-scale quadratic, 512 variables | 390.45–396.29 | 344.28–354.26 |

The small-scale complete fits are 21.15% faster at 64 variables and 11.20% faster at 512 variables. Both objective and gradient request counts fall at the same stopping tolerance. The fixed-step Rosenbrock cases improve by 16.24% and 4.05%, with the convergence caveat below.

There is a repeatable performance cost in the unchanged 512-variable native-bounded control: +5.44% in the full run and +3.71% / +4.70% in longer-warmup, reversed-order repeats using a baseline built in the same checkout. The unbounded 64-variable repeat costs +1.80% / +1.23%. These costs are retained as adverse results. Their remaining cause has not been isolated; identical final values and request counts do not establish that the timing penalty is harmless. This item repairs correctness bugs and helps fits that need step expansion, but it is not a uniform performance improvement. The final decision should weigh these fixes and targeted gains against the bounded-control cost.

The existing ten workloads and their stopping criteria are unchanged. The two active-bound cases from the rejected experiment are included as line-search coverage, with no subspace cache code retained. They fit a 512-variable diagonal quadratic with either 8 or 384 positive lower-bound optima, verify projected-gradient termination at 1e-6, feasibility and objective gap below 1e-9, and retain MaxSteps(500) only as a safeguard. Their definitions match the original-version comparison measurements.

Two new complete-fit cases exercise expansion above one. They multiply the same diagonal quadratic by 1e-3, preserving its condition number and minimum while changing objective units. They use the normal default maximum step of 1e8, start at all ones, skip final error estimation, and stop at gradient norm below 1e-8. Untimed preflights require the normal gradient-convergence reason, objective below 1e-12 and position norm below 1e-5. All versions use identical definitions and tolerances. The new cases are measured on the original source as well as before and after.

**Accuracy and evaluation work**

All ten pre-existing complete-fit preflights (least squares, ordinary quadratic paths and active-bound fits) have exactly the same objective, gradient/projected norm and f/g requests before and after. The changed trajectories are shown below:

| Workload | Before objective | After objective | Before gradient norm | After gradient norm | Before f / g | After f / g |
|---|---:|---:|---:|---:|---:|---:|
| rosenbrock-fixed/2 | 0.00000e+00 | 0.00000e+00 | 0.00000e+00 | 0.00000e+00 | 82 / 66 | 70 / 54 |
| rosenbrock-fixed/8 | 6.61061e-11 | 4.95084e-12 | 2.54490e-04 | 9.56070e-05 | 112 / 66 | 83 / 65 |
| small-scale/64 | 4.90789e-15 | 1.80629e-14 | 7.16329e-09 | 9.43135e-09 | 62 / 40 | 35 / 33 |
| small-scale/512 | 2.91408e-15 | 1.42652e-14 | 7.89203e-09 | 7.92018e-09 | 65 / 42 | 39 / 37 |

The final small-scale cases make fewer objective and gradient requests, and both versions meet the same gradient tolerance. The benchmark initially used twofold expansion; that accepted shorter steps and increased gradient/optimizer work enough to offset its fewer objective calls. Fourfold expansion reduced both types of requests in these cases. The initial attempt and its unfavorable timings are retained separately.

The historical Rosenbrock benchmarks run exactly 60 steps and retain final error estimation. The 2-variable fit reaches zero in both versions; rejecting a zero/non-descent direction avoids unnecessary evaluations in its remaining fixed-step iterations. Its timing improvement should not be read as the same speedup for a fit stopped immediately at convergence. The 8-variable fit improves its final objective and gradient norm with fewer function calls at the same 60-step budget. The complete-fit comparisons above are kept distinct from those fixed-step results.

**Experiments and adverse controls**

The first candidate used twofold expansion. It reduced objective requests on the small-scale cases from 62/65 to 46/47, but gradient requests rose from 40/42 to 42/43. Its complete-fit timings were +4.35% / −0.51%, so fewer objective calls alone did not establish a useful speedup. Fourfold expansion changed the accepted early steps and reduced both objective and gradient work; all correctness checks and the full timing suite were rerun on that final source. The initial timings are preserved here:

| Initial twofold candidate | Before µs | After µs | Change |
|---|---:|---:|---:|
| Existing Rosenbrock, 2 variables | 40.50 | 33.51 | -17.26% |
| Existing Rosenbrock, 8 variables | 54.47 | 52.42 | -3.76% |
| Least squares, separate, More–Thuente | 233.86 | 232.31 | -0.66% |
| Least squares, fused, More–Thuente | 226.02 | 219.00 | -3.11% |
| Least squares, fused, Hager–Zhang | 184.03 | 181.14 | -1.57% |
| Unbounded diagonal quadratic, 64 variables | 67.79 | 67.77 | -0.03% |
| Unbounded diagonal quadratic, 512 variables | 440.88 | 433.91 | -1.58% |
| Explicitly unbounded limits, 64 variables | 78.52 | 75.80 | -3.46% |
| Native bounded control, 64 variables | 674.92 | 689.00 | +2.09% |
| Native bounded control, 512 variables | 4060.85 | 4125.37 | +1.59% |
| Active bounds: 8 of 512 | 3612.06 | 3505.40 | -2.95% |
| Active bounds: 384 of 512 | 2548.00 | 2712.14 | +6.44% |
| Small-scale quadratic, 64 variables | 64.42 | 67.22 | +4.35% |
| Small-scale quadratic, 512 variables | 405.35 | 403.29 | -0.51% |

Dependency locks, manifests, compiler version and environment flags match across builds. Symbol inspection found that both the main search and zoom were already out of line; no inlining attributes were added. To test for a build-location effect, the approved implementation was also built temporarily at the working checkout's path, into a separate target directory, then the validated item 7 source was restored byte for byte. Its complete preflight output matches the archive baseline exactly.

Comparing those identical-source builds on the dense-active case gave root-versus-archive differences of −2.86% and +0.07% with reversed order. The same root binary varied from 2525.15 to 2654.48 µs across runs. This did not establish a consistent path penalty, and it shows why small timing differences require care. It does not erase the unfavorable repeated measurements below.

The two adverse default-search controls were then repeated with both versions built in the same checkout, 80 samples, five-second warmups and five-second collection targets:

| Control | Execution order | Before µs | After µs | Change |
|---|---|---:|---:|---:|
| unconstrained/64 | before → after | 68.36 | 69.59 | +1.80% |
| unconstrained/64 | after → before | 71.11 | 71.98 | +1.23% |
| bounded-control/512 | before → after | 4090.85 | 4242.43 | +3.71% |
| bounded-control/512 | after → before | 3928.73 | 4113.44 | +4.70% |

The full comparison table remains the original complete-suite run; these repeats are supplementary and are not substituted to select more favorable figures. All raw runs and source hashes are retained.

**Validation**

- Default suite: 138 tests passed.
- Python-feature suite: 144 tests passed.
- Ndarray suite with OpenBLAS: 145 tests passed, including 117 library tests, 24 integration tests and four doctests.
- All-features/all-targets compile check: passed.
- Default all-targets clippy with warnings denied: passed.
- Formatting and whitespace checks: passed.

The generic quadratic checks cover f32/f64 and both providers, three Wolfe-constant pairs, five direction scales and five limits including no maximum. They independently check trial bounds, f/g consistency, Armijo, and either strong curvature or the documented decreasing boundary exception. Existing seeded-start and checkpoint tests continue to pass. There is no new diagnostic instrumentation or inlining attribute in production code.

The first ndarray doctest run failed to find OpenBLAS because rustdoc did not inherit the native library search path. Setting its own native search path and rpath resolved this; all four doctests then passed. No source change was needed for linking. Cargo's existing redundant-homepage warning remains. Correctness covers both providers; timings use nalgebra.

**Method and reproduction**

AMD Ryzen 7 3700X; Linux 6.18.56 x86_64; `rustc 1.101.0-nightly (822182db4 2026-10-13)`. Each source version has a separate Cargo target directory. Binaries were pinned to logical CPU 4, and no builds/tests ran during timings. Source and binary hashes, validation logs, preflight data and raw samples are retained under `target/lbfgsb-performance/item7/final/`. The shared final benchmark SHA-256 is `44a48da6f73aa7cdb832e33a97ab2c299251136a7c00a2bf23032c3558425138`.

Before and original snapshots are archives of item 5 and the original commit at `/tmp/ganesh-lbfgsb-item6-before` and `/tmp/ganesh-lbfgsb-item6-original`; their names come from their earlier creation, and both now contain the identical final benchmark definitions. The after build uses `/tmp/ganesh-lbfgsb-item7-after-target`. The source-hash record verifies that `src/algorithms/gradient/lbfgsb.rs` is unchanged from item 5.

```sh
rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item7-after-target cargo bench --offline --bench benchmark_matrix --no-run
rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item7-after-target CRITERION_HOME=/home/dene/research/ganesh/target/criterion taskset -c 4 cargo bench --offline --bench benchmark_matrix -- 'algorithms/(optimization/lbfgsb|lbfgsb-evaluations|lbfgsb-paths|lbfgsb-active|lbfgsb-steps)' --save-baseline item7-repeat-after --sample-size 80 --warm-up-time 3 --measurement-time 5 --noplot
```

The final paired driver invokes the prebuilt binaries one case at a time to keep before and after adjacent. The script and metadata are saved beside `summary.json`. Original estimates for the prior workloads are retained from earlier reports; the two new small-scale original measurements were collected during this item. Historical original timings use the earlier warmup protocol and are subject to between-run scheduling/frequency variation. Confidence intervals describe each run's sampling uncertainty and do not include that broader variation. The named `line-search-item7-after` baseline is retained for subsequent work.

Ndarray runtime checks use the OpenBLAS flags documented in item 4. For doctests, `RUSTDOCFLAGS` uses `-L native=/nix/store/rbcl4dl0g5c3b5irzjls0r9nz7xlakr1-openblas-0.3.33/lib -C link-arg=-Wl,-rpath,/nix/store/rbcl4dl0g5c3b5irzjls0r9nz7xlakr1-openblas-0.3.33/lib`; `RUSTFLAGS` additionally uses `-l openblas`.
