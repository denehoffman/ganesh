**L-BFGS-B — Cauchy-point curvature correction**

Pre-commit review report, 2026-10-16. Branch: `development`; base: approved breakpoint-ordering item 8, `c178caa5aaee1c57b1d9123bdb753b38f0d5f1b7`. Original-main comparison: `5e05d41f0bbb69d275e9fd49145a2053dece45d6`. This fix is uncommitted. Proposed commit: `fix: correct L-BFGS-B Cauchy-point updates`.

**Problem and correction**

The reported curvature error is real. A full fix also needs to handle simultaneous breakpoints and a path that has no moving coordinates. These are three related failures in the same generalized-Cauchy-point calculation; none changes the public API, checkpoint format, line-search policy, compact factorization cache, history policy, deferred ordering, or final error-estimation policy.

1. **Correct the compact curvature terms.** The model Hessian is B=theta*I-W*M*Wᵀ, with p=Wᵀd. When coordinate i reaches a bound, removing d_i=-g_i changes p to p+g_i*w_i. Expanding the quadratic gives a curvature change of -theta*g_i² - 2*g_i*w_iᵀMp - g_i²*w_iᵀMw_i. The implementation added the last two terms instead of subtracting them. Both signs are now corrected, using p before its update.
2. **Process equal-time bounds together.** If the next breakpoint has zero distance, there is no interval in which to test a minimum. The loop now processes it even when the intermediate direction has a positive slope. Otherwise it can stop halfway through a set of simultaneous bounds, although the actual projected path continues downhill after the whole set is fixed. Only exact zero-length intervals are grouped; no tolerance merges nearby distinct times.
3. **Stop when the path is stationary.** If no coordinates move, there is no next minimum to solve for. Previously 0/0 could produce NaN and contaminate a zero-gradient coordinate. A local count of moving coordinates now makes the remaining step zero. It is initialized during the existing coordinate pass and updated in constant time at each breakpoint; no full-vector scan is added inside the breakpoint loop.

**Reproductions and diagnosis**

The two-variable regression has x=[0,0], g=[1,1], theta=5/3 and s=[1,1], y=[1,2]. Its dense BFGS Hessian is [[7/6,-1/6],[-1/6,13/6]]. With x0 bounded below by -.1 and x1 unbounded, the path minimum is [-.1,-61/130], approximately [-.1,-.4692307692]. The committed calculation returns [-.1,-1.7]. The corrected calculation matches the analytic point.

A sign-only correction passes that example but fails the broader seven-pair fixture. Direct instrumentation separated three hypotheses: incorrect recurrence terms, disagreement between compact/dense models, and stopping between simultaneous bounds. The compact/dense Hessian-vector products agree to 1.12e-14. After the sign correction, the recursively maintained slope and curvature match direct calculations, but the search stops after the first of several equal-time bounds. This identifies the remaining failure as breakpoint handling, not the factorization or reference model.

The simultaneous-bound case reduces to three variables with B=diag(1,10,1), g=[1,1,1], and lower bounds -.2 on coordinates 0 and 1. This model comes from two valid correction pairs, s0=[0,1,0], y0=[0,10,0] and s1=[0,0,1], y1=[0,0,1], with theta=1. Both bounds are reached at t=.2. Fixing only coordinate 0 gives an intermediate slope of +.2, but after both bounds the right-hand slope is -.8. The true next minimum is [-.2,-.2,-1]; the old loop with signs corrected stops at [-.2,-.2,-.2]. Processing the whole zero-length group fixes both this minimal case and the original seven-pair fixture.

A second two-variable edge case has gradient [1,0], no history, and a lower bound -.1 on coordinate 0. After fixing that coordinate, the old calculation returns [-.1,NaN] instead of [-.1,0]. A fully zero gradient also needs to leave the starting point unchanged. The moving-coordinate count covers both cases without a curvature division.

An end-to-end test uses f(x)=(x0-3)²+x1², starts at [1,0], and bounds x0 above by 1.1 while x1 is unbounded. Running the newly added test against the committed Cauchy-point function fails at its five-step safeguard. With the fix it reaches [1.1,0], f=3.61 and normal projected-gradient convergence in one step with two f/g requests. The zero-gradient coordinate remains finite and unchanged.

The red-capable commands are `rtk cargo test --offline cauchy` and `rtk cargo test --offline lbfgsb_converges_with_a_stationary`. The red logs record the original failures, the sign-only probe, the separately minimized tied-bound failure, the stationary-path failure and the end-to-end failure on committed production code. Temporary instrumentation used the tag `[DEBUG-curvature]` and has been removed. The final tests keep the analytic regressions and restore all previously excluded dense-model cases; tolerances are unchanged.

**Before/after and original-main benchmarks**

All 17 existing benchmark definitions and tolerances are unchanged. Times are microseconds; negative percentages mean less elapsed time. Each point uses 80 samples, a three-second warmup and five-second collection target on logical CPU 4. Before/after pairs run consecutively, alternating order across cases; original main is measured after each pair. All three versions use identical benchmark source. Builds and tests finish before timing starts. Estimates use Criterion's regression slope when available, otherwise its mean.

| Workload | Original main | Before: item 8 | After: fix | Change vs item 8 | Change vs original |
|---|---:|---:|---:|---:|---:|
| Existing Rosenbrock, 2 variables | 760.33 | 30.11 | 30.19 | +0.24% | -96.03% |
| Existing Rosenbrock, 8 variables | 913.87 | 48.43 | 47.35 | -2.23% | -94.82% |
| Least squares, separate, More–Thuente | 393.02 | 209.59 | 211.39 | +0.86% | -46.21% |
| Least squares, fused, More–Thuente | 391.47 | 203.46 | 203.29 | -0.08% | -48.07% |
| Least squares, fused, Hager–Zhang | 283.16 | 165.51 | 165.54 | +0.02% | -41.54% |
| Unbounded diagonal quadratic, 64 variables | 950.62 | 63.72 | 63.51 | -0.34% | -93.32% |
| Unbounded diagonal quadratic, 512 variables | 4788.74 | 403.10 | 401.86 | -0.31% | -91.61% |
| Explicitly unbounded limits, 64 variables | 961.26 | 70.08 | 70.17 | +0.13% | -92.70% |
| Native bounded control, 64 variables | 796.06 | 565.42 | 573.69 | +1.46% | -27.93% |
| Native bounded control, 512 variables | 4469.81 | 3238.69 | 3390.31 | +4.68% | -24.15% |
| Active bounds: 8 of 512 | 3917.53 | 2903.46 | 2732.08 | -5.90% | -30.26% |
| Active bounds: 384 of 512 | 3252.37 | 2198.16 | 2151.23 | -2.13% | -33.86% |
| Small-scale quadratic, 64 variables | 845.78 | 45.13 | 45.66 | +1.18% | -94.60% |
| Small-scale quadratic, 512 variables | 4085.43 | 306.58 | 305.60 | -0.32% | -92.52% |
| Interior native bounds, 512 variables | 4261.39 | 2312.61 | 2321.03 | +0.36% | -45.53% |
| Interior native bounds, 4096 variables | 102815.10 | 88393.25 | 89087.99 | +0.79% | -13.35% |
| All-active native bounds, 4096 variables | Did not converge | 74689.67 | 72238.02 | -3.28% | — |

95% confidence intervals, in microseconds:

| Workload | Before | After |
|---|---:|---:|
| Existing Rosenbrock, 2 variables | 30.00–30.27 | 30.15–30.22 |
| Existing Rosenbrock, 8 variables | 48.39–48.47 | 47.29–47.43 |
| Least squares, separate, More–Thuente | 209.07–210.45 | 210.54–212.66 |
| Least squares, fused, More–Thuente | 203.35–203.58 | 203.14–203.47 |
| Least squares, fused, Hager–Zhang | 165.37–165.69 | 165.41–165.70 |
| Unbounded diagonal quadratic, 64 variables | 63.69–63.76 | 63.44–63.57 |
| Unbounded diagonal quadratic, 512 variables | 402.71–403.53 | 401.33–402.52 |
| Explicitly unbounded limits, 64 variables | 70.01–70.16 | 70.03–70.31 |
| Native bounded control, 64 variables | 564.97–565.90 | 567.17–582.65 |
| Native bounded control, 512 variables | 3231.03–3247.55 | 3350.68–3438.92 |
| Active bounds: 8 of 512 | 2893.01–2918.72 | 2724.98–2739.98 |
| Active bounds: 384 of 512 | 2184.81–2219.77 | 2146.17–2156.98 |
| Small-scale quadratic, 64 variables | 45.10–45.17 | 45.55–45.82 |
| Small-scale quadratic, 512 variables | 306.13–307.14 | 304.93–306.32 |
| Interior native bounds, 512 variables | 2309.10–2316.82 | 2317.78–2324.89 |
| Interior native bounds, 4096 variables | 88168.75–88629.65 | 88340.84–90002.45 |
| All-active native bounds, 4096 variables | 74430.23–74965.88 | 71936.41–72559.46 |

I recommend this correction because the old Cauchy calculation demonstrably returns the wrong projected minimum and can stall an otherwise trivial bounded fit. Performance is mixed, and the fix should not be presented as a uniform optimization.

In the primary matrix, the sparse active-bound fit is 5.90% faster, the dense active-bound fit 2.13% faster, and the all-active case 3.28% faster. The ordinary bounded controls cost 1.46% at 64 variables and 4.68% at 512; the native-interior cases cost 0.36% / 0.79%. These unchanged-work cases retain identical outputs and f/g counts. The added bookkeeping and altered generated code may affect overhead, but the remaining cause of each timing difference has not been isolated.

The repeated ordinary bounded 512-variable control is also slower (+2.62%), so its cost is treated as a reproducible adverse result rather than dismissed as noise. Paths that bypass the corrected calculation range from -2.23% to +1.18% in the primary matrix.

Reversed-order repeat pairs used 80 samples, five-second warmups and five-second collection targets:

| Repeat workload | Before (us) | After (us) | Change |
|---|---:|---:|---:|
| Native bounded control, 512 variables | 3259.52 | 3344.77 | +2.62% |
| Active bounds: 384 of 512 | 2234.62 | 2161.80 | -3.26% |
| All-active native bounds, 4096 variables | 75047.95 | 72441.47 | -3.47% |

Repeat order is before/after for the bounded control and dense active fit, after/before for the all-active fit: the reverse of the corresponding main pairs. They confirm the bounded-control cost and active-bound gains. Reduced callback counts alone do not imply an equal proportional speedup in the whole fit.


**Fit quality and evaluation work**

Fourteen preflight results are identical before and after: the fixed-step Rosenbrock cases, least squares, unconstrained/explicit-unbounded fits, bounded controls, unbounded small-scale fits and native-interior fits. Only the two active-bound cases and the all-active case change numerical work. All complete-fit quality and feasibility checks pass at their existing tolerances.

| Changed fit | Before: f/g | After: f/g | Before projected norm | After projected norm | Before / after objective gap |
|---|---:|---:|---:|---:|---:|
| Active bounds: 8 of 512 | 43 / 36 | 40 / 34 | 1.70707e-7 | 5.79696e-7 | 2.20712e-13 / 2.20512e-12 |
| Active bounds: 384 of 512 | 44 / 40 | 44 / 41 | 8.74292e-7 | 9.42225e-7 | 1.13687e-13 / 9.09495e-13 |
| All-active, 4096 | 12 / 12 | 5 / 5 | 0 | 0 | 1.45661e-13 / 1.45661e-13 |

The two 512-variable active fits retain the projected-gradient tolerance of 1e-6 and objective-gap check below 1e-9. The new trajectories meet the same requested quality, with different achieved precision; they are not claimed to return bit-identical results. The all-active case retains its 1e-8 projected tolerance and 1e-10 objective-gap check. Its final objective remains 30.201015968426596 and its solution is the exact vector of lower bounds.

The Rosenbrock timings remain fixed at 60 steps, as in the existing suite, and must not be interpreted as complete-fit convergence speedups. Original main still fails the all-active 4096-variable preflight at 500 steps with f=33.792 and 103001 / 2001 f/g requests. Its timing is skipped rather than compared with a completed fit. All other original comparisons meet their existing checks. Full precision objectives, norms, counts and all confidence intervals are retained in the raw artifacts.

**Validation and scope**

- Default suite: 144 passed, including doctests.
- Python feature suite: 150 passed, including doctests.
- Full ndarray suite: 152 passed, including doctests, using the local OpenBLAS configuration.
- `rtk cargo check --offline --all-features --all-targets`: passed.
- `rtk cargo clippy --offline --all-targets -- -D warnings`: passed.
- Formatting and whitespace checks: passed.

The independent dense reference explicitly builds the BFGS Hessian and minimizes its quadratic along each projected segment. It now covers histories of 0, 1, 3 and 7 pairs at five breakpoint scales for f32/f64 and both nalgebra/ndarray backends, including bound crossings, simultaneous times, fixed/one-sided/unbounded coordinates and zero gradient components. It checks the Cauchy point, c=Wᵀ(xcp-x), and the free set. Existing full-fit, rolling-history and checkpoint/resume tests also pass.

This is a correctness correction that changes bounded optimization trajectories. No changes from the rejected restricted-subspace cache are present. Scratch-allocation work (item 9) has not started; this remains one reviewable fix and will not be committed until approved.

**Reproduction and artifacts**

Base commit: `c178caa5aaee1c57b1d9123bdb753b38f0d5f1b7`; original: `5e05d41f0bbb69d275e9fd49145a2053dece45d6`. CPU: AMD Ryzen 7 3700X; compiler: rustc 1.101.0-nightly (822182db4 2026-10-13). Cargo.lock, benchmark source and compiler match across versions. The baseline is the verified binary from the now-committed item 8 source, built in this checkout and preserved locally; the candidate uses the same checkout with a separate Cargo target directory. Original main is the source archive used for the earlier comparisons. The baseline binary hash is verified against its recorded item 8 metadata.

Local artifacts are in `target/lbfgsb-performance/curvature-fix/` (ignored by Git): `summary.json`, `benchmark-metadata.json`, `build-metadata.json`, source/binary hashes, red/probe/green logs, before/after/original preflights, validation logs, and frozen estimates/samples under `raw/<case>/<version>`. `benchmark.py` records exact binaries, affinity, order and timing arguments. `validate.py` records all validation commands and the ndarray link flags. `before-lbfgsb.rs` preserves the committed implementation for reproduction. Benchmark source SHA-256: `2a27494c52bbdac8800ddfb8aad7db7f264e21d95b395a7745c2f9b1225c3ff1`.

Repeat the complete timing matrix with `rtk proxy python3 target/lbfgsb-performance/curvature-fix/benchmark.py` while its recorded binaries remain available. Build the candidate first with `rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-curvature-after-target cargo bench --offline --bench benchmark_matrix --no-run`. Run `rtk proxy python3 target/lbfgsb-performance/curvature-fix/validate.py` for validation. Do not compile or run tests during timing.
