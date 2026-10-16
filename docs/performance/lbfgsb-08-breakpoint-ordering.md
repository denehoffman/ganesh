**L-BFGS-B item 8 — deferred breakpoint ordering**

Pre-commit review report, 2026-10-15 (America/Chicago). Branch: `development`; base: approved item 7, `52de6b753ecd4577f38b20ba00adc5165a41cf44`. Original comparison: `5e05d41f0bbb69d275e9fd49145a2053dece45d6`. This item is uncommitted. Proposed commit: `perf: defer L-BFGS-B breakpoint ordering`.

**Change**

The generalized Cauchy point previously stably sorted every positive breakpoint before examining its first segment. It now finds the first minimum with a linear scan, and builds an index heap only when a second breakpoint is requested. Remaining breakpoints are extracted as needed. Equal times are ordered by coordinate index, exactly preserving the previous stable order. Positive infinity remains eligible; nonpositive values and NaNs remain excluded by the same predicate.

For n coordinates and k requested breakpoints, ordering needs O(n) work when only the first is requested, and O(n + k log n) if more are needed, versus the previous worst-case O(n log n) full sort. Consuming all breakpoints has the same asymptotic cost as sorting and can have a larger constant. The heap stores indices rather than duplicating scalar values. It uses one index vector; all other Cauchy arithmetic, curvature recurrences, bounds, line searches, correction history, checkpoints and final error-estimation policy are unchanged. Entirely unbounded fits continue to bypass this code.

**Before/after and original comparison**

Times are microseconds; negative percentages mean less elapsed time. All three source versions were freshly measured on identical benchmark source. Before and after were built at the same source location with separate build targets, then timed consecutively per case, alternating their order across cases. The original was timed after each pair. Each timing uses 80 samples, a three-second warmup, and a five-second collection target. Point estimates use Criterion's regression slope when available, otherwise its mean. Confidence intervals describe within-run sampling uncertainty and do not capture all between-run variation.

| Workload | Original main | Before: item 7 | After: item 8 | Change vs item 7 | Change vs original |
|---|---:|---:|---:|---:|---:|
| Existing Rosenbrock, 2 variables | 780.73 | 30.29 | 30.47 | +0.60% | -96.10% |
| Existing Rosenbrock, 8 variables | 916.58 | 48.63 | 49.17 | +1.11% | -94.64% |
| Least squares, separate, More–Thuente | 392.13 | 210.82 | 212.28 | +0.70% | -45.86% |
| Least squares, fused, More–Thuente | 392.81 | 204.40 | 205.14 | +0.36% | -47.78% |
| Least squares, fused, Hager–Zhang | 284.08 | 167.33 | 170.18 | +1.70% | -40.10% |
| Unbounded diagonal quadratic, 64 variables | 953.63 | 63.17 | 64.04 | +1.37% | -93.29% |
| Unbounded diagonal quadratic, 512 variables | 4811.73 | 403.31 | 402.64 | -0.16% | -91.63% |
| Explicitly unbounded limits, 64 variables | 959.49 | 69.83 | 70.22 | +0.55% | -92.68% |
| Native bounded control, 64 variables | 791.21 | 607.46 | 570.05 | -6.16% | -27.95% |
| Native bounded control, 512 variables | 4519.14 | 3555.86 | 3305.23 | -7.05% | -26.86% |
| Active bounds: 8 of 512 | 3936.75 | 3194.64 | 2908.91 | -8.94% | -26.11% |
| Active bounds: 384 of 512 | 3274.95 | 2264.27 | 2198.17 | -2.92% | -32.88% |
| Small-scale quadratic, 64 variables | 831.78 | 45.97 | 45.67 | -0.64% | -94.51% |
| Small-scale quadratic, 512 variables | 4110.19 | 307.14 | 308.14 | +0.33% | -92.50% |
| Interior native bounds, 512 variables | 4213.29 | 2626.12 | 2331.23 | -11.23% | -44.67% |
| Interior native bounds, 4096 variables | 103256.62 | 92158.42 | 88378.42 | -4.10% | -14.41% |
| All-active native bounds, 4096 variables | Did not converge | 75097.11 | 75002.59 | -0.13% | — |

95% confidence intervals, in microseconds:

| Workload | Before | After |
|---|---:|---:|
| Existing Rosenbrock, 2 variables | 30.24–30.35 | 30.39–30.62 |
| Existing Rosenbrock, 8 variables | 48.36–49.16 | 49.08–49.28 |
| Least squares, separate, More–Thuente | 210.65–211.00 | 212.17–212.42 |
| Least squares, fused, More–Thuente | 204.30–204.51 | 204.88–205.43 |
| Least squares, fused, Hager–Zhang | 167.02–167.63 | 170.06–170.35 |
| Unbounded diagonal quadratic, 64 variables | 63.10–63.27 | 63.97–64.11 |
| Unbounded diagonal quadratic, 512 variables | 402.77–403.96 | 402.36–403.04 |
| Explicitly unbounded limits, 64 variables | 69.76–69.93 | 70.17–70.27 |
| Native bounded control, 64 variables | 606.86–608.18 | 569.39–570.72 |
| Native bounded control, 512 variables | 3545.27–3566.94 | 3298.45–3312.23 |
| Active bounds: 8 of 512 | 3185.72–3203.00 | 2905.20–2912.54 |
| Active bounds: 384 of 512 | 2259.88–2268.41 | 2195.76–2201.12 |
| Small-scale quadratic, 64 variables | 45.91–46.03 | 45.62–45.73 |
| Small-scale quadratic, 512 variables | 306.61–307.71 | 307.70–308.59 |
| Interior native bounds, 512 variables | 2618.83–2634.39 | 2326.86–2335.46 |
| Interior native bounds, 4096 variables | 91849.50–92501.28 | 88081.69–88681.37 |
| All-active native bounds, 4096 variables | 74894.79–75317.20 | 74801.20–75224.92 |

I recommend retaining this optimization. In the main paired run, the existing bounded controls improve by 6.16% and 7.05%, and the sparse/dense active-bound fits by 8.94% and 2.92%. The new interior-bound cases improve by 11.23% at 512 variables and 4.10% at 4096. The all-active 4096-variable case is essentially unchanged (-0.13%).

The paths that bypass this ordering code vary from -0.64% to +1.70%; the optimization does not provide a general speedup for unbounded fits. These small control differences are retained in the table rather than attributed to numerical work changes.

Reversed-order repeat pairs used 80 samples, five-second warmups and five-second collection targets. They confirm the bounded/interior gains and no material change in the all-active case:

| Repeat workload | Before (us) | After (us) | Change |
|---|---:|---:|---:|
| Native bounded control, 512 variables | 3537.77 | 3250.68 | -8.11% |
| Interior native bounds, 4096 variables | 91362.01 | 87675.79 | -4.03% |
| All-active native bounds, 4096 variables | 74701.84 | 74764.65 | +0.08% |

The repeat order is the reverse of each case's main pair: before/after for the bounded control and interior case, after/before for the all-active case. Fit-quality and evaluation checks remain identical. The all-active before/after confidence intervals overlap in both rounds, so its sub-percent differences are treated as unchanged.


**Fit quality and benchmark coverage**

All 17 before/after preflight outputs match exactly: objectives, gradient or projected-gradient norms where reported, objective gaps and f/g request counts. Feasibility and convergence-reason checks pass. The two existing Rosenbrock benchmarks remain fixed at 60 steps, as in the original suite; their times are fixed-work comparisons, not complete-fit convergence claims. The other 15 cases verify complete-fit quality, and the three new cases explicitly require projected-gradient termination rather than the safeguard limit.

The previous 14 benchmark definitions and tolerances are unchanged. Three new cases use the existing small-scale diagonal quadratic (weights 1 through 32, multiplied by 1e-3), starting at all ones, with history size 10, the default line-search maximum of 1e8, and final Hessian/error estimation skipped. They stop at projected-gradient infinity norm below 1e-8 with a 500-step safeguard. Bound distances vary with a deterministic permutation to introduce distinct breakpoint times. Every successful fit checks feasibility and objective gap below 1e-10.

| New fit | Before and after final f | Projected norm | f/g requests | Original main |
|---|---:|---:|---:|---|
| Interior, 512 | 2.1010823055221334e-13 | 3.6627282909503233e-9 | 34 / 32 | Converged, 82 / 75 requests |
| Interior, 4096 | 1.6808658444225537e-12 | 3.6627282909534628e-9 | 34 / 32 | Converged, 82 / 74 requests |
| All-active, 4096 | 30.201015968426596 | 0 | 12 / 12 | Did not converge within 500 steps |

The interior cases use lower bounds -5-5u and upper bounds 5+5u, where u=((977 i) mod n)/n. Their optimum is zero. The all-active case uses lower bounds .9+.09u and upper bounds 2, so the optimum is the vector of lower bounds; the final objective gap is 1.46e-13. This case reaches many breakpoints and provides a counterweight to the interior workload.

Original main fails on the all-active case with objective 33.792 and 103001 / 2001 f/g requests at the 500-step limit. Its timing is explicitly skipped. Comparing that unfinished run against a successful fit would give a misleading speedup. The original interior fits meet the same tolerance, with final objective/projected norm 1.16145e-13 / 5.70141e-9 at 512 and 1.00402e-12 / 7.61087e-9 at 4096. Achieved precision and trajectories may differ from the current branch even though requested quality is the same.

**Independent correctness finding, deferred**

The new dense-model validation exposed a pre-existing error after a breakpoint is crossed with correction history. It reproduces identically on the committed item 7 source, which still uses the full stable sort. This optimization preserves that behavior; it does not repair the recurrence.

A minimal example has x=[0,0], gradient=[1,1], theta=5/3, and one correction pair s=[1,1], y=[1,2]. The BFGS Hessian is [[7/6,-1/6],[-1/6,13/6]]. Give coordinate 0 a lower bound of -.1 and leave coordinate 1 unbounded. After reaching the bound, the projected quadratic's minimum is [-.1,-61/130], approximately [-.1,-.4692307692]. The implementation returns [-.1,-1.7].

A module-local regression test for a follow-up correction is:

```rust
#[test]
fn cauchy_curvature_after_one_bound() {
    let mut algorithm = LBFGSB::<f64> {
        x: Vector::zeros(2),
        gradient: Vector::from_vec(vec![1.0, 1.0]),
        theta: 5.0 / 3.0,
        s_history: vec![Vector::from_vec(vec![1.0, 1.0])],
        y_history: vec![Vector::from_vec(vec![1.0, 2.0])],
        ..Default::default()
    };
    algorithm.update_compact_matrices();
    let bounds = [ScalarBound::Lower(-0.1), ScalarBound::Unbounded];
    let (x, _, _) = algorithm.generalized_cauchy_point(Some(&bounds));
    assert!((x.get(0) + 0.1).abs() < 1e-14);
    assert!((x.get(1) + 61.0 / 130.0).abs() < 1e-14);
}
```

The curvature after removing coordinate i from direction d must account for p becoming p+g_i w_i. Expanding theta*||d||²-pᵀMp gives a curvature change of -theta*g_i² - 2*g_i*w_iᵀMp - g_i²*w_iᵀMw_i. The current recurrence has positive signs on the two compact terms. A reversible sign-only probe makes the two-variable test pass. A broader seven-pair fixture still disagrees after that probe, so this is not presented as a complete correction or a validated production patch. All diagnostic source changes were removed.

The local reproduction is retained as `target/lbfgsb-performance/item8/reproduce-curvature.py`. Running it through `rtk proxy python3` reproduces the failure on the saved committed source and restores the working file afterward; `--probe` checks the proposed signs on the minimal example. Do not run this compilation during benchmark timing. The self-contained test is also retained as `curvature-repro-test.rs`. The broader failing fixture and both diagnostic logs are retained beside it. This deserves a separate correctness investigation and commit, with its own before/after fit-quality measurements.

**Validation**

- Default suite: 140 passed, including doctests.
- Python feature suite: 146 passed, including doctests.
- Full ndarray suite: 148 passed, including doctests, using the configured local OpenBLAS library.
- `rtk cargo check --offline --all-features --all-targets`: passed.
- `rtk cargo clippy --offline --all-targets -- -D warnings`: passed.
- Formatting and whitespace checks: passed.

New ordering tests compare all yielded indices against Rust's independent stable sort for f32 and f64, including empty/singleton inputs, repeated times, permuted/reversed inputs, infinities, nonpositive values and NaNs. Independent dense BFGS-model tests verify the Cauchy point, projected displacement and free variables with both scalar types and both backends. Those mathematical checks cover crossed segments without correction history and the first interval with history; they intentionally do not assert correctness of the known-broken history recurrence after crossing. The failing broader diagnostic remains separate and documented above. Existing bounded fits, checkpoint/resume and history-eviction tests also pass.

**Reproduction and artifacts**

CPU: AMD Ryzen 7 3700X; logical CPU 4 pinned for each timing. Compiler: rustc 1.101.0-nightly (822182db4 2026-10-13). Cargo.lock and Cargo.toml match across source versions. Before and after use the same checkout path and separate Cargo target directories; original main is an archived source snapshot with the identical benchmark file. No builds or test suites ran during timing.

Local raw estimates, samples, preflight output, validation logs, binary/source hashes, original-main convergence failure and diagnostic probes are in `target/lbfgsb-performance/item8/`. The directory is ignored by Git. `summary.json` retains all estimates and confidence intervals; `raw/<case>/{before,after,original}` freezes their named Criterion baselines. `benchmark.py` records the invocation/order; `benchmark-metadata.json` records the exact binaries and protocol. Benchmark source SHA-256: `2a27494c52bbdac8800ddfb8aad7db7f264e21d95b395a7745c2f9b1225c3ff1`.

The complete paired timing run can be repeated with `rtk proxy python3 target/lbfgsb-performance/item8/benchmark.py` while its recorded binaries remain available. Build each source first with `rtk proxy env CARGO_TARGET_DIR=<separate-target> cargo bench --offline --bench benchmark_matrix --no-run`, using identical benchmark source in each version. The ndarray validation requires `RUSTFLAGS` and `RUSTDOCFLAGS` pointing to the local OpenBLAS directory, as recorded in the validation command script/log. Compilation must finish before starting the timed script.
