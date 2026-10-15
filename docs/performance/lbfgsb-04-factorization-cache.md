**L-BFGS-B item 4 — reuse compact-system LU factors**

Pre-commit review report, 2026-10-15. Based on item 3, committed with the user's approval as `c02fb6b6ec6662cf1314a47de0eb4842da114056`. Original comparison: `5e05d41f0bbb69d275e9fd49145a2053dece45d6`. This item is uncommitted and has not been pushed. Proposed commit: `perf: cache L-BFGS-B compact-system factorizations`.

**Changes**

The bounded path now factors its 2m × 2m compact system once whenever correction history or theta changes. Generalized-Cauchy-point and subspace calculations reuse that factorization for both vector and matrix right-hand sides. This removes repeated matrix copies and O(m³) factorizations; each subsequent application uses O(m²) triangular solves per right-hand side. No explicit inverse is constructed.

`LinearSolve::lu_factorize` is an additive method with a default implementation. Its immutable, owned `LuSolver` handle supports vector and matrix solves. Existing custom providers continue compiling and use their existing solve methods through the default; they can override the new method to retain factors. Nalgebra retains its native LU. Ndarray retains its LAPACK LU and reuses it across every matrix RHS column as well. An `Arc` allows cloned algorithms to share immutable factors without adding a required associated type or stronger bounds to the public algorithm type.

Replacing the compact model replaces its solver. Reinitialization, rejected line searches, and transitions to the unbounded fast path retain the existing invalidation rules. A bounded checkpoint restore rebuilds factors from saved correction pairs; checkpoint data is unchanged. Singular systems retain the existing zero-result fallback, and failed ndarray factorizations are retained as failures rather than retried for each RHS.

While filling the history matrices, the implementation now borrows each correction column once and reads each coordinate once, reusing those values to fill S, Y, and W. This preserves the same arithmetic and Gram construction while avoiding redundant lookups. It addresses a construction-cost regression detected in the initial cache patch, described below.

The changing subspace matrix is still solved once where needed. Gram construction and restricted-subspace products remain unchanged for later items. Item 2 remains excluded.

**Before/after and original comparison**

All times below are microseconds. Estimates use Criterion's regression slope where available and its mean otherwise. Percentages are ratios of these point estimates; Criterion's bootstrap mean comparison can differ.

| Workload | Original | Before: item 3 | After: item 4 | Change vs item 3 | Change vs original |
|---|---:|---:|---:|---:|---:|
| Existing Rosenbrock, 2 variables | 772.87 | 36.01 | 35.54 | -1.30% | -95.40% |
| Existing Rosenbrock, 8 variables | 908.93 | 48.99 | 48.87 | -0.25% | -94.62% |
| Least squares, separate, More–Thuente | 390.69 | 211.14 | 211.48 | +0.16% | -45.87% |
| Least squares, fused, More–Thuente | 390.47 | 205.40 | 204.89 | -0.25% | -47.53% |
| Least squares, fused, Hager–Zhang | 283.20 | 166.21 | 167.19 | +0.59% | -40.96% |
| Unbounded diagonal quadratic, 64 variables | 948.02 | 63.40 | 63.32 | -0.14% | -93.32% |
| Unbounded diagonal quadratic, 512 variables | 4778.00 | 416.73 | 410.96 | -1.39% | -91.40% |
| Explicitly unbounded limits, 64 variables | 955.80 | 69.88 | 70.11 | +0.33% | -92.67% |
| Native bounded control, 64 variables | 791.95 | 788.52 | 685.70 | -13.04% | -13.42% |
| Native bounded control, 512 variables | 4556.30 | 4413.01 | 4268.71 | -3.27% | -6.31% |

95% confidence intervals, also in microseconds:

| Workload | Before | After |
|---|---:|---:|
| Existing Rosenbrock, 2 variables | 35.83–36.26 | 35.47–35.62 |
| Existing Rosenbrock, 8 variables | 48.95–49.04 | 48.82–48.92 |
| Least squares, separate, More–Thuente | 210.83–211.50 | 211.21–211.85 |
| Least squares, fused, More–Thuente | 204.67–206.51 | 204.75–205.05 |
| Least squares, fused, Hager–Zhang | 165.99–166.43 | 167.04–167.36 |
| Unbounded diagonal quadratic, 64 variables | 63.35–63.47 | 63.26–63.38 |
| Unbounded diagonal quadratic, 512 variables | 411.50–423.44 | 410.62–411.32 |
| Explicitly unbounded limits, 64 variables | 69.83–69.94 | 70.05–70.17 |
| Native bounded control, 64 variables | 780.74–800.75 | 684.87–686.56 |
| Native bounded control, 512 variables | 4407.53–4419.10 | 4250.30–4290.73 |

The existing Rosenbrock cases still execute a fixed 60-step workload, including their existing final-Hessian policy. The least-squares and diagonal-quadratic cases run to their normal convergence criteria with a step cap. The bounded quadratics use limits [-5, 5], history size 10, weights `1 + (i % 32)`, starting coordinates all 1, maximum line-search step 100, and final error mode `Skip`. Their untimed preflights assert gradient norm below 1e-6 and objective below 1e-10. The only added benchmark definition is the 512-variable bounded control.

The initial cache implementation showed small regressions in the 64-variable unbounded control and 512-variable bounded case. Both bounded cases and both 64-variable unbounded controls were repeated as a group, before then after, with the same parameters. These diagnostic results describe the initial implementation, before the history-column read reuse; they are retained separately and are not the final numbers in the main table:

| Workload | First pass, before → after (µs) | Repeat, before → after (µs) |
|---|---:|---:|
| Unbounded diagonal quadratic, 64 variables | 63.40 → 64.33 | 63.94 → 64.82 |
| Explicitly unbounded limits, 64 variables | 69.88 → 70.82 | 70.60 → 71.15 |
| Native bounded control, 64 variables | 788.52 → 717.73 | 799.85 → 719.80 |
| Native bounded control, 512 variables | 4413.01 → 4469.57 | 4423.45 → 4576.27 |

Reversing the run order also reproduced the bounded 512 slowdown. A temporary phase-timing build then compared 102 fits per source. Median totals per fit were approximately: Cauchy calculation 1004 → 969 µs; subspace calculation 2029 → 1908 µs; compact-model update 1141 → 1444 µs. The factorization itself accounted for about 46 µs of the updated model time, and the Gram products were effectively unchanged (688 → 684 µs). The remaining increase was in construction of the history matrices. Function-boundary experiments did not fix it and were reverted.

Explicitly reusing correction-column values restored construction performance. The final diagnostic update total was 1113 µs, including 45 µs for factorization and 680 µs for Gram products. Cauchy and subspace totals remained about 968 and 1901 µs. This diagnostic build contains timing overhead and is not used for the main benchmark comparison. The final code was then validated and the entire benchmark matrix above was rerun without instrumentation. A targeted final 512-variable run also measured 4.20 ms against a 4.36 ms preceding-version run, reversing the original regression.

The final bounded fits are 13.04% faster at 64 variables and 3.27% faster at 512 variables than item 3, and 13.42% / 6.31% faster than the original implementation. The larger case remains dominated by dimension-dependent matrix construction and subspace products, which later items target. The unbounded path does not use this cache; Criterion reported no change for its three path controls. Other unbounded workloads were either unchanged or within Criterion's noise threshold. No final case was classified as a performance regression. These results are for the nalgebra backend; ndarray runtime correctness was tested, but its speed was not benchmarked.

**Fit quality**

Both bounded cases have exactly the same reported objective, gradient norm, and evaluation requests before and after. All unbounded and least-squares preflight results and request counts also match item 3.

| Bounded variables | Before = after objective | Before = after gradient norm | Before = after f / g requests | Original f / g requests |
|---|---:|---:|---:|---:|
| 64 | 1.9439135271037637e-13 | 7.8152550674562832e-7 | 43 / 37 | 85 / 71 |
| 512 | 1.3996892674482807e-14 | 4.5633778002547548e-7 | 49 / 42 | 97 / 81 |

**Validation**

Seven new default-feature tests cover prepared vector/matrix solves; ownership after caller mutation; f32 and f64; pivoting and indefinite systems; repeated singular solves; compact-system agreement with freshly factored reference systems; theta changes; cloning; history eviction; reinitialization; rejected searches; bounded checkpoint continuation; and a downstream provider using only the original required methods. Ndarray has an additional f32/f64 prepared-solver test.

- `rtk cargo test --offline`: 130 tests passed.
- `rtk cargo test --offline --features python`: 136 tests passed.
- Ndarray library tests linked to local OpenBLAS: 114 passed.
- `rtk cargo check --offline --all-features --all-targets`: passed.
- `rtk cargo clippy --offline --all-targets -- -D warnings`: passed.
- Formatting and whitespace checks: passed.

Running the ndarray suite exposed an existing assertion comparing a near-zero solve result to zero with too small an absolute tolerance: OpenBLAS produced 4.44e-16. That shared test now allows 1e-12 absolute error for the zero component. The new solver tests separately check residuals at 1e-12 for f64 and 1e-5 for f32. No optimization stopping tolerance was changed. Cargo's existing redundant-homepage warning remains.

**Method and reproduction**

AMD Ryzen 7 3700X; Linux 6.18.56 x86_64; rustc 1.101.0-nightly (822182db4 2026-10-13). Release build, logical CPU 4, 80 samples, one-second warmup, five-second target collection, no concurrent compilation/tests during timings. Original, before, and after used separate build directories. The benchmark file SHA-256 is `ed2ffd0c69320cbd0a4036562b3e38427270c39131a37cceb0e84ddb5b13e719` in all three snapshots.

The full before run used `/tmp/ganesh-lbfgsb-item4-before`, an archive of `c02fb6b6ec6662cf1314a47de0eb4842da114056` with the current benchmark file copied in:

```sh
rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item4-before-target CRITERION_HOME=/home/dene/research/ganesh/target/criterion taskset -c 4 cargo bench --offline --bench benchmark_matrix -- 'algorithms/(optimization/lbfgsb|lbfgsb-evaluations|lbfgsb-paths)' --save-baseline item4-before --sample-size 80 --warm-up-time 1 --measurement-time 5 --noplot
```

The full after run used the working repository with `CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item4-after-target` and `--baseline item4-before`. For the repeated group, replace the filter with `'algorithms/lbfgsb-paths/(unconstrained/64|explicit-unbounded/64|bounded-control)'`, use `--save-baseline item4-before-repeat` before and `--baseline item4-before-repeat` after. All remaining flags stay the same.

Original results are retained from earlier reports for unchanged cases. The new bounded 512 case was measured from an archive of the original commit at `/tmp/ganesh-lbfgsb-item4-original`, with its own target directory and `--save-baseline item4-original`.

Ndarray runtime test reproduction (the LAPACK source is an application choice, so no dependency feature was changed):

```sh
rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item4-ndarray-target RUSTFLAGS='-L native=/nix/store/rbcl4dl0g5c3b5irzjls0r9nz7xlakr1-openblas-0.3.33/lib -l openblas -C link-arg=-Wl,-rpath,/nix/store/rbcl4dl0g5c3b5irzjls0r9nz7xlakr1-openblas-0.3.33/lib' cargo test --offline --features backend-ndarray --lib
```

Raw estimates/samples for the original, before, final after, initial attempts, diagnostic repeats, and phase profiles are saved locally under `target/lbfgsb-performance/item4/final/`, alongside `summary.json` and the validation log. The named `line-search-item4-after` baseline is retained for subsequent items. No instrumentation is present in the working source. After review and approval, the next improvement is item 5: incremental history Gram updates.
