**L-BFGS-B item 5 — incremental history Gram products**

Pre-commit review report, 2026-10-15. Based on approved item 4, committed as `53e8c277c7ea6c0ecdecfd00228e5d6da50edf9e`. Original comparison: `5e05d41f0bbb69d275e9fd49145a2053dece45d6`. Item 5 is uncommitted and has not been pushed. Proposed commit: `perf: incrementally update L-BFGS-B history products`.

**Changes**

Retain `SᵀY` and unscaled `SᵀS` for the bounded correction history. When a new pair is accepted, copy the retained block and compute only its new row and column. When the oldest pair is evicted, the retained block starts at row/column 1 of the preceding products. The history still uses its actual current pair count m during warmup and after eviction; its configured memory limit is unchanged.

A regular update uses 3m−1 length-n dot products plus O(m²) copying, replacing full O(nm²) Gram recomputation with O(nm + m²) work. The unscaled `SᵀS` cache stays valid as theta changes; the compact-system assembly applies the latest theta. Building W remains O(nm), and the LU cache introduced in item 4 is rebuilt once for each changed compact system. This also removes the temporary dense S/Y copies and their transposes. The retained cache adds two m×m scalar matrices.

Products are built lazily for the bounded path and rebuilt from correction vectors on bounded checkpoint restore or a transition from unbounded history. The rebuild uses the same dot products as incremental updates, so checkpoint continuation reproduces the cached values exactly. Reinitialization, history reset, and accepted unbounded updates clear the cache. The unbounded two-loop path does not allocate or compute these products. The optional cache owns a box, keeping the inactive representation to one optional pointer; the outer allocation is reused across updates.

There is no public API or checkpoint-format change. Curvature acceptance, theta, history-size policy, bounds, line searches, termination criteria, final Hessian/covariance behavior, and restricted-subspace work are unchanged. Item 2 remains excluded; item 6 is deferred until this change has been reviewed and committed.

**Before/after and original comparison**

All times are microseconds. Point estimates use Criterion's regression slope where available and mean otherwise. Percentages are ratios of these estimates; Criterion's bootstrap mean comparison may differ.

| Workload | Original | Before: item 4 | After: item 5 | Change vs item 4 | Change vs original |
|---|---:|---:|---:|---:|---:|
| Existing Rosenbrock, 2 variables | 772.87 | 36.14 | 36.23 | +0.25% | -95.31% |
| Existing Rosenbrock, 8 variables | 908.93 | 49.31 | 49.93 | +1.26% | -94.51% |
| Least squares, separate, More–Thuente | 390.69 | 212.29 | 210.51 | -0.84% | -46.12% |
| Least squares, fused, More–Thuente | 390.47 | 207.01 | 203.92 | -1.49% | -47.78% |
| Least squares, fused, Hager–Zhang | 283.20 | 166.38 | 166.18 | -0.12% | -41.32% |
| Unbounded diagonal quadratic, 64 variables | 948.02 | 63.85 | 63.65 | -0.31% | -93.29% |
| Unbounded diagonal quadratic, 512 variables | 4778.00 | 410.50 | 408.90 | -0.39% | -91.44% |
| Explicitly unbounded limits, 64 variables | 955.80 | 70.80 | 70.56 | -0.34% | -92.62% |
| Native bounded control, 64 variables | 791.95 | 681.81 | 606.17 | -11.09% | -23.46% |
| Native bounded control, 512 variables | 4556.30 | 4277.52 | 3535.52 | -17.35% | -22.40% |

95% confidence intervals, in microseconds:

| Workload | Before | After |
|---|---:|---:|
| Existing Rosenbrock, 2 variables | 35.57–36.95 | 36.17–36.31 |
| Existing Rosenbrock, 8 variables | 49.17–49.45 | 49.85–50.02 |
| Least squares, separate, More–Thuente | 211.63–213.13 | 210.28–210.71 |
| Least squares, fused, More–Thuente | 204.87–210.31 | 203.71–204.15 |
| Least squares, fused, Hager–Zhang | 166.21–166.56 | 166.05–166.34 |
| Unbounded diagonal quadratic, 64 variables | 63.79–63.90 | 63.60–63.71 |
| Unbounded diagonal quadratic, 512 variables | 410.13–410.89 | 408.49–409.33 |
| Explicitly unbounded limits, 64 variables | 70.75–70.86 | 70.49–70.65 |
| Native bounded control, 64 variables | 681.01–682.69 | 605.30–607.32 |
| Native bounded control, 512 variables | 4218.81–4355.51 | 3530.07–3541.45 |

The bounded fits are 11.09% faster at 64 variables and 17.35% faster at 512 variables than item 4, with the same evaluation requests and matched accuracy. Relative to the original implementation, the reductions are 23.46% and 22.40%. Most control comparisons are unchanged or within Criterion's noise threshold. The main run flags a small 8-variable Rosenbrock increase: 49.31 → 49.93 µs (+1.26% by point estimate). A separate before/after repeat did not reproduce that increase: 50.26 → 49.05 µs, with before/after 95% intervals of 49.16–51.68 and 49.01–49.09 µs. Both runs are retained; the main table keeps the original full-suite values. The unbounded controls do not use this cache, so their small timing movements are not credited to incremental Gram computation.

All benchmark definitions are unchanged. Existing Rosenbrock cases execute their fixed 60-step workload and retain their final-Hessian policy. Least-squares and diagonal-quadratic fits stop on convergence with their existing safeguards. The bounded cases use history size 10, limits [-5, 5], starting coordinates all 1, weights `1 + (i % 32)`, maximum line-search step 100, and final error mode `Skip`. Untimed preflights assert gradient norm below 1e-6 and objective below 1e-10.

**Matched accuracy and work**

| Bounded variables | Before objective | After objective | Before gradient norm | After gradient norm | Before = after f / g requests |
|---|---:|---:|---:|---:|---:|
| 64 | 1.9439135271037637e-13 | 1.9439135271069365e-13 | 7.8152550674562832e-07 | 7.8152550674655000e-07 | 43 / 37 |
| 512 | 1.3996892674482807e-14 | 1.3996892674282524e-14 | 4.5633778002547548e-07 | 4.5633778002253279e-07 | 49 / 42 |

Changing from matrix-matrix products to vector dot products slightly changes floating-point summation. The fits attain the same stopping tolerance with the same evaluation requests; the last-digit differences above are not reduced-accuracy fits. Unbounded and least-squares preflight results and request counts match exactly.

The initial implementation stored both matrices inline in the optimizer. Its unbounded 512-variable control measured 428.15 µs against 410.50 µs before (a 4.30% point-estimate regression), despite never building the cache. Keeping the optional state behind one box removed that regression in a targeted repeat (401.84 µs). The complete final suite above was then rerun on the boxed version. Initial estimates and logs are retained separately. No diagnostic instrumentation or compiler-inlining attributes were added to the final code.

**Validation**

The new tests compare incremental products against independently constructed dense products, in both f32 and f64, through growth and repeated eviction with limits 1, 3, and 10 and dimension 7. They also require cold rebuilds to reproduce incremental entries exactly. Ndarray runs the same product checks and a complete native-bounded optimization. Existing tests were strengthened to verify theta changes preserve the unscaled products, invalidation on rejection/reinitialization/unbounded transitions, and identical bounded checkpoint continuation after history eviction.

- `rtk cargo test --offline`: 131 tests passed.
- `rtk cargo test --offline --features python`: 137 tests passed.
- Ndarray library tests with local OpenBLAS: 117 passed.
- `rtk cargo check --offline --all-features --all-targets`: passed.
- `rtk cargo clippy --offline --all-targets -- -D warnings`: passed.
- Formatting and whitespace checks: passed.

Cargo's existing redundant-homepage warning remains. Ndarray correctness was exercised at runtime; the timing table measures nalgebra.

**Method and reproduction**

AMD Ryzen 7 3700X; Linux 6.18.56 x86_64; rustc 1.101.0-nightly (822182db4 2026-10-13). Release build pinned to logical CPU 4, 80 samples, one-second warmup, five-second collection target, no concurrent builds/tests during timings. The original, before, and after source builds used separate target directories. The benchmark SHA-256 remains `ed2ffd0c69320cbd0a4036562b3e38427270c39131a37cceb0e84ddb5b13e719` in every source snapshot.

Before: run from `/tmp/ganesh-lbfgsb-item5-before`, an archive of the base commit:

```sh
rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item5-before-target CRITERION_HOME=/home/dene/research/ganesh/target/criterion taskset -c 4 cargo bench --offline --bench benchmark_matrix -- 'algorithms/(optimization/lbfgsb|lbfgsb-evaluations|lbfgsb-paths)' --save-baseline item5-before --sample-size 80 --warm-up-time 1 --measurement-time 5 --noplot
```

After: run from the working repository:

```sh
rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item5-after-target CRITERION_HOME=/home/dene/research/ganesh/target/criterion taskset -c 4 cargo bench --offline --bench benchmark_matrix -- 'algorithms/(optimization/lbfgsb|lbfgsb-evaluations|lbfgsb-paths)' --baseline item5-before --sample-size 80 --warm-up-time 1 --measurement-time 5 --noplot
```

Original measurements are retained from the earlier reports for these identical workloads. Ndarray runtime tests use the local OpenBLAS linking flags recorded in the item 4 report, reusing `/tmp/ganesh-lbfgsb-item4-ndarray-target` for the same working checkout. Raw original/before/after estimates and samples, initial-attempt data, preflight logs, final timing output, validation log, and `summary.json` are saved locally under `target/lbfgsb-performance/item5/final/`. The named `line-search-item5-after` baseline is retained for subsequent improvements.

The Rosenbrock control repeat used the same settings and an exact case filter, with `--save-baseline item5-control-before` and `--save-baseline item5-control-after`. Its raw data are retained under `target/lbfgsb-performance/item5/final/control-repeat/` and its estimates are included in `summary.json`.
