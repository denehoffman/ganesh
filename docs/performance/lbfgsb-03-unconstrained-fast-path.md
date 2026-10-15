**L-BFGS-B item 3 — unconstrained two-loop fast path**

Pre-commit review report, 2026-10-15. Item 1 was committed with the user's approval as `f2ffbff19b46692ced43a4b46099d898b3e74301`. This change is based on that commit. Original comparison: `5e05d41f0bbb69d275e9fd49145a2053dece45d6`. Item 3 has not been committed or pushed. Proposed commit: `perf: use two-loop L-BFGS for unconstrained fits`.

Item 2 remains excluded at the user's request. After review and approval of this change, the next improvement is item 4 (reuse compact-system factorizations for bounded fits).

**Changes**

- Use the standard L-BFGS two-loop inverse-Hessian multiplication when native bounds are absent or every native bound is `ScalarBound::Unbounded`.
- Preserve the existing scaling: the initial inverse-Hessian model is `I / theta`, using the same `theta` and correction-pair acceptance rule as before. The implementation applies the inverse model directly to the negative gradient.
- Skip generalized-Cauchy-point calculation, breakpoint sorting, the subspace solve, and compact-matrix construction on this path. Direction calculation and history updates therefore cost O(nm) for n variables and m retained pairs, instead of retaining dimension-dependent O(nm²) matrix construction plus repeated small factorizations.
- Retain the constrained path for any finite native limit. If bounds are introduced after an unbounded history, rebuild its compact model before using it.
- Restore unbounded checkpoints from the stored correction history without constructing compact matrices; bounded restores rebuild the model. The checkpoint format and public configuration are unchanged.

The two-loop ordering and scaling were cross-checked against the [Ceres implementation](https://ceres-solver.googlesource.com/ceres-solver/+/master/internal/ceres/low_rank_inverse_hessian.cc). Correctness is independently tested against dense rank-two inverse-BFGS updates and the existing compact representation. The scalar curvature divisions use direct quotients, rather than first forming potentially overflowing reciprocal curvatures.

The bounded Gram-matrix update, factorization strategy, line-search policy, final Hessian estimation, and summary covariance allocation remain as before. This patch does not implement the later bounded-matrix or scratch-buffer improvements. The O(nm) statement concerns direction/history work; result construction can still allocate a dense covariance placeholder.

**Before/after results and comparison with the original benchmarks**

Times are Criterion point estimates (regression slope where available, mean otherwise). Percentages below are ratios of those estimates; negative means less elapsed time. Criterion's bootstrap mean comparison can differ slightly because it uses a different estimator.

| Workload | Original | Before: item 1 | After: item 3 | Change vs item 1 | Change vs original |
|---|---:|---:|---:|---:|---:|
| Existing Rosenbrock, 2 variables | 772.87 µs | 768.02 µs | 35.91 µs | -95.32% | -95.35% |
| Existing Rosenbrock, 8 variables | 908.93 µs | 913.14 µs | 49.00 µs | -94.63% | -94.61% |
| Least squares, separate, More–Thuente | 390.69 µs | 217.06 µs | 210.52 µs | -3.02% | -46.12% |
| Least squares, fused, More–Thuente | 390.47 µs | 210.98 µs | 204.00 µs | -3.31% | -47.75% |
| Least squares, fused, Hager–Zhang | 283.20 µs | 170.69 µs | 166.51 µs | -2.45% | -41.20% |
| Unbounded diagonal quadratic, 64 variables | 948.02 µs | 949.54 µs | 62.91 µs | -93.37% | -93.36% |
| Unbounded diagonal quadratic, 512 variables | 4778.00 µs | 4744.50 µs | 408.66 µs | -91.39% | -91.45% |
| Explicitly unbounded limits, 64 variables | 955.80 µs | 958.41 µs | 70.42 µs | -92.65% | -92.63% |
| Native bounded control, 64 variables | 791.95 µs | 792.15 µs | 791.36 µs | -0.10% | -0.07% |

The bounded-control comparison detected no performance change (Criterion p=0.18); its results and evaluation counts are identical. Expensive least-squares evaluations dominate those cases, so their additional benefit is much smaller than the optimizer-heavy workloads. The existing Rosenbrock benchmarks deliberately execute a fixed 60-step workload; their roughly 95% reduction should be interpreted as a fixed-work result. The larger quadratic fits below provide separate complete-fit evidence at matched accuracy.

95% confidence intervals for the before/after time estimates:

| Workload | Before | After |
|---|---:|---:|
| Existing Rosenbrock, 2 variables | 767.00–768.92 µs | 35.84–35.99 µs |
| Existing Rosenbrock, 8 variables | 911.88–914.58 µs | 48.68–49.41 µs |
| Least squares, separate, More–Thuente | 216.82–217.35 µs | 210.27–210.79 µs |
| Least squares, fused, More–Thuente | 210.87–211.10 µs | 203.76–204.28 µs |
| Least squares, fused, Hager–Zhang | 170.57–170.82 µs | 166.35–166.69 µs |
| Unbounded diagonal quadratic, 64 variables | 948.51–950.74 µs | 62.80–63.05 µs |
| Unbounded diagonal quadratic, 512 variables | 4739.98–4749.01 µs | 408.05–409.39 µs |
| Explicitly unbounded limits, 64 variables | 956.95–960.09 µs | 70.31–70.55 µs |
| Native bounded control, 64 variables | 790.11–794.31 µs | 790.40–792.53 µs |

**Matched fit quality and evaluation work**

The new diagonal-quadratic fits use an analytic objective and gradient, weights `1 + (i % 32)`, starting coordinates all 1, history size 10, maximum line-search step 100, and final error mode `Skip` to isolate optimizer costs. They stop on gradient norm below 1e-6, with a 500-step safeguard. An untimed preflight asserts gradient norm below 1e-6 and objective below 1e-10 for every case, including the original implementation. The native bounded control uses limits [-5, 5]. Final Hessian behavior in the existing benchmarks is unchanged.

| Workload | Before objective | After objective | Before gradient norm | After gradient norm | Objective / gradient requests, before = after |
|---|---:|---:|---:|---:|---:|
| Unbounded, 64 variables | 8.845766162290962e-14 | 8.845766162248790e-14 | 9.836374135117609e-7 | 9.836374135092966e-7 | 51 / 43 |
| Unbounded, 512 variables | 3.102258334369156e-14 | 3.102258334355677e-14 | 9.105532600299960e-7 | 9.105532600212081e-7 | 55 / 46 |
| Explicitly unbounded, 64 variables | 8.845766162290962e-14 | 8.845766162248790e-14 | 9.836374135117609e-7 | 9.836374135092966e-7 | 51 / 43 |
| Native bounded control, 64 variables | 1.943913527103764e-13 | 1.943913527103764e-13 | 7.815255067456283e-7 | 7.815255067456283e-7 | 43 / 37 |

The last-digit differences in unbounded results reflect the new arithmetic ordering. These tests compare the mathematical direction within numerical tolerances and complete fits at the same stopping tolerance; they do not require bit-for-bit identity between different linear-algebra formulations. The least-squares preflight objectives and request counts also match item 1 exactly.

**Validation**

Ten new unit tests cover dense inverse-BFGS agreement with 0, 1, 3, and 7 correction pairs; agreement with the old compact direction; native f32; finite zero-gradient directions; native-bound classification; avoiding compact construction; introducing/removing bounds; checkpoint continuation and both restore modes; and a constrained solution with mixed unbounded, lower-only, and upper-only limits.

- `rtk cargo test --offline`: 123 tests passed.
- `rtk cargo test --offline --features python`: 129 tests passed.
- `rtk cargo check --offline --all-features --all-targets`: passed.
- `rtk cargo clippy --offline --all-targets -- -D warnings`: passed.
- `rtk cargo fmt -- --check` and `rtk git diff --check`: passed.

Cargo's existing redundant homepage metadata warning remains. The strict all-features PyO3 lint diagnostics recorded in the item 1 report are outside this change.

**Benchmark method and reproduction**

AMD Ryzen 7 3700X; Linux 6.18.56 x86_64; rustc 1.101.0-nightly (822182db4 2026-10-13). Release profile, pinned to logical CPU 4, 80 samples, one-second warmup, five-second target measurement time. Criterion may extend collection to obtain all samples. No concurrent build/test process ran during final timings.

Original source, item 1 source, and working source used separate build directories and identical benchmark definitions. Existing case definitions were preserved; four new path/scale cases were copied into each source snapshot. The original measurements for the existing cases are retained from item 1; the four new cases were also measured against the original source for this report.

Run the before command from `/tmp/ganesh-lbfgsb-item3-before`, a source archive of `f2ffbff` containing the current benchmark file:

```sh
rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item3-before-target CRITERION_HOME=/home/dene/research/ganesh/target/criterion taskset -c 4 cargo bench --offline --bench benchmark_matrix -- 'algorithms/(optimization/lbfgsb|lbfgsb-evaluations|lbfgsb-paths)' --save-baseline item3-before --sample-size 80 --warm-up-time 1 --measurement-time 5 --noplot
```

Run the after command from the working repository:

```sh
rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item3-after-target CRITERION_HOME=/home/dene/research/ganesh/target/criterion taskset -c 4 cargo bench --offline --bench benchmark_matrix -- 'algorithms/(optimization/lbfgsb|lbfgsb-evaluations|lbfgsb-paths)' --baseline item3-before --sample-size 80 --warm-up-time 1 --measurement-time 5 --noplot
```

The added cases were measured on the original source from `/tmp/ganesh-lbfgsb-item1-baseline` with its separate build directory and `--save-baseline item3-original`. Their data were added to the common `line-search-original` baseline. Raw estimates, samples, and `summary.json` are saved locally under `target/lbfgsb-performance/item3/final/`. The named `line-search-item3-after` baseline is retained for subsequent improvements.
