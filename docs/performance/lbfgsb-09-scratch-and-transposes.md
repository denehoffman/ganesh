**L-BFGS-B item 9 — reuse vector storage and transposes**

Pre-commit review report, 2026-10-16. Branch: `development`; base: approved Cauchy-point fix, `7fd507bd0b46282129ae61b7442487cc84b8b531`. Original-main comparison: `5e05d41f0bbb69d275e9fd49145a2053dece45d6`. This item is uncommitted. Proposed commit: `perf: reuse L-BFGS-B vector storage and transposes`.

**Changes**

The two-loop recursion previously allocated a replacement direction vector for every scaled addition and for the middle scaling. It now updates its existing direction storage. The Cauchy calculation similarly updates its p/c work vectors and fills one local row buffer when crossing bounds, rather than allocating a new row at every crossing. On the built-in providers, the row buffer allocates only if a crossing needs a nonempty row; it is not retained across calls.

The initial W-transpose/vector product now uses a provider operation that does not require an owned transpose on the built-in backends. Subspace minimization materializes its restricted transpose once and reuses it for both products, and reuses its vector storage through the final correction/scaling. The compact Gram calculation is still performed directly; the rejected restricted-product cache has not returned.

Three optional provider operations support this: `vec_add_scaled_mut`, `vec_scale_mut`, and `mat_transpose_vec`, exposed as `Vector::add_scaled_mut`, `Vector::scale_mut` and `Matrix::transpose_mul_vec`. Their defaults delegate to the existing allocating operations, preserving existing custom providers without new required methods. Nalgebra reuses its existing AXPY/scaling operations and direct transpose multiplication. Ndarray uses mutable element operations and a transposed view for dot products. Only L-BFGS-B is migrated in this item.

For an unbounded iteration retaining m correction pairs, counting the owning operations in the two-loop recursion removes 2m+1 vector allocations on nalgebra and 4m+1 on ndarray. At m=10 that is 21 and 41 allocations, respectively. These are source-level counts of the replaced operations, not whole-fit allocator-profiler measurements. The initial direction and coefficient vector remain local allocations. Storage-reuse tests verify that the built-in mutable vector operations retain their backing address.

All work storage remains local to the calculation. No persistent scratch cache, optimizer field, checkpoint format, history policy, line-search policy or final Hessian/error-estimation policy is changed.

**Before/after and original comparison**

All 17 existing benchmark definitions and tolerances are unchanged. The timing matrix uses nalgebra; ndarray correctness and storage reuse are validated separately. Times are microseconds; negative percentages mean less elapsed time. Before/after are measured consecutively per case, alternating order across cases; original main is freshly measured after each pair. Each timing uses 80 samples, three-second warmup and five-second collection target, pinned to logical CPU 4. Estimates use Criterion's regression slope when available, otherwise its mean. Confidence intervals describe within-run uncertainty and do not include all between-run variation.

| Workload | Original main | Before: curvature fix | After: item 9 | Change vs before | Change vs original |
|---|---:|---:|---:|---:|---:|
| Existing Rosenbrock, 2 variables | 762.30 | 30.31 | 23.27 | -23.23% | -96.95% |
| Existing Rosenbrock, 8 variables | 914.42 | 47.56 | 39.95 | -16.01% | -95.63% |
| Least squares, separate, More–Thuente | 392.32 | 210.24 | 210.68 | +0.21% | -46.30% |
| Least squares, fused, More–Thuente | 392.31 | 204.33 | 203.12 | -0.59% | -48.23% |
| Least squares, fused, Hager–Zhang | 284.34 | 166.34 | 166.19 | -0.09% | -41.55% |
| Unbounded diagonal quadratic, 64 variables | 943.25 | 63.49 | 52.04 | -18.04% | -94.48% |
| Unbounded diagonal quadratic, 512 variables | 4889.47 | 402.40 | 355.62 | -11.63% | -92.73% |
| Explicitly unbounded limits, 64 variables | 960.53 | 70.34 | 58.90 | -16.28% | -93.87% |
| Native bounded control, 64 variables | 797.34 | 567.16 | 528.61 | -6.80% | -33.70% |
| Native bounded control, 512 variables | 4447.36 | 3279.77 | 2649.38 | -19.22% | -40.43% |
| Active bounds: 8 of 512 | 3927.45 | 2744.65 | 2190.13 | -20.20% | -44.24% |
| Active bounds: 384 of 512 | 3256.42 | 2145.56 | 1606.11 | -25.14% | -50.68% |
| Small-scale quadratic, 64 variables | 818.74 | 47.08 | 37.62 | -20.09% | -95.40% |
| Small-scale quadratic, 512 variables | 4126.96 | 305.93 | 270.48 | -11.59% | -93.45% |
| Interior native bounds, 512 variables | 4188.06 | 2345.71 | 1860.90 | -20.67% | -55.57% |
| Interior native bounds, 4096 variables | 105042.96 | 88490.90 | 84369.35 | -4.66% | -19.68% |
| All-active native bounds, 4096 variables | Did not converge | 73919.38 | 72141.08 | -2.41% | — |

95% confidence intervals, in microseconds:

| Workload | Before | After |
|---|---:|---:|
| Existing Rosenbrock, 2 variables | 30.23–30.40 | 23.24–23.30 |
| Existing Rosenbrock, 8 variables | 47.53–47.60 | 39.92–39.98 |
| Least squares, separate, More–Thuente | 209.94–210.58 | 210.31–211.13 |
| Least squares, fused, More–Thuente | 204.02–204.68 | 202.97–203.28 |
| Least squares, fused, Hager–Zhang | 166.01–166.74 | 165.92–166.49 |
| Unbounded diagonal quadratic, 64 variables | 63.45–63.54 | 51.96–52.18 |
| Unbounded diagonal quadratic, 512 variables | 401.61–403.27 | 350.57–362.40 |
| Explicitly unbounded limits, 64 variables | 70.23–70.46 | 58.80–59.00 |
| Native bounded control, 64 variables | 566.29–568.29 | 522.69–539.52 |
| Native bounded control, 512 variables | 3273.60–3286.08 | 2611.39–2696.54 |
| Active bounds: 8 of 512 | 2731.20–2764.15 | 2174.11–2209.24 |
| Active bounds: 384 of 512 | 2140.68–2151.69 | 1602.89–1609.92 |
| Small-scale quadratic, 64 variables | 46.19–48.21 | 37.53–37.76 |
| Small-scale quadratic, 512 variables | 305.36–306.55 | 270.10–270.89 |
| Interior native bounds, 512 variables | 2334.42–2360.93 | 1857.84–1864.14 |
| Interior native bounds, 4096 variables | 88198.43–88797.52 | 84091.02–84665.46 |
| All-active native bounds, 4096 variables | 73175.20–74759.80 | 71875.64–72415.02 |

I recommend keeping this item. The complete unbounded quadratic fits improve by about 12–20%, the native bounded controls by 7% / 19%, and the sparse/dense active-bound fits by 20% / 25%. The native-interior cases improve by 21% at 512 variables and 5% at 4096; the all-active 4096-variable case improves by 2%. Least-squares cases remain within 1% of the baseline. The fixed-60-step Rosenbrock results improve by 16–23%, with the convergence caveat below.

Reversed-order repeats used 80 samples, five-second warmups and five-second collection targets:

| Repeat workload | Before (us) | After (us) | Change |
|---|---:|---:|---:|
| Unbounded diagonal quadratic, 512 variables | 404.32 | 350.28 | -13.37% |
| Native bounded control, 512 variables | 3280.78 | 2631.52 | -19.79% |
| Interior native bounds, 4096 variables | 87659.42 | 84947.38 | -3.09% |

These repeats confirm roughly 13% gain in the 512-variable unbounded fit, 20% in the bounded control and 3% in the large interior fit. Repeat order is after/before for the unbounded case, before/after for both bounded cases: the reverse of their main pairs. Equal evaluation counts make these elapsed-time improvements an optimizer-overhead comparison, with no tolerance relaxation or early-exit difference.


**Numerical work and quality**

Every before/after case uses exactly the same f/g request counts and meets its existing feasibility and accuracy checks. Eleven preflight result lines are identical. The bounded controls, sparse/dense active-bound fits and native-interior fits have tiny differences in final low-order digits after using direct transpose multiplication; they meet their unchanged stopping tolerances. The provider contract compares this product against an explicitly transposed matrix at the existing scalar-appropriate tolerance. No bitwise-equivalence claim is made for the changed product.

The two existing Rosenbrock cases remain fixed at 60 steps; their timing changes are fixed-work results, not complete-fit convergence claims. The other cases check fit quality, and the active/interior cases explicitly verify projected-gradient termination rather than a safeguard exit. The existing line-search, Cauchy-point correctness, mixed-bound, f32, rolling-history and checkpoint/resume regressions all pass.

Original main still cannot complete the all-active 4096-variable case within 500 steps. Its f=33.792, 103001 / 2001 f/g result is retained as nonconvergence and its timing is skipped. Every other original comparison meets its existing checks. Current before/after all-active fits both finish at the exact lower-bound solution with f=30.201015968426596 and 5 / 5 requests.

**Validation**

- Default suite: 146 passed, including doctests.
- Python feature suite: 152 passed, including doctests.
- Full ndarray suite: 155 passed, including doctests, with local OpenBLAS linking.
- All-feature/all-target check: passed.
- Strict clippy for default and ndarray configurations, all targets: passed.
- Formatting and whitespace checks: passed.

New contract tests cover f32/f64 on both backends, vector lengths 0, 1, 7, 64 and 512, and four scaling coefficients. They compare mutable operations with the allocating results and verify that backing addresses remain unchanged. Rectangular and empty matrix tests compare direct transpose-vector multiplication with the explicit-transpose reference. A downstream scalar/provider test implements none of the new methods and verifies their defaults.

**Reproduction and artifacts**

CPU: AMD Ryzen 7 3700X; compiler: rustc 1.101.0-nightly (822182db4 2026-10-13). All versions use the identical benchmark source and matching Cargo.lock. Before is the verified binary of the now-committed curvature fix, built in this checkout; after uses the same checkout and a separate Cargo target directory. Original main uses the archived source snapshot from the preceding reports. No builds or test suites run during timing.

Local artifacts are in `target/lbfgsb-performance/item9/` (ignored by Git): `summary.json`, binary/source metadata and hashes, before/after/original preflights, paired timing logs, validation logs and frozen samples/estimates under `raw/<case>/<version>`. `before-source/` preserves the approved source files. `benchmark.py` records exact binaries, affinity, order and timing arguments; `validate.py` records the suites and native linking configuration. Benchmark source SHA-256: `2a27494c52bbdac8800ddfb8aad7db7f264e21d95b395a7745c2f9b1225c3ff1`.

Repeat the complete matrix with `rtk proxy python3 target/lbfgsb-performance/item9/benchmark.py` while its recorded binaries remain available. Build the candidate first with `rtk proxy env CARGO_TARGET_DIR=/tmp/ganesh-lbfgsb-item9-after-target cargo bench --offline --bench benchmark_matrix --no-run`. Run `rtk proxy python3 target/lbfgsb-performance/item9/validate.py` for the main validation suites. The additional strict ndarray clippy command and its successful output are recorded in `ndarray-clippy-final.log`; native link flags match `validate.py`. Complete compilation and validation before timing.
