# Developer Baselines

This file records local baseline commands and one concrete baseline snapshot for performance and
memory checks.

## Baseline Environment

- Platform: local Linux workstation
- Build mode for recorded runs: `cargo run --release`
- Timing/memory wrapper: `python3 scripts/profile_with_rss.py`

## Benchmark Matrix

Compile and run the lightweight benchmark matrix:

```bash
cargo bench --bench benchmark_matrix
```

This bench target covers:

- `LBFGSB` on Rosenbrock
- `Nelder-Mead` on Rosenbrock
- `PSO` on Rastrigin
- `AIES` on Rosenbrock log density
- `ESS` on Rosenbrock log density

## Recorded Baseline Snapshot

The table below records a fixed small baseline run per algorithm family. The "convergence" column
is a convergence/status proxy from the summary message for that single run, not a statistical rate
over repeated trials.

| Algorithm | Problem | Config | cost_evals | gradient_evals | elapsed_s | peak_rss_kb | convergence/status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `LBFGSB` | Rosenbrock | `dim=2`, `MaxSteps(80)` | 206 | 109 | 0.880754 | 330516 | `success=true`, `F_EVAL CONVERGED` |
| `Nelder-Mead` | Rosenbrock | `dim=2`, `MaxSteps(120)` | 156 | 0 | 0.877897 | 50848 | `success=true`, `term_f = STDDEV` |
| `PSO` | Rastrigin | `dim=2`, `particles=24`, `MaxSteps(60)` | 1464 | 0 | 0.767087 | 51412 | `success=false`, `Maximum number of steps reached (60)` |
| `AIES` | Rosenbrock log density | `dim=2`, `walkers=12`, `MaxSteps(40)` | 492 | 0 | 0.752175 | 51000 | `success=true`, `Maximum number of steps reached (40)` |
| `ESS` | Rosenbrock log density | `dim=2`, `walkers=12`, `MaxSteps(40)` | 4200 | 0 | 0.766931 | 51236 | `success=true`, `Maximum number of steps reached (40)` |

## Memory-Focused Snapshot

Representative high-memory workload baselines from the local profiling scripts:

| Workload | Config | elapsed_s | peak_rss_kb | extra output |
| --- | --- | ---: | ---: | --- |
| Finite-difference Hessian | `dim=200` | 0.156216 | 51184 | `trace=925931.367778441` |
| ESS memory profile | `dim=2`, `walkers=8`, `steps=10` | 0.717665 | 265504 | `evals=996` |

## Commands Used For This Snapshot

```bash
python3 scripts/profile_with_rss.py cargo run --release --example profile_baseline_metrics -- lbfgsb 2
python3 scripts/profile_with_rss.py cargo run --release --example profile_baseline_metrics -- nelder_mead 2
python3 scripts/profile_with_rss.py cargo run --release --example profile_baseline_metrics -- pso 2
python3 scripts/profile_with_rss.py cargo run --release --example profile_baseline_metrics -- aies 2 12 40
python3 scripts/profile_with_rss.py cargo run --release --example profile_baseline_metrics -- ess 2 12 40
python3 scripts/profile_with_rss.py cargo run --release --example profile_hessian_memory -- 200
./scripts/profile_mcmc_memory.sh ess 2 8 10
```

## Supporting Scripts

The profiling scripts below save full output under `target/memory-profiles/`.

### Long-Running MCMC Chains

```bash
./scripts/profile_mcmc_memory.sh aies 4 32 2000
./scripts/profile_mcmc_memory.sh ess 4 32 2000
```

Arguments:

- sampler: `aies` or `ess`
- dimension: problem dimension
- walkers: number of walkers
- steps: number of MCMC steps

### High-Dimensional Finite-Difference Hessians

```bash
./scripts/profile_hessian_memory.sh 200
```

Argument:

- dimension: Rosenbrock dimension passed to the default finite-difference Hessian path
