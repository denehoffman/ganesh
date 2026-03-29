# Developer Baselines

This file records local baseline commands for performance and memory checks.

## Benchmark Matrix

Compile and run the lightweight benchmark matrix:

```bash
cargo bench --bench benchmark_matrix
```

## Memory Profiling

The profiling scripts below run the target command and record elapsed time plus maximum resident
set size (peak RSS) using a small local wrapper. Full output is saved under
`target/memory-profiles/`.

### Long-Running MCMC Chains

Examples:

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

Example:

```bash
./scripts/profile_hessian_memory.sh 200
```

Argument:

- dimension: Rosenbrock dimension passed to the default finite-difference Hessian path

## Suggested First Baseline Runs

```bash
./scripts/profile_mcmc_memory.sh aies 4 32 2000
./scripts/profile_mcmc_memory.sh ess 4 32 2000
./scripts/profile_hessian_memory.sh 200
```

Record the following from each log:

- elapsed wall time
- maximum resident set size
- relevant evaluation counters printed by the example
