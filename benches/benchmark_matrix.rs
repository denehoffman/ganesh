use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ganesh::{
    DVector, Float,
    algorithms::{
        gradient::{LBFGSB, lbfgsb::LBFGSBConfig},
        gradient_free::{NelderMead, nelder_mead::NelderMeadConfig},
        mcmc::{AIES, AIESConfig, ESS, ESSConfig, ESSMove},
        particles::{PSO, PSOConfig, Swarm, SwarmPositionInitializer},
    },
    core::MaxSteps,
    test_functions::{rastrigin::Rastrigin, rosenbrock::Rosenbrock},
    traits::Algorithm,
};

const OPT_DIMS: [usize; 2] = [2, 8];
const MCMC_DIMS: [usize; 2] = [2, 4];
const MCMC_WALKERS: usize = 12;
const MCMC_STEPS: usize = 40;
const PSO_STEPS: usize = 60;
const LBFGSB_STEPS: usize = 80;
const NELDER_MEAD_STEPS: usize = 120;

fn rosenbrock_start(n: usize) -> Vec<Float> {
    vec![5.0; n]
}

fn matrix_walkers(n: usize) -> Vec<DVector<Float>> {
    (0..MCMC_WALKERS)
        .map(|i| {
            DVector::from_fn(n, |j, _| {
                let center = if (i + j) % 2 == 0 { -1.5 } else { 1.5 };
                center + 0.1 * (i as Float) + 0.05 * (j as Float)
            })
        })
        .collect()
}

fn benchmark_lbfgsb(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix/LBFGSB");
    for n in OPT_DIMS {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, &ndim| {
            let base_cfg = LBFGSBConfig::new(rosenbrock_start(ndim));
            b.iter_batched(
                || {
                    (
                        Rosenbrock { n: ndim },
                        LBFGSB::default(),
                        base_cfg.clone(),
                        LBFGSB::default_callbacks().with_terminator(MaxSteps(LBFGSB_STEPS)),
                    )
                },
                |(problem, mut solver, cfg, callbacks)| {
                    black_box(solver.process(&problem, &(), cfg, callbacks).unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_nelder_mead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix/Nelder-Mead");
    for n in OPT_DIMS {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, &ndim| {
            let base_cfg = NelderMeadConfig::new(rosenbrock_start(ndim));
            b.iter_batched(
                || {
                    (
                        Rosenbrock { n: ndim },
                        NelderMead::default(),
                        base_cfg.clone(),
                        NelderMead::default_callbacks()
                            .with_terminator(MaxSteps(NELDER_MEAD_STEPS)),
                    )
                },
                |(problem, mut solver, cfg, callbacks)| {
                    black_box(solver.process(&problem, &(), cfg, callbacks).unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_pso(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix/PSO");
    for n in OPT_DIMS {
        group.bench_with_input(BenchmarkId::new("Rastrigin", n), &n, |b, &ndim| {
            let bounds = vec![(-5.12, 5.12); ndim];
            let base_cfg = PSOConfig::new(Swarm::new(SwarmPositionInitializer::RandomInLimits {
                bounds,
                n_particles: 24,
            }))
            .with_c1(0.1)
            .unwrap()
            .with_c2(0.1)
            .unwrap()
            .with_omega(0.8)
            .unwrap();
            b.iter_batched(
                || {
                    (
                        Rastrigin { n: ndim },
                        PSO::default(),
                        base_cfg.clone(),
                        PSO::default_callbacks().with_terminator(MaxSteps(PSO_STEPS)),
                    )
                },
                |(problem, mut solver, cfg, callbacks)| {
                    black_box(solver.process(&problem, &(), cfg, callbacks).unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_aies(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix/AIES");
    for n in MCMC_DIMS {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, &ndim| {
            let base_cfg = AIESConfig::new(matrix_walkers(ndim));
            b.iter_batched(
                || {
                    (
                        Rosenbrock { n: ndim },
                        AIES::default(),
                        base_cfg.clone(),
                        AIES::default_callbacks().with_terminator(MaxSteps(MCMC_STEPS)),
                    )
                },
                |(problem, mut solver, cfg, callbacks)| {
                    black_box(solver.process(&problem, &(), cfg, callbacks).unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_ess(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix/ESS");
    for n in MCMC_DIMS {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, &ndim| {
            let base_cfg = ESSConfig::new(matrix_walkers(ndim))
                .with_moves([ESSMove::gaussian(0.2), ESSMove::differential(0.8)])
                .with_n_adaptive(5)
                .with_max_steps(64);
            b.iter_batched(
                || {
                    (
                        Rosenbrock { n: ndim },
                        ESS::default(),
                        base_cfg.clone(),
                        ESS::default_callbacks().with_terminator(MaxSteps(MCMC_STEPS)),
                    )
                },
                |(problem, mut solver, cfg, callbacks)| {
                    black_box(solver.process(&problem, &(), cfg, callbacks).unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_lbfgsb,
    benchmark_nelder_mead,
    benchmark_pso,
    benchmark_aies,
    benchmark_ess
);
criterion_main!(benches);
