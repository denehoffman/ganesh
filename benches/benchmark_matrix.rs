use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ganesh::{
    algorithms::{
        gradient::{lbfgsb::LBFGSBConfig, LBFGSB},
        gradient_free::{
            nelder_mead::{NelderMeadConfig, NelderMeadInit},
            NelderMead,
        },
        mcmc::{aies::AIESInit, ess::ESSInit, AIESConfig, ESSConfig, ESSMove, AIES, ESS},
        particles::{PSOConfig, Swarm, SwarmPositionInitializer, PSO},
    },
    core::MaxSteps,
    test_functions::{rastrigin::Rastrigin, rosenbrock::Rosenbrock},
    traits::Algorithm,
    DVector, Float,
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
            b.iter_batched(
                || {
                    (
                        Rosenbrock { n: ndim },
                        LBFGSB::default(),
                        DVector::from_vec(rosenbrock_start(ndim)),
                        LBFGSBConfig::default(),
                        LBFGSB::default_callbacks().with_terminator(MaxSteps(LBFGSB_STEPS)),
                    )
                },
                |(problem, mut solver, init, cfg, callbacks)| {
                    black_box(solver.process(&problem, &(), init, cfg, callbacks).unwrap());
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
            b.iter_batched(
                || {
                    (
                        Rosenbrock { n: ndim },
                        NelderMead::default(),
                        NelderMeadInit::new(rosenbrock_start(ndim)),
                        NelderMeadConfig::default(),
                        NelderMead::default_callbacks()
                            .with_terminator(MaxSteps(NELDER_MEAD_STEPS)),
                    )
                },
                |(problem, mut solver, init, cfg, callbacks)| {
                    black_box(solver.process(&problem, &(), init, cfg, callbacks).unwrap());
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
            b.iter_batched(
                || {
                    (
                        Rastrigin { n: ndim },
                        PSO::default(),
                        Swarm::new(SwarmPositionInitializer::RandomInLimits {
                            bounds: bounds.clone(),
                            n_particles: 24,
                        }),
                        PSOConfig::default()
                            .with_c1(0.1)
                            .unwrap()
                            .with_c2(0.1)
                            .unwrap()
                            .with_omega(0.8)
                            .unwrap(),
                        PSO::default_callbacks().with_terminator(MaxSteps(PSO_STEPS)),
                    )
                },
                |(problem, mut solver, init, cfg, callbacks)| {
                    black_box(solver.process(&problem, &(), init, cfg, callbacks).unwrap());
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
            b.iter_batched(
                || {
                    (
                        Rosenbrock { n: ndim },
                        AIES::default(),
                        AIESInit::new(matrix_walkers(ndim)).unwrap(),
                        AIESConfig::default(),
                        AIES::default_callbacks().with_terminator(MaxSteps(MCMC_STEPS)),
                    )
                },
                |(problem, mut solver, init, cfg, callbacks)| {
                    black_box(solver.process(&problem, &(), init, cfg, callbacks).unwrap());
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
            b.iter_batched(
                || {
                    (
                        Rosenbrock { n: ndim },
                        ESS::default(),
                        ESSInit::new(matrix_walkers(ndim)).unwrap(),
                        ESSConfig::default()
                            .with_moves([ESSMove::gaussian(0.2), ESSMove::differential(0.8)])
                            .unwrap()
                            .with_n_adaptive(5)
                            .with_max_steps(64),
                        ESS::default_callbacks().with_terminator(MaxSteps(MCMC_STEPS)),
                    )
                },
                |(problem, mut solver, init, cfg, callbacks)| {
                    black_box(solver.process(&problem, &(), init, cfg, callbacks).unwrap());
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
