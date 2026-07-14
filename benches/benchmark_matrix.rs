use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ganesh::{
    algorithms::{
        gradient::{
            Adam, AdamConfig, ConjugateGradient, ConjugateGradientConfig, LBFGSBConfig,
            TrustRegion, TrustRegionConfig, LBFGSB,
        },
        gradient_free::{
            CMAESConfig, DifferentialEvolution, DifferentialEvolutionConfig, NelderMead,
            NelderMeadConfig, SimulatedAnnealing, SimulatedAnnealingConfig, CMAES,
        },
        line_search::{BacktrackingLineSearch, HagerZhangLineSearch, MoreThuenteLineSearch},
        mcmc::{AIESConfig, AIESInit, ESSConfig, ESSInit, AIES, ESS},
        particles::{PSOConfig, PSO},
    },
    core::{Callbacks, EvalCounts, MaxSteps},
    test_functions::{rastrigin::Rastrigin, rosenbrock::Rosenbrock},
    traits::{Algorithm, LineSearch},
    Bounds, Vector,
};

const DIMS: [usize; 2] = [2, 8];

fn start(n: usize) -> Vector<f64> {
    vec![2.0; n].into()
}

fn walkers(n: usize) -> Vec<Vector<f64>> {
    (0..12)
        .map(|i| {
            (0..n)
                .map(|j| if (i + j) % 2 == 0 { -1.5 } else { 1.5 })
                .collect::<Vec<_>>()
                .into()
        })
        .collect()
}

macro_rules! bench_optimizer {
    ($group:expr, $name:literal, $n:expr, $setup:expr) => {
        $group.bench_with_input(BenchmarkId::new($name, $n), &$n, |b, &_n| {
            b.iter_batched(
                || $setup,
                |(problem, mut algorithm, init, config, callbacks)| {
                    black_box(
                        algorithm
                            .process(&problem, &(), init, config, callbacks)
                            .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    };
}

fn benchmark_optimizers(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms/optimization");
    for n in DIMS {
        bench_optimizer!(group, "adam", n, {
            (
                Rosenbrock { n },
                Adam::default(),
                start(n),
                AdamConfig::default(),
                Callbacks::empty().with_terminator(MaxSteps(100)),
            )
        });
        bench_optimizer!(group, "conjugate-gradient", n, {
            (
                Rosenbrock { n },
                ConjugateGradient::<f64>::default(),
                start(n),
                ConjugateGradientConfig::default(),
                Callbacks::empty().with_terminator(MaxSteps(60)),
            )
        });
        bench_optimizer!(group, "lbfgsb", n, {
            (
                Rosenbrock { n },
                LBFGSB::<f64>::default(),
                start(n),
                LBFGSBConfig::default(),
                Callbacks::empty().with_terminator(MaxSteps(60)),
            )
        });
        bench_optimizer!(group, "trust-region", n, {
            (
                Rosenbrock { n },
                TrustRegion::default(),
                start(n),
                TrustRegionConfig::default(),
                Callbacks::empty().with_terminator(MaxSteps(30)),
            )
        });
        bench_optimizer!(group, "differential-evolution", n, {
            (
                Rosenbrock { n },
                DifferentialEvolution::new(Some(7)),
                start(n),
                DifferentialEvolutionConfig::default()
                    .with_population_size(24)
                    .unwrap(),
                Callbacks::empty().with_terminator(MaxSteps(40)),
            )
        });
        bench_optimizer!(group, "nelder-mead", n, {
            (
                Rosenbrock { n },
                NelderMead::default(),
                start(n),
                NelderMeadConfig::default(),
                Callbacks::empty().with_terminator(MaxSteps(80)),
            )
        });
        bench_optimizer!(group, "cmaes", n, {
            (
                Rosenbrock { n },
                CMAES::new(Some(7)),
                start(n),
                CMAESConfig::default().with_population_size(24).unwrap(),
                Callbacks::empty().with_terminator(MaxSteps(40)),
            )
        });
        bench_optimizer!(group, "simulated-annealing", n, {
            (
                Rosenbrock { n },
                SimulatedAnnealing::new(Some(7)),
                start(n),
                SimulatedAnnealingConfig::default(),
                Callbacks::empty().with_terminator(MaxSteps(400)),
            )
        });
        let bounds: Bounds = Bounds::new(vec![(-5.12, 5.12); n]).unwrap();
        bench_optimizer!(group, "pso", n, {
            (
                Rastrigin { n },
                PSO::new(Some(7)),
                Vector::zeros(n),
                PSOConfig::default()
                    .with_particles(24)
                    .unwrap()
                    .with_transform(bounds.clone()),
                Callbacks::empty().with_terminator(MaxSteps(40)),
            )
        });
    }
    group.finish();
}

fn benchmark_samplers(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms/sampling");
    for n in [2, 4] {
        bench_optimizer!(group, "aies", n, {
            (
                Rosenbrock { n },
                AIES::new(Some(7)),
                AIESInit::new(walkers(n)).unwrap(),
                AIESConfig::default(),
                Callbacks::empty().with_terminator(MaxSteps(30)),
            )
        });
        bench_optimizer!(group, "ess", n, {
            (
                Rosenbrock { n },
                ESS::new(Some(7)),
                ESSInit::new(walkers(n)).unwrap(),
                ESSConfig::default(),
                Callbacks::empty().with_terminator(MaxSteps(30)),
            )
        });
    }
    group.finish();
}

fn benchmark_line_searches(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms/line-search");
    let problem = Rosenbrock { n: 2 };
    let x: Vector = [-1.2, 1.0].into();
    let direction: Vector = [215.6, 88.0].into();
    for (name, kind) in [("backtracking", 0), ("more-thuente", 1), ("hager-zhang", 2)] {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut evals = EvalCounts::default();
                let _ = match kind {
                    0 => black_box(
                        BacktrackingLineSearch::default()
                            .search(&x, &direction, None, &problem, &(), &mut evals)
                            .unwrap(),
                    ),
                    1 => black_box(
                        MoreThuenteLineSearch::default()
                            .search(&x, &direction, None, &problem, &(), &mut evals)
                            .unwrap(),
                    ),
                    _ => black_box(
                        HagerZhangLineSearch::default()
                            .search(&x, &direction, None, &problem, &(), &mut evals)
                            .unwrap(),
                    ),
                };
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_optimizers,
    benchmark_samplers,
    benchmark_line_searches
);
criterion_main!(benches);
