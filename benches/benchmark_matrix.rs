use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ganesh::{
    algorithms::{
        gradient::{
            Adam, AdamConfig, ConjugateGradient, ConjugateGradientConfig, LBFGSBConfig,
            LBFGSBErrorMode, LBFGSBGTerminator, LBFGSBInfNormGTerminator, TrustRegion,
            TrustRegionConfig, LBFGSB,
        },
        gradient_free::{
            CMAESConfig, DifferentialEvolution, DifferentialEvolutionConfig, NelderMead,
            NelderMeadConfig, SimulatedAnnealing, SimulatedAnnealingConfig, CMAES,
        },
        line_search::{
            BacktrackingLineSearch, HagerZhangLineSearch, MoreThuenteLineSearch,
            StrongWolfeLineSearch,
        },
        mcmc::{AIESConfig, AIESInit, ESSConfig, ESSInit, AIES, ESS},
        particles::{PSOConfig, PSO},
    },
    core::{Callbacks, EvalCounts, MaxSteps},
    test_functions::{rastrigin::Rastrigin, rosenbrock::Rosenbrock},
    traits::{Algorithm, CostFunction, Gradient, LineSearch},
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

// A data-fitting objective whose fused override shares the residual calculation.
// Keep it separate from the small Rosenbrock matrix above so both workloads remain comparable.
struct LeastSquares {
    features: Vec<[f64; 8]>,
    targets: Vec<f64>,
    fused: bool,
}

impl LeastSquares {
    fn new(fused: bool) -> Self {
        let features: Vec<[f64; 8]> = (0..4096)
            .map(|row| {
                std::array::from_fn(|col| ((row + 1) as f64 * (col + 1) as f64 * 0.17).sin())
            })
            .collect();
        let targets = features
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(col, a)| a * (col + 1) as f64 * 0.1)
                    .sum()
            })
            .collect();
        Self {
            features,
            targets,
            fused,
        }
    }

    fn residual(row: &[f64; 8], target: f64, x: &Vector) -> f64 {
        row.iter()
            .enumerate()
            .map(|(col, a)| a * x.get(col))
            .sum::<f64>()
            - target
    }
}

impl CostFunction for LeastSquares {
    fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, std::convert::Infallible> {
        Ok(self
            .features
            .iter()
            .zip(&self.targets)
            .map(|(row, target)| {
                let residual = Self::residual(row, *target, x);
                0.5 * residual * residual
            })
            .sum())
    }
}

impl Gradient for LeastSquares {
    fn gradient(&self, x: &Vector, _: &()) -> Result<Vector, std::convert::Infallible> {
        let mut gradient = [0.0; 8];
        for (row, target) in self.features.iter().zip(&self.targets) {
            let residual = Self::residual(row, *target, x);
            for (entry, a) in gradient.iter_mut().zip(row) {
                *entry += residual * a;
            }
        }
        Ok(Vector::from_vec(gradient.to_vec()))
    }

    fn evaluate_with_gradient(
        &self,
        x: &Vector,
        args: &(),
    ) -> Result<(f64, Vector), std::convert::Infallible> {
        if !self.fused {
            return Ok((self.evaluate(x, args)?, self.gradient(x, args)?));
        }
        let mut fx = 0.0;
        let mut gradient = [0.0; 8];
        for (row, target) in self.features.iter().zip(&self.targets) {
            let residual = Self::residual(row, *target, x);
            fx += 0.5 * residual * residual;
            for (entry, a) in gradient.iter_mut().zip(row) {
                *entry += residual * a;
            }
        }
        Ok((fx, Vector::from_vec(gradient.to_vec())))
    }
}

fn benchmark_lbfgsb_evaluations(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms/lbfgsb-evaluations");
    for (name, fused, search) in [
        (
            "least-squares/separate/more-thuente",
            false,
            StrongWolfeLineSearch::default(),
        ),
        (
            "least-squares/fused/more-thuente",
            true,
            StrongWolfeLineSearch::default(),
        ),
        (
            "least-squares/fused/hager-zhang",
            true,
            StrongWolfeLineSearch::from(HagerZhangLineSearch::default()),
        ),
    ] {
        let problem = LeastSquares::new(fused);
        let setup = || {
            (
                LBFGSB::<f64>::default(),
                start(8),
                LBFGSBConfig::<f64>::default()
                    .with_max_step(1.0)
                    .unwrap()
                    .with_error_mode(LBFGSBErrorMode::Skip)
                    .with_line_search(search.clone()),
                LBFGSB::<f64>::default_callbacks().with_terminator(MaxSteps(60)),
            )
        };
        let (mut algorithm, init, config, callbacks) = setup();
        let result = algorithm
            .process(&problem, &(), init, config, callbacks)
            .unwrap();
        assert!(result.fx < 1e-10, "{name} did not converge: {}", result.fx);
        eprintln!(
            "{name}: fx={:.16e}, f={}, g={}",
            result.fx,
            result.evals.f(),
            result.evals.g()
        );
        group.bench_function(name, |b| {
            b.iter_batched(
                &setup,
                |(mut algorithm, init, config, callbacks)| {
                    black_box(
                        algorithm
                            .process(&problem, &(), init, config, callbacks)
                            .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// Analytic, moderately conditioned objectives isolate optimizer work at larger dimensions.
struct DiagonalQuadratic;

impl DiagonalQuadratic {
    fn weight(index: usize) -> f64 {
        1.0 + (index % 32) as f64
    }
}

impl CostFunction for DiagonalQuadratic {
    fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, std::convert::Infallible> {
        Ok((0..x.len())
            .map(|index| 0.5 * Self::weight(index) * x.get(index).powi(2))
            .sum())
    }
}

impl Gradient for DiagonalQuadratic {
    fn gradient(&self, x: &Vector, _: &()) -> Result<Vector, std::convert::Infallible> {
        Ok(Vector::from_vec(
            (0..x.len())
                .map(|index| Self::weight(index) * x.get(index))
                .collect(),
        ))
    }
}

fn benchmark_lbfgsb_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms/lbfgsb-paths");
    for (name, n, bound_kind) in [
        ("unconstrained", 64, 0),
        ("unconstrained", 512, 0),
        ("explicit-unbounded", 64, 1),
        ("bounded-control", 64, 2),
        ("bounded-control", 512, 2),
    ] {
        let setup = || {
            let config = LBFGSBConfig::<f64>::default()
                .with_max_step(100.0)
                .unwrap()
                .with_error_mode(LBFGSBErrorMode::Skip);
            let config = match bound_kind {
                1 => config
                    .with_bounds(vec![(f64::NEG_INFINITY, f64::INFINITY); n])
                    .unwrap(),
                2 => config.with_bounds(vec![(-5.0, 5.0); n]).unwrap(),
                _ => config,
            };
            (
                LBFGSB::<f64>::default(),
                Vector::from_vec(vec![1.0; n]),
                config,
                Callbacks::empty()
                    .with_terminator(LBFGSBGTerminator::new(1e-6).unwrap())
                    .with_terminator(MaxSteps(500)),
            )
        };
        let (mut algorithm, init, config, callbacks) = setup();
        let result = algorithm
            .process(&DiagonalQuadratic, &(), init, config, callbacks)
            .unwrap();
        let norm = DiagonalQuadratic.gradient(&result.x, &()).unwrap().norm();
        assert!(
            norm < 1e-6 && result.fx < 1e-10,
            "{name}/{n} did not converge: fx={}, |g|={norm}",
            result.fx
        );
        eprintln!(
            "{name}/{n}: fx={:.16e}, |g|={norm:.16e}, f={}, g={}",
            result.fx,
            result.evals.f(),
            result.evals.g()
        );
        group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
            b.iter_batched(
                &setup,
                |(mut algorithm, init, config, callbacks)| {
                    black_box(
                        algorithm
                            .process(&DiagonalQuadratic, &(), init, config, callbacks)
                            .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn benchmark_lbfgsb_active_bounds(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms/lbfgsb-active");
    let n = 512;
    for (name, active) in [("sparse", 8), ("dense", 384)] {
        let setup = || {
            (
                LBFGSB::<f64>::default(),
                Vector::from_vec(vec![1.0; n]),
                LBFGSBConfig::<f64>::default()
                    .with_max_step(100.0)
                    .unwrap()
                    .with_error_mode(LBFGSBErrorMode::Skip)
                    .with_bounds(
                        (0..n)
                            .map(|i| (if i < active { 0.25 } else { -5.0 }, 5.0))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
                Callbacks::empty()
                    .with_terminator(LBFGSBInfNormGTerminator::new(1e-6).unwrap())
                    .with_terminator(MaxSteps(500)),
            )
        };
        let (mut algorithm, init, config, callbacks) = setup();
        let result = algorithm
            .process(&DiagonalQuadratic, &(), init, config, callbacks)
            .unwrap();
        let optimum: f64 = (0..active)
            .map(|i| 0.5 * DiagonalQuadratic::weight(i) * 0.25_f64.powi(2))
            .sum();
        assert!(
            result
                .message
                .to_string()
                .contains("PROJECTED GRADIENT WITHIN TOLERANCE"),
            "{name}: {}",
            result.message
        );
        let gradient = DiagonalQuadratic.gradient(&result.x, &()).unwrap();
        let mut projected_inf = 0.0_f64;
        for i in 0..n {
            let x = result.x.get(i);
            let lower = if i < active { 0.25 } else { -5.0 };
            assert!(x >= lower && x <= 5.0);
            let g = gradient.get(i);
            let projected = if (x == lower && g > 0.0) || (x == 5.0 && g < 0.0) {
                0.0
            } else {
                g
            };
            projected_inf = projected_inf.max(projected.abs());
        }
        let gap = (result.fx - optimum).abs();
        assert!(
            projected_inf < 1e-6 && gap < 1e-9,
            "{name}: gap={gap}, projected={projected_inf}"
        );
        eprintln!("active-{name}/{n}: fx={:.16e}, gap={gap:.16e}, projected_inf={projected_inf:.16e}, f={}, g={}", result.fx, result.evals.f(), result.evals.g());
        group.bench_with_input(BenchmarkId::new(name, n), &n, |b, _| {
            b.iter_batched(
                &setup,
                |(mut algorithm, init, config, callbacks)| {
                    black_box(
                        algorithm
                            .process(&DiagonalQuadratic, &(), init, config, callbacks)
                            .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// Objective units can make the first steepest-descent direction need a step above one.
struct SmallScaleQuadratic;

impl CostFunction for SmallScaleQuadratic {
    fn evaluate(&self, x: &Vector, args: &()) -> Result<f64, std::convert::Infallible> {
        Ok(1e-3 * DiagonalQuadratic.evaluate(x, args)?)
    }
}

impl Gradient for SmallScaleQuadratic {
    fn gradient(&self, x: &Vector, args: &()) -> Result<Vector, std::convert::Infallible> {
        Ok(DiagonalQuadratic.gradient(x, args)?.scale(1e-3))
    }
}

fn benchmark_lbfgsb_step_policy(c: &mut Criterion) {
    // Record the accuracy of the unchanged fixed-step Rosenbrock benchmarks as well.
    for n in [2, 8] {
        let problem = Rosenbrock { n };
        let result = LBFGSB::<f64>::default()
            .process(
                &problem,
                &(),
                start(n),
                LBFGSBConfig::default(),
                Callbacks::empty().with_terminator(MaxSteps(60)),
            )
            .unwrap();
        let norm = problem.gradient(&result.x, &()).unwrap().norm();
        eprintln!(
            "rosenbrock-fixed/{n}: fx={:.16e}, |g|={norm:.16e}, f={}, g={}",
            result.fx,
            result.evals.f(),
            result.evals.g()
        );
    }

    let mut group = c.benchmark_group("algorithms/lbfgsb-steps");
    for n in [64, 512] {
        let setup = || {
            (
                LBFGSB::<f64>::default(),
                Vector::from_vec(vec![1.0; n]),
                LBFGSBConfig::<f64>::default().with_error_mode(LBFGSBErrorMode::Skip),
                Callbacks::empty()
                    .with_terminator(LBFGSBGTerminator::new(1e-8).unwrap())
                    .with_terminator(MaxSteps(500)),
            )
        };
        let (mut algorithm, init, config, callbacks) = setup();
        let result = algorithm
            .process(&SmallScaleQuadratic, &(), init, config, callbacks)
            .unwrap();
        let norm = SmallScaleQuadratic.gradient(&result.x, &()).unwrap().norm();
        assert!(result.message.to_string().contains("GRADIENT CONVERGED"));
        assert!(norm < 1e-8 && result.fx < 1e-12 && result.x.norm() < 1e-5);
        eprintln!(
            "small-scale/{n}: fx={:.16e}, |g|={norm:.16e}, f={}, g={}",
            result.fx,
            result.evals.f(),
            result.evals.g()
        );
        group.bench_with_input(BenchmarkId::new("small-scale", n), &n, |b, _| {
            b.iter_batched(
                &setup,
                |(mut algorithm, init, config, callbacks)| {
                    black_box(
                        algorithm
                            .process(&SmallScaleQuadratic, &(), init, config, callbacks)
                            .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
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
    benchmark_lbfgsb_evaluations,
    benchmark_lbfgsb_paths,
    benchmark_lbfgsb_active_bounds,
    benchmark_lbfgsb_step_policy,
    benchmark_samplers,
    benchmark_line_searches
);
criterion_main!(benches);
