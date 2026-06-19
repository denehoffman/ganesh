use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ganesh::{
    algorithms::{
        gradient::{GradientStatus, TrustRegion as CurrentTrustRegion, TrustRegionConfig},
        gradient_free::{
            DifferentialEvolution as CurrentDifferentialEvolution, DifferentialEvolutionConfig,
            DifferentialEvolutionInit, GradientFreeStatus,
        },
    },
    prototype::scalar::{
        CostFunction as GenericCostFunction, DifferentialEvolution as GenericDifferentialEvolution,
        DifferentialEvolutionConfig as GenericDifferentialEvolutionConfig,
        Gradient as GenericGradient, TrustRegion as GenericTrustRegion,
        TrustRegionConfig as GenericTrustRegionConfig,
    },
    traits::{Algorithm, CostFunction, Gradient},
    DMatrix, DVector, Float, LinearAlgebra, Matrix, NalgebraBackend, RealScalar, Vector,
};
use std::convert::Infallible;

const DIMS: [usize; 2] = [2, 8];
const TRUST_REGION_STEPS: usize = 32;
const DE_STEPS: usize = 80;
const DE_POPULATION: usize = 24;

struct Quadratic;

impl CostFunction for Quadratic {
    fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
        Ok(x.dot(x))
    }
}

impl Gradient for Quadratic {
    fn gradient(&self, x: &DVector<Float>, _args: &()) -> Result<DVector<Float>, Infallible> {
        Ok(x.scale(2.0))
    }

    fn hessian(&self, x: &DVector<Float>, _args: &()) -> Result<DMatrix<Float>, Infallible> {
        Ok(DMatrix::identity(x.len(), x.len()).scale(2.0))
    }
}

impl<T, B> GenericCostFunction<T, B> for Quadratic
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn evaluate(&self, x: &Vector<T, B>, _args: &()) -> Result<T, Infallible> {
        Ok(x.dot(x))
    }
}

impl<T, B> GenericGradient<T, B> for Quadratic
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn gradient(&self, x: &Vector<T, B>, _args: &()) -> Result<Vector<T, B>, Infallible> {
        Ok(x.scale(T::literal(2.0)))
    }

    fn hessian(&self, x: &Vector<T, B>, _args: &()) -> Result<Matrix<T, B>, Infallible> {
        Ok(Matrix::<T, B>::identity(x.len()).scale(T::literal(2.0)))
    }
}

fn start(n: usize) -> DVector<f64> {
    DVector::from_element(n, 5.0)
}

fn generic_start(n: usize) -> Vector<f64, NalgebraBackend> {
    Vector::from_vec(vec![5.0; n])
}

fn bench_trust_region(c: &mut Criterion) {
    let mut group = c.benchmark_group("PrototypeScalar/TrustRegion");
    for n in DIMS {
        group.bench_with_input(BenchmarkId::new("current_f64", n), &n, |b, &ndim| {
            b.iter_batched(
                || {
                    (
                        Quadratic,
                        CurrentTrustRegion::default(),
                        GradientStatus::default(),
                        start(ndim),
                        TrustRegionConfig::default(),
                    )
                },
                |(problem, mut solver, mut status, init, config)| {
                    solver
                        .initialize(&problem, &mut status, &(), &init, &config)
                        .unwrap();
                    for step in 0..TRUST_REGION_STEPS {
                        solver
                            .step(step, &problem, &mut status, &(), &config)
                            .unwrap();
                    }
                    black_box((solver, status));
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("generic_f64", n), &n, |b, &ndim| {
            b.iter_batched(
                || {
                    (
                        Quadratic,
                        GenericTrustRegion::<f64>::default(),
                        generic_start(ndim),
                        GenericTrustRegionConfig::default(),
                    )
                },
                |(problem, mut solver, init, config)| {
                    black_box(
                        solver
                            .run_steps(&problem, &(), init, config, TRUST_REGION_STEPS)
                            .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_differential_evolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("PrototypeScalar/DifferentialEvolution");
    for n in DIMS {
        group.bench_with_input(BenchmarkId::new("current_f64", n), &n, |b, &ndim| {
            b.iter_batched(
                || {
                    (
                        Quadratic,
                        CurrentDifferentialEvolution::new(Some(7)),
                        GradientFreeStatus::default(),
                        DifferentialEvolutionInit::new(start(ndim).as_slice()).unwrap(),
                        DifferentialEvolutionConfig::default()
                            .with_population_size(DE_POPULATION)
                            .unwrap(),
                    )
                },
                |(problem, mut solver, mut status, init, config)| {
                    solver
                        .initialize(&problem, &mut status, &(), &init, &config)
                        .unwrap();
                    for step in 0..DE_STEPS {
                        solver
                            .step(step, &problem, &mut status, &(), &config)
                            .unwrap();
                    }
                    black_box((solver, status));
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("generic_f64", n), &n, |b, &ndim| {
            b.iter_batched(
                || {
                    (
                        Quadratic,
                        GenericDifferentialEvolution::<f64>::new(Some(7)),
                        generic_start(ndim),
                        GenericDifferentialEvolutionConfig {
                            population_size: DE_POPULATION,
                            ..GenericDifferentialEvolutionConfig::default()
                        },
                    )
                },
                |(problem, mut solver, init, config)| {
                    black_box(
                        solver
                            .run_steps(&problem, &(), init, config, DE_STEPS)
                            .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_trust_region, bench_differential_evolution);
criterion_main!(benches);
