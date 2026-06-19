use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ganesh::{
    algorithms::gradient_free::{
        nelder_mead::{NelderMeadConfig, NelderMeadInit, SimplexConstructionMethod},
        NelderMead,
    },
    test_functions::rosenbrock::Rosenbrock,
    traits::Algorithm,
};

fn nelder_mead_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Nelder Mead");
    for n in [2, 3, 4, 5] {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, ndim| {
            b.iter_batched(
                || {
                    let problem = Rosenbrock { n: *ndim };
                    let solver = NelderMead::default();
                    let init = NelderMeadInit::new_with_method(
                        SimplexConstructionMethod::orthogonal(vec![5.0; *ndim]),
                    );
                    let cfg = NelderMeadConfig::default();
                    let cbs = NelderMead::default_callbacks();
                    (problem, solver, init, cfg, cbs)
                },
                |(problem, mut solver, init, cfg, cbs)| {
                    let result = solver.process(&problem, &(), init, cfg, cbs).unwrap();
                    black_box(result);
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(
            BenchmarkId::new("Rosenbrock (adaptive)", n),
            &n,
            |b, ndim| {
                b.iter_batched(
                    || {
                        let problem = Rosenbrock { n: *ndim };
                        let solver = NelderMead::default();
                        let init = NelderMeadInit::new_with_method(
                            SimplexConstructionMethod::orthogonal(vec![5.0; *ndim]),
                        );
                        let cfg = NelderMeadConfig::default().with_adaptive(*ndim).unwrap();
                        let cbs = NelderMead::default_callbacks();
                        (problem, solver, init, cfg, cbs)
                    },
                    |(problem, mut solver, init, cfg, cbs)| {
                        let result = solver.process(&problem, &(), init, cfg, cbs).unwrap();
                        black_box(result);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, nelder_mead_benchmark);
criterion_main!(benches);
