use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ganesh::{
    algorithms::gradient_free::{nelder_mead::NelderMeadConfig, NelderMead},
    test_functions::rosenbrock::Rosenbrock,
    traits::Algorithm,
};

fn nelder_mead_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Nelder Mead");
    for n in [2, 3, 4, 5] {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, ndim| {
            let base_cfg = NelderMeadConfig::default().with_x0(vec![5.0; *ndim]);
            b.iter_batched(
                || {
                    let problem = Rosenbrock { n: *ndim };
                    let solver = NelderMead::default();
                    let cbs = NelderMead::default_callbacks();
                    (problem, solver, base_cfg.clone(), cbs)
                },
                |(mut problem, mut solver, cfg, cbs)| {
                    let result = solver.process(&mut problem, &mut (), cfg, cbs).unwrap();
                    black_box(result);
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_with_input(
            BenchmarkId::new("Rosenbrock (adaptive)", n),
            &n,
            |b, ndim| {
                let base_cfg = NelderMeadConfig::default()
                    .with_x0(vec![5.0; *ndim])
                    .with_adaptive(*ndim);
                b.iter_batched(
                    || {
                        let problem = Rosenbrock { n: *ndim };
                        let solver = NelderMead::default();
                        let cbs = NelderMead::default_callbacks();
                        (problem, solver, base_cfg.clone(), cbs)
                    },
                    |(mut problem, mut solver, cfg, cbs)| {
                        let result = solver.process(&mut problem, &mut (), cfg, cbs).unwrap();
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
