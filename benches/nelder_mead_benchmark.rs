use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ganesh::{
    algorithms::gradient_free::{
        nelder_mead::{NelderMeadConfig, NelderMeadFTerminator, NelderMeadXTerminator},
        NelderMead,
    },
    test_functions::rosenbrock::Rosenbrock,
    traits::{Algorithm, Callback},
};

fn nelder_mead_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Nelder Mead");
    for n in [2, 3, 4, 5] {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, ndim| {
            let base_cfg = NelderMeadConfig::default().with_x0(vec![5.0; *ndim]);
            let terms = vec![
                NelderMeadFTerminator::default().build(),
                NelderMeadXTerminator::default().build(),
            ];

            b.iter_batched(
                || {
                    let problem = Rosenbrock { n: *ndim };
                    let solver = NelderMead::default();
                    let cfg = base_cfg.clone();
                    (problem, solver, cfg)
                },
                |(mut problem, mut solver, cfg)| {
                    let result = solver.process(&mut problem, &mut (), cfg, &terms).unwrap();
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
                let terms = vec![
                    NelderMeadFTerminator::default().build(),
                    NelderMeadXTerminator::default().build(),
                ];

                b.iter_batched(
                    || {
                        let problem = Rosenbrock { n: *ndim };
                        let solver = NelderMead::default();
                        let cfg = base_cfg.clone();
                        (problem, solver, cfg)
                    },
                    |(mut problem, mut solver, cfg)| {
                        let result = solver.process(&mut problem, &mut (), cfg, &terms).unwrap();
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
