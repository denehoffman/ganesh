use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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
            let mut problem = Rosenbrock { n: *ndim };
            let mut solver = NelderMead::default();
            b.iter(|| {
                let result = solver
                    .process(
                        &mut problem,
                        &mut (),
                        NelderMeadConfig::default().with_x0(vec![5.0; *ndim]),
                        &[
                            NelderMeadFTerminator::default().build(),
                            NelderMeadXTerminator::default().build(),
                        ],
                    )
                    .unwrap();
                black_box(&result);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("Rosenbrock (adaptive)", n),
            &n,
            |b, ndim| {
                let mut problem = Rosenbrock { n: *ndim };
                let mut solver = NelderMead::default();
                b.iter(|| {
                    let result = solver
                        .process(
                            &mut problem,
                            &mut (),
                            NelderMeadConfig::default()
                                .with_x0(vec![5.0; *ndim])
                                .with_adaptive(n),
                            &[
                                NelderMeadFTerminator::default().build(),
                                NelderMeadXTerminator::default().build(),
                            ],
                        )
                        .unwrap();
                    black_box(&result);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, nelder_mead_benchmark);
criterion_main!(benches);
