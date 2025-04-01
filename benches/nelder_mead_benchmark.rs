use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ganesh::algorithms::NelderMead;
use ganesh::test_functions::rosenbrock::Rosenbrock;
use ganesh::Minimizer;

fn nelder_mead_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Nelder Mead");
    for n in [2, 3, 4, 5] {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, ndim| {
            let mut problem = Rosenbrock { n: *ndim };
            let nm = NelderMead::default();
            let mut m = Minimizer::new(Box::new(nm), *ndim).with_max_steps(10_000_000);
            let x0 = vec![5.0; *ndim];
            b.iter(|| {
                m.minimize(&mut problem, &x0, &mut ()).unwrap();
                black_box(&m.status);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("Rosenbrock (adaptive)", n),
            &n,
            |b, ndim| {
                let mut problem = Rosenbrock { n: *ndim };
                let nm = NelderMead::default().with_adaptive(n);
                let mut m = Minimizer::new(Box::new(nm), *ndim).with_max_steps(10_000_000);
                let x0 = vec![5.0; *ndim];
                b.iter(|| {
                    m.minimize(&mut problem, &x0, &mut ()).unwrap();
                    black_box(&m.status);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, nelder_mead_benchmark);
criterion_main!(benches);
