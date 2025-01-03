use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ganesh::algorithms::LBFGS;
use ganesh::test_functions::rosenbrock::Rosenbrock;
use ganesh::Minimizer;

fn lbfgs_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("LBFGS");
    for n in [2, 3, 4, 5] {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, ndim| {
            let problem = Rosenbrock { n: *ndim };
            let nm = LBFGS::default();
            let mut m = Minimizer::new(Box::new(nm), *ndim).with_max_steps(10_000_000);
            let x0 = vec![5.0; *ndim];
            b.iter(|| {
                m.minimize(&problem, &x0, &mut ()).unwrap();
                black_box(&m.status);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, lbfgs_benchmark);
criterion_main!(benches);
