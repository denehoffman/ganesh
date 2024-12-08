use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ganesh::algorithms::BFGS;
use ganesh::test_functions::rosenbrock::Rosenbrock;
use ganesh::Minimizer;

fn bfgs_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("BFGS");
    for n in [2, 3, 4, 5] {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, ndim| {
            let problem = Rosenbrock { n: *ndim };
            let nm = BFGS::default();
            let mut m = Minimizer::new(&nm, *ndim).with_max_steps(10_000_000);
            let x0 = vec![5.0; *ndim];
            b.iter(|| {
                m.minimize(&problem, &x0, &mut ()).unwrap();
                black_box(&m.status);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bfgs_benchmark);
criterion_main!(benches);
