use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ganesh::algorithms::NelderMead;
use ganesh::prelude::*;
use ganesh::test_functions::rosenbrock::Rosenbrock;

fn rosenbrock_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("nelder-mead: rosenbrock");
    // Standard can't handle n >= 15, but Adaptive does it quickly!
    for n in [2, 3, 4, 5, 10] {
        group.bench_with_input(BenchmarkId::new("Standard", n), &n, |b, ndim| {
            let problem = Rosenbrock { n: *ndim };
            let nm = NelderMead::default();
            let mut m = Minimizer::new(nm, *ndim).with_max_steps(10_000_000);
            let x0 = vec![5.0; *ndim];
            b.iter(|| {
                m.minimize(&problem, &x0, &mut ()).unwrap();
                black_box(&m.status);
            });
        });
        group.bench_with_input(BenchmarkId::new("Adaptive", n), &n, |b, ndim| {
            let problem = Rosenbrock { n: *ndim };
            let nm = NelderMead::default().with_adaptive(n);
            let mut m = Minimizer::new(nm, *ndim).with_max_steps(10_000_000);
            let x0 = vec![5.0; *ndim];
            b.iter(|| {
                m.minimize(&problem, &x0, &mut ()).unwrap();
                black_box(&m.status);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, rosenbrock_benchmark);
criterion_main!(benches);
