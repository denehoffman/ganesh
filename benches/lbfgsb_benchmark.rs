use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ganesh::core::{CtrlCAbortSignal, Minimizer};
use ganesh::solvers::gradient::LBFGSB;
use ganesh::test_functions::rosenbrock::Rosenbrock;
use ganesh::traits::AbortSignal;

fn lbfgsb_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("LBFGSB");
    for n in [2, 3, 4, 5] {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, ndim| {
            let problem = Rosenbrock { n: *ndim };
            let nm = LBFGSB::default();
            let mut m = Minimizer::new(Box::new(nm)).setup(|m| {
                m.with_abort_signal(CtrlCAbortSignal::new().boxed())
                    .with_max_steps(10_000_000)
                    .on_status(|s| s.with_x0(vec![5.0; *ndim]))
            });
            b.iter(|| {
                m.minimize(&problem).unwrap();
                black_box(&m.status);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, lbfgsb_benchmark);
criterion_main!(benches);
