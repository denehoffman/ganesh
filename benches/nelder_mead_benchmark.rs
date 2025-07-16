use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ganesh::algorithms::gradient_free::NelderMead;
use ganesh::core::{CtrlCAbortSignal, Engine};
use ganesh::test_functions::rosenbrock::Rosenbrock;

fn nelder_mead_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Nelder Mead");
    for n in [2, 3, 4, 5] {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, ndim| {
            let problem = Rosenbrock { n: *ndim };
            let mut m = Engine::new(NelderMead::default()).setup(|m| {
                m.with_abort_signal(CtrlCAbortSignal::new())
                    .with_max_steps(10_000_000)
                    .configure(|c| c.with_x0(vec![5.0; *ndim]))
            });
            b.iter(|| {
                m.process(&problem).unwrap();
                black_box(&m.status);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("Rosenbrock (adaptive)", n),
            &n,
            |b, ndim| {
                let problem = Rosenbrock { n: *ndim };
                let mut m = Engine::new(NelderMead::default()).setup(|e| {
                    e.configure(|c| c.with_adaptive(n).with_x0(vec![5.0; *ndim]))
                        .with_abort_signal(CtrlCAbortSignal::new())
                        .with_max_steps(10_000_000)
                });
                b.iter(|| {
                    m.process(&problem).unwrap();
                    black_box(&m.status);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, nelder_mead_benchmark);
criterion_main!(benches);
