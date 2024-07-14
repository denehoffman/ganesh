use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ganesh::{algorithms::NelderMead, test_functions::Rosenbrock};
use ganesh::{algorithms::NelderMeadOptions, prelude::*};

fn rosenbrock_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("nelder-mead: rosenbrock");
    for n in [2, 3, 4, 5, 10, 15, 20, 30] {
        group.bench_with_input(BenchmarkId::new("Standard", n), &n, |b, ndim| {
            let x0 = vec![5.0; *ndim];
            b.iter(|| {
                let rb = Rosenbrock { n };
                let mut m = NelderMead::new(
                    rb,
                    &x0,
                    Some(NelderMeadOptions::builder().max_iters(1000000).build()),
                );
                black_box(minimize!(m).unwrap());
            });
        });
        group.bench_with_input(BenchmarkId::new("Adaptive", n), &n, |b, ndim| {
            let x0 = vec![5.0; *ndim];
            b.iter(|| {
                let rb = Rosenbrock { n };
                let mut m = NelderMead::new(
                    rb,
                    &x0,
                    Some(NelderMeadOptions::adaptive(n).max_iters(1000000).build()),
                );
                black_box(minimize!(m).unwrap());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, rosenbrock_benchmark);
criterion_main!(benches);
