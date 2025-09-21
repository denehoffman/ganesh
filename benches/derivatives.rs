use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::convert::Infallible;
use std::time::Duration;

use ganesh::{
    traits::{CostFunction, Gradient},
    DVector, Float,
};

struct Rosenbrock;
impl CostFunction for Rosenbrock {
    fn evaluate(&self, x: &DVector<Float>, _: &()) -> Result<Float, Infallible> {
        let mut s = 0.0 as Float;
        for i in 0..(x.len() - 1) {
            let xi = x[i];
            let xi1 = x[i + 1];
            let t1 = 1.0 - xi;
            let t2 = xi1 - xi * xi;
            s += t1 * t1 + 100.0 * t2 * t2;
        }
        Ok(s)
    }
}
impl Gradient for Rosenbrock {}

fn random_x_with(rng: &mut fastrand::Rng, n: usize) -> DVector<Float> {
    DVector::from_fn(n, |_, _| rng.f64() as Float * 4.0 - 2.0)
}

fn bench_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("derivatives");
    for &n in &[16usize, 64, 256] {
        group.bench_with_input(BenchmarkId::new("gradient", n), &n, |b, &n| {
            let f = Rosenbrock;
            let mut rng = fastrand::Rng::with_seed(0);
            b.iter_batched(
                || random_x_with(&mut rng, n),
                |x| {
                    let g = f.gradient(&x, &()).unwrap();
                    black_box(g);
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("hessian", n), &n, |b, &n| {
            let f = Rosenbrock;
            let mut rng = fastrand::Rng::with_seed(0);
            b.iter_batched(
                || random_x_with(&mut rng, n),
                |x| {
                    let h = f.hessian(&x, &()).unwrap();
                    black_box(h);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn custom_criterion() -> Criterion {
    Criterion::default()
        .configure_from_args()
        .warm_up_time(Duration::from_secs(10))
        .measurement_time(Duration::from_secs(20))
        .sample_size(150)
        .noise_threshold(0.01)
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = bench_derivatives
}
criterion_main!(benches);
