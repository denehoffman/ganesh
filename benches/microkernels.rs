use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ganesh::{
    algorithms::line_search::HagerZhangLineSearch,
    core::transforms::Bounds,
    test_functions::Rosenbrock,
    traits::{LineSearch, Transform},
    DVector,
};

fn bench_hager_zhang_origin(c: &mut Criterion) {
    let mut group = c.benchmark_group("Micro/HagerZhang");
    for n in [4usize, 16] {
        group.bench_with_input(BenchmarkId::new("origin_eval", n), &n, |b, &ndim| {
            let x0 = DVector::from_element(ndim, 1.25);
            let p = DVector::from_element(ndim, -0.5);
            b.iter_batched(
                || {
                    (
                        HagerZhangLineSearch::default(),
                        Rosenbrock { n: ndim },
                        ganesh::algorithms::gradient::GradientStatus::default(),
                    )
                },
                |(mut ls, problem, mut status)| {
                    let _ = black_box(
                        ls.search(&x0, &p, Some(1.0), &problem, None, &(), &mut status)
                            .unwrap(),
                    );
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_bounds_jacobians(c: &mut Criterion) {
    let mut group = c.benchmark_group("Micro/BoundsTransform");
    for n in [4usize, 16] {
        let bounds = Bounds::from(vec![(-4.0, 4.0); n]);
        let x = DVector::from_element(n, 0.75);
        let z = bounds.to_internal(&x).into_owned();
        group.bench_with_input(BenchmarkId::new("to_external_jacobian", n), &n, |b, _| {
            b.iter(|| black_box(bounds.to_external_jacobian(&z)));
        });
        group.bench_with_input(BenchmarkId::new("to_internal_jacobian", n), &n, |b, _| {
            b.iter(|| black_box(bounds.to_internal_jacobian(&x)));
        });
        group.bench_with_input(
            BenchmarkId::new("to_internal_component_hessian", n),
            &n,
            |b, _| {
                b.iter(|| black_box(bounds.to_internal_component_hessian(0, &x)));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_hager_zhang_origin, bench_bounds_jacobians);
criterion_main!(benches);
