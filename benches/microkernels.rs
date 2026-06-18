use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ganesh::{
    algorithms::line_search::HagerZhangLineSearch,
    algorithms::{gradient::GradientStatus, gradient_free::GradientFreeStatus},
    core::transforms::Bounds,
    test_functions::Rosenbrock,
    traits::{LineSearch, ProgressStatus, Status, Transform},
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

fn bench_status_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("Micro/StatusUpdates");
    for n in [4usize, 16] {
        group.bench_with_input(
            BenchmarkId::new("gradient_position_structured", n),
            &n,
            |b, &ndim| {
                let x = DVector::from_element(ndim, 1.25);
                b.iter_batched(
                    GradientStatus::default,
                    |mut status| {
                        status.set_position((black_box(x.clone()), black_box(3.5)));
                        black_box(status);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gradient_position_formatted_legacy_pattern", n),
            &n,
            |b, &ndim| {
                let x = DVector::from_element(ndim, 1.25);
                b.iter_batched(
                    GradientStatus::default,
                    |mut status| {
                        let fx = black_box(3.5);
                        let message = format!("f(x) = {fx}");
                        status.set_message().step_with_message(&message);
                        status.x = black_box(x.clone());
                        status.fx = fx;
                        black_box(status);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gradient_free_position_structured", n),
            &n,
            |b, &ndim| {
                let x = DVector::from_element(ndim, 1.25);
                b.iter_batched(
                    GradientFreeStatus::default,
                    |mut status| {
                        status.set_position((black_box(x.clone()), black_box(3.5)));
                        black_box(status);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.bench_function("gradient_eval_counters", |b| {
        b.iter_batched(
            GradientStatus::default,
            |mut status| {
                status.inc_n_f_evals();
                status.inc_n_g_evals();
                status.inc_n_h_evals();
                black_box(status);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("gradient_progress_render", |b| {
        let mut status = GradientStatus::default();
        status.set_position((DVector::from_element(8, 1.25), 3.5));
        status.inc_n_f_evals();
        let mut out = String::new();
        b.iter(|| {
            out.clear();
            status.write_progress(black_box(&mut out)).unwrap();
            black_box(&out);
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_hager_zhang_origin,
    bench_bounds_jacobians,
    bench_status_updates
);
criterion_main!(benches);
