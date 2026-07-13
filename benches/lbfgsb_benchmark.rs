use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ganesh::{
    algorithms::gradient::{LegacyLBFGSB, LegacyLBFGSBConfig},
    test_functions::rosenbrock::Rosenbrock,
    traits::Algorithm,
};

fn lbfgsb_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("LegacyLBFGSB");
    for n in [2, 3, 4, 5] {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, ndim| {
            b.iter_batched(
                || {
                    let problem = Rosenbrock { n: *ndim };
                    let solver = LegacyLBFGSB::default();
                    let init = ganesh::DVector::from_vec(vec![5.0; *ndim]);
                    let cfg = LegacyLBFGSBConfig::default();
                    let cbs = LegacyLBFGSB::default_callbacks();
                    (problem, solver, init, cfg, cbs)
                },
                |(problem, mut solver, init, cfg, cbs)| {
                    let result = solver.process(&problem, &(), init, cfg, cbs).unwrap();
                    black_box(result);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, lbfgsb_benchmark);
criterion_main!(benches);
