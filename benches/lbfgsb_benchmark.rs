use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use ganesh::{
    algorithms::gradient::{
        lbfgsb::{LBFGSBConfig, LBFGSBFTerminator, LBFGSBGTerminator, LBFGSBInfNormGTerminator},
        LBFGSB,
    },
    test_functions::rosenbrock::Rosenbrock,
    traits::{Algorithm, Callback},
};

fn lbfgsb_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("LBFGSB");
    for n in [2, 3, 4, 5] {
        group.bench_with_input(BenchmarkId::new("Rosenbrock", n), &n, |b, ndim| {
            let base_cfg = LBFGSBConfig::default().with_x0(vec![5.0; *ndim]);
            let terms = vec![
                LBFGSBFTerminator.build(),
                LBFGSBGTerminator.build(),
                LBFGSBInfNormGTerminator.build(),
            ];

            b.iter_batched(
                || {
                    let problem = Rosenbrock { n: *ndim };
                    let solver = LBFGSB::default();
                    let cfg = base_cfg.clone();
                    (problem, solver, cfg)
                },
                |(mut problem, mut solver, cfg)| {
                    let result = solver.process(&mut problem, &mut (), cfg, &terms).unwrap();
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
