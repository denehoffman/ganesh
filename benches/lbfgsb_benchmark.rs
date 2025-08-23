use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
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
            let mut problem = Rosenbrock { n: *ndim };
            let mut solver = LBFGSB::default();
            b.iter(|| {
                let result = solver
                    .process(
                        &mut problem,
                        &mut (),
                        LBFGSBConfig::default().with_x0(vec![5.0; *ndim]),
                        &[
                            LBFGSBFTerminator.build(),
                            LBFGSBGTerminator.build(),
                            LBFGSBInfNormGTerminator.build(),
                        ],
                    )
                    .unwrap();
                black_box(&result);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, lbfgsb_benchmark);
criterion_main!(benches);
