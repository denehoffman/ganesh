//! Find the global basin of the two-dimensional Rastrigin function with deterministic restarts.

use fastrand::Rng;
use ganesh::{
    algorithms::gradient_free::{GradientFreeStatus, NelderMead, NelderMeadConfig},
    core::{minimize_multistart, restart_seed, FixedRestarts, MaxSteps, MultiStartState},
    traits::{Algorithm, Bounds, CostFunction, SupportsParameterNames},
    NalgebraProvider, Vector,
};
use serde_json::json;
use std::{convert::Infallible, error::Error, fs::File, path::PathBuf};

const LOWER: f64 = -5.12;
const UPPER: f64 = 5.12;
const RUNS: usize = 24;
const BASE_SEED: u64 = 552;

struct Rastrigin;

impl CostFunction for Rastrigin {
    fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, Infallible> {
        Ok(20.0
            + (0..2)
                .map(|index| {
                    let value = x.get(index);
                    value * value - 10.0 * (std::f64::consts::TAU * value).cos()
                })
                .sum::<f64>())
    }
}

fn output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/multistart/data.json")
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut factory = |run_index: usize, _state: &MultiStartState| {
        let mut rng = Rng::with_seed(restart_seed(BASE_SEED, run_index));
        let start = Vector::from_vec(
            (0..2)
                .map(|_| LOWER + (UPPER - LOWER) * rng.f64())
                .collect(),
        );
        let config = NelderMeadConfig::default()
            .with_parameter_names(["x", "y"])
            .with_transform(Bounds::new([(LOWER, UPPER), (LOWER, UPPER)]).unwrap());
        (
            NelderMead::default(),
            start,
            config,
            NelderMead::default_callbacks().with_terminator(MaxSteps(800)),
        )
    };
    let mut policy = FixedRestarts::new(RUNS);
    let summary = minimize_multistart::<
        Rastrigin,
        (),
        Infallible,
        NelderMead,
        GradientFreeStatus,
        _,
        _,
        f64,
        NalgebraProvider,
    >(&Rastrigin, &(), &mut factory, &mut policy)?;

    assert_eq!(summary.completed_runs(), RUNS);
    assert_eq!(summary.best_run_index, Some(18));
    let nearest_start_distance = summary
        .runs
        .iter()
        .map(|run| run.x0.norm())
        .fold(f64::INFINITY, f64::min);
    assert!(nearest_start_distance > 0.4);
    assert!(
        summary.best().unwrap().fx < 1e-3,
        "best objective was {}",
        summary.best().unwrap().fx
    );
    serde_json::to_writer_pretty(
        File::create(output_path())?,
        &json!({
            "title": "Deterministic multistart Rastrigin minimization",
            "bounds": [LOWER, UPPER],
            "starts": summary.runs.iter().map(|run| run.x0.to_vec()).collect::<Vec<_>>(),
            "endpoints": summary.runs.iter().map(|run| run.x.to_vec()).collect::<Vec<_>>(),
            "values": summary.runs.iter().map(|run| run.fx).collect::<Vec<_>>(),
            "best_run_index": summary.best_run_index,
        }),
    )?;
    println!("{summary}");
    println!("\nBest run\n{}", summary.best().unwrap());
    println!("wrote {}", output_path().display());
    Ok(())
}
