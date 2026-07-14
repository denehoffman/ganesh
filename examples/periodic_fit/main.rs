//! Fit a sinusoid whose phase crosses the displayed periodic seam.

use ganesh::{
    algorithms::gradient::{ConjugateGradient, ConjugateGradientConfig},
    core::MaxSteps,
    traits::{
        Algorithm, Bounds, CostFunction, Gradient, PeriodicTransform, ScaleTransform,
        SupportsParameterNames, Transform,
    },
    Vector,
};
use serde_json::json;
use std::{convert::Infallible, error::Error, fs::File, path::PathBuf};

const NAMES: [&str; 3] = ["offset", "amplitude", "phase"];
const TRUTH: [f64; 3] = [0.35, 1.7, -std::f64::consts::PI + 0.12];

struct SinusoidFit;

impl CostFunction<f64, ganesh::NalgebraProvider, Vec<(f64, f64)>> for SinusoidFit {
    fn evaluate(&self, parameters: &Vector, data: &Vec<(f64, f64)>) -> Result<f64, Infallible> {
        let offset = parameters.get(0);
        let amplitude = parameters.get(1);
        let phase = parameters.get(2);
        Ok(data
            .iter()
            .map(|&(time, observed)| {
                let residual = offset + amplitude * (time - phase).cos() - observed;
                residual * residual
            })
            .sum::<f64>()
            / data.len() as f64)
    }
}

impl Gradient<f64, ganesh::NalgebraProvider, Vec<(f64, f64)>> for SinusoidFit {
    fn gradient(&self, parameters: &Vector, data: &Vec<(f64, f64)>) -> Result<Vector, Infallible> {
        let offset = parameters.get(0);
        let amplitude = parameters.get(1);
        let phase = parameters.get(2);
        let mut gradient = [0.0; 3];
        for &(time, observed) in data {
            let angle = time - phase;
            let cosine = angle.cos();
            let residual = offset + amplitude * cosine - observed;
            gradient[0] += residual;
            gradient[1] += residual * cosine;
            gradient[2] += residual * amplitude * angle.sin();
        }
        let normalization = 2.0 / data.len() as f64;
        Ok(gradient.map(|value| normalization * value).into())
    }
}

fn circular_distance(lhs: f64, rhs: f64) -> f64 {
    (lhs - rhs + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

fn output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/periodic_fit/data.json")
}

fn main() -> Result<(), Box<dyn Error>> {
    let data: Vec<(f64, f64)> = (0..100)
        .map(|index| {
            let time = std::f64::consts::TAU * index as f64 / 100.0;
            let noise = 0.03 * (7.0 * time).sin() + 0.02 * (11.0 * time).cos();
            let observed = TRUTH[0] + TRUTH[1] * (time - TRUTH[2]).cos() + noise;
            (time, observed)
        })
        .collect();
    let initial = [0.0, 0.8, std::f64::consts::PI - 0.08];

    let scale = ScaleTransform::from_parameter_scales([0.5, 2.0, std::f64::consts::PI])?;
    let periodic = PeriodicTransform::new([
        None,
        None,
        Some((-std::f64::consts::PI, std::f64::consts::PI)),
    ])?;
    let bounds = Bounds::new([
        (f64::NEG_INFINITY, f64::INFINITY),
        (0.0, f64::INFINITY),
        (f64::NEG_INFINITY, f64::INFINITY),
    ])?;
    let transform = scale.then(periodic).then(bounds);
    let config: ConjugateGradientConfig = ConjugateGradientConfig::default()
        .with_parameter_names(NAMES)
        .with_transform(transform);
    let summary = ConjugateGradient::<f64>::default().process(
        &SinusoidFit,
        &data,
        initial,
        config,
        ConjugateGradient::<f64>::default_callbacks().with_terminator(MaxSteps(200)),
    )?;

    let phase_error = circular_distance(summary.x.get(2), TRUTH[2]).abs();
    assert!(phase_error < 0.03, "phase error was {phase_error}");
    assert!((summary.x.get(1) - TRUTH[1]).abs() < 0.03);

    let fitted: Vec<f64> = data
        .iter()
        .map(|&(time, _)| summary.x.get(0) + summary.x.get(1) * (time - summary.x.get(2)).cos())
        .collect();
    serde_json::to_writer_pretty(
        File::create(output_path())?,
        &json!({
            "title": "Mixed-domain periodic phase fit",
            "parameter_names": NAMES,
            "truth": TRUTH,
            "initial": initial,
            "fit": summary.x.to_vec(),
            "phase_error": phase_error,
            "time": data.iter().map(|point| point.0).collect::<Vec<_>>(),
            "observed": data.iter().map(|point| point.1).collect::<Vec<_>>(),
            "fitted": fitted,
        }),
    )?;
    println!("{summary}");
    println!("wrote {}", output_path().display());
    Ok(())
}
