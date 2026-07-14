//! Sample a correlated five-dimensional Gaussian with ensemble slice sampling.

use fastrand::Rng;
use ganesh::{
    algorithms::mcmc::{ESSConfig, ESSInit, ESSMove, ESS},
    core::{Callbacks, MaxSteps},
    traits::{Algorithm, LogDensity},
    Matrix, NalgebraProvider, Vector,
};
use serde_json::json;
use std::{convert::Infallible, error::Error, fs::File, path::PathBuf};

struct CorrelatedGaussian;

impl LogDensity<f64, NalgebraProvider, Matrix> for CorrelatedGaussian {
    fn log_density(&self, x: &Vector, precision: &Matrix) -> Result<f64, Infallible> {
        Ok(-0.5 * x.dot(&precision.mul_vec(x)))
    }
}

fn normal(rng: &mut Rng, scale: f64) -> f64 {
    let radius = (-2.0 * rng.f64().max(f64::MIN_POSITIVE).ln()).sqrt();
    scale * radius * (std::f64::consts::TAU * rng.f64()).cos()
}

fn output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/multivariate_normal_ess/data.json")
}

fn main() -> Result<(), Box<dyn Error>> {
    const DIMENSION: usize = 5;
    let precision = Matrix::from_vec(
        DIMENSION,
        DIMENSION,
        (0..DIMENSION)
            .flat_map(|row| (0..DIMENSION).map(move |col| if row == col { 1.0 } else { 0.12 }))
            .collect(),
    );
    let covariance = precision
        .lu_inverse()
        .ok_or_else(|| std::io::Error::other("the demonstration matrix must be invertible"))?;
    let mut rng = Rng::with_seed(11);
    let walkers: Vec<Vector> = (0..80)
        .map(|_| {
            (0..DIMENSION)
                .map(|_| normal(&mut rng, 3.0))
                .collect::<Vec<_>>()
                .into()
        })
        .collect();
    let config: ESSConfig = ESSConfig::default().with_moves([
        ESSMove::gaussian(0.2),
        ESSMove::global(0.5),
        ESSMove::differential(0.3),
    ])?;
    let summary = ESS::new(Some(11)).process(
        &CorrelatedGaussian,
        &precision,
        ESSInit::new(walkers)?,
        config,
        Callbacks::empty().with_terminator(MaxSteps(1_000)),
    )?;
    let diagnostics = summary.diagnostics(Some(200), Some(2));
    let chains: Vec<Vec<Vec<f64>>> = summary
        .chain
        .iter()
        .map(|chain| chain.iter().map(Vector::to_vec).collect())
        .collect();
    let covariance: Vec<Vec<f64>> = (0..DIMENSION)
        .map(|row| (0..DIMENSION).map(|col| covariance.get(row, col)).collect())
        .collect();

    serde_json::to_writer_pretty(
        File::create(output_path())?,
        &json!({
            "title": "Correlated Gaussian ensemble slice sampling",
            "parameter_names": ["x₀", "x₁", "x₂", "x₃", "x₄"],
            "chains": chains,
            "burn": 200,
            "thin": 2,
            "target_covariance": covariance,
            "r_hat": diagnostics.r_hat.as_slice(),
            "effective_sample_size": diagnostics.ess.as_slice(),
        }),
    )?;
    println!("{summary}");
    println!("wrote {}", output_path().display());
    Ok(())
}
