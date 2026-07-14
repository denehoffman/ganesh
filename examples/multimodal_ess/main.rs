//! Sample the four modes of Himmelblau's function with ensemble slice sampling.

use fastrand::Rng;
use ganesh::{
    algorithms::mcmc::{AutocorrelationTerminator, ESSConfig, ESSInit, ESSMove, ESS},
    core::{Callbacks, MaxSteps},
    traits::{Algorithm, LogDensity},
    Vector,
};
use serde_json::json;
use std::{convert::Infallible, error::Error, fs::File, path::PathBuf};

struct Himmelblau;

impl LogDensity for Himmelblau {
    fn log_density(&self, x: &Vector, _: &()) -> Result<f64, Infallible> {
        let a = x.get(0).powi(2) + x.get(1) - 11.0;
        let b = x.get(0) + x.get(1).powi(2) - 7.0;
        Ok(-(a.mul_add(a, b * b)) / 8.0)
    }
}

fn normal(rng: &mut Rng, scale: f64) -> f64 {
    let radius = (-2.0 * rng.f64().max(f64::MIN_POSITIVE).ln()).sqrt();
    scale * radius * (std::f64::consts::TAU * rng.f64()).cos()
}

fn output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/multimodal_ess/data.json")
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = Rng::with_seed(7);
    let walkers: Vec<Vector> = (0..64)
        .map(|_| [normal(&mut rng, 6.0), normal(&mut rng, 6.0)].into())
        .collect();

    let config: ESSConfig = ESSConfig::default().with_moves([
        ESSMove::gaussian(0.15),
        ESSMove::custom_global(0.70, None, Some(0.35), Some(4))?,
        ESSMove::differential(0.15),
    ])?;
    let summary = ESS::new(Some(7)).process(
        &Himmelblau,
        &(),
        ESSInit::new(walkers)?,
        config,
        Callbacks::empty()
            .with_terminator(MaxSteps(1_500))
            .with_terminator(AutocorrelationTerminator::default()),
    )?;
    let diagnostics = summary.diagnostics(Some(300), Some(2));
    let chains: Vec<Vec<Vec<f64>>> = summary
        .chain
        .iter()
        .map(|chain| chain.iter().map(Vector::to_vec).collect())
        .collect();

    serde_json::to_writer_pretty(
        File::create(output_path())?,
        &json!({
            "title": "Himmelblau ensemble slice sampling",
            "parameter_names": ["x", "y"],
            "chains": chains,
            "burn": 300,
            "thin": 2,
            "r_hat": diagnostics.r_hat.as_slice(),
            "effective_sample_size": diagnostics.ess.as_slice(),
            "acceptance_rate": diagnostics.mean_acceptance_rate,
        }),
    )?;
    println!("{summary}");
    println!("wrote {}", output_path().display());
    Ok(())
}
