//! Fit and sample a bivariate Gaussian with a smooth positive-definite transform.

use fastrand::Rng;
use ganesh::{
    algorithms::{
        gradient::{ConjugateGradient, ConjugateGradientConfig},
        gradient_free::{NelderMead, NelderMeadConfig},
        mcmc::{ESSConfig, ESSInit, ESSMove, ESS},
    },
    core::{Callbacks, MaxSteps},
    traits::{Algorithm, CostFunction, Gradient, LogDensity, SupportsParameterNames, Transform},
    Matrix, NalgebraProvider, Vector,
};
use serde_json::json;
use std::{convert::Infallible, error::Error, fs::File, path::PathBuf};

const NAMES: [&str; 5] = ["μ₀", "μ₁", "Σ₀₀", "Σ₀₁", "Σ₁₁"];

struct GaussianFit;

impl CostFunction<f64, NalgebraProvider, Vec<Vector>> for GaussianFit {
    fn evaluate(&self, x: &Vector, data: &Vec<Vector>) -> Result<f64, Infallible> {
        let (s00, s01, s11) = (x.get(2), x.get(3), x.get(4));
        let determinant = s00.mul_add(s11, -s01 * s01);
        if determinant <= f64::EPSILON {
            return Ok(f64::INFINITY);
        }
        let quadratic = data.iter().fold(0.0, |sum, datum| {
            let dx = datum.get(0) - x.get(0);
            let dy = datum.get(1) - x.get(1);
            sum + (s11 * dx * dx - 2.0 * s01 * dx * dy + s00 * dy * dy) / determinant
        });
        Ok((data.len() as f64).mul_add(determinant.ln(), quadratic))
    }
}

impl Gradient<f64, NalgebraProvider, Vec<Vector>> for GaussianFit {}

impl LogDensity<f64, NalgebraProvider, Vec<Vector>> for GaussianFit {
    fn log_density(&self, x: &Vector, data: &Vec<Vector>) -> Result<f64, Infallible> {
        Ok(-0.5 * self.evaluate(x, data)?)
    }
}

#[derive(Clone, Copy)]
struct PositiveDefinite;

impl Transform<f64, NalgebraProvider> for PositiveDefinite {
    fn to_external(&self, internal: &Vector) -> Vector {
        let mut external = internal.clone();
        let (l00, l10, l11) = (internal.get(2), internal.get(3), internal.get(4));
        external.set(2, l00 * l00);
        external.set(3, l00 * l10);
        external.set(4, l10.mul_add(l10, l11 * l11));
        external
    }

    fn to_internal(&self, external: &Vector) -> Vector {
        let mut internal = external.clone();
        let l00 = external.get(2).sqrt();
        let l10 = external.get(3) / l00;
        let l11 = (external.get(4) - l10 * l10).max(f64::EPSILON).sqrt();
        internal.set(2, l00);
        internal.set(3, l10);
        internal.set(4, l11);
        internal
    }

    fn to_external_jacobian(&self, z: &Vector) -> Matrix {
        let mut jacobian = Matrix::identity(5);
        let (l00, l10, l11) = (z.get(2), z.get(3), z.get(4));
        jacobian.set(2, 2, 2.0 * l00);
        jacobian.set(3, 2, l10);
        jacobian.set(3, 3, l00);
        jacobian.set(4, 3, 2.0 * l10);
        jacobian.set(4, 4, 2.0 * l11);
        jacobian
    }

    fn to_external_component_hessian(&self, component: usize, _: &Vector) -> Matrix {
        let mut hessian = Matrix::zeros(5, 5);
        match component {
            2 => hessian.set(2, 2, 2.0),
            3 => {
                hessian.set(2, 3, 1.0);
                hessian.set(3, 2, 1.0);
            }
            4 => {
                hessian.set(3, 3, 2.0);
                hessian.set(4, 4, 2.0);
            }
            _ => {}
        }
        hessian
    }
}

fn normal(rng: &mut Rng) -> f64 {
    let radius = (-2.0 * rng.f64().max(f64::MIN_POSITIVE).ln()).sqrt();
    radius * (std::f64::consts::TAU * rng.f64()).cos()
}

fn output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/multivariate_normal_fit/data.json")
}

fn main() -> Result<(), Box<dyn Error>> {
    let truth: [f64; 5] = [1.2, 2.3, 0.6, 0.25, 0.7];
    let l00 = truth[2].sqrt();
    let l10 = truth[3] / l00;
    let l11 = (truth[4] - l10 * l10).sqrt();
    let mut rng = Rng::with_seed(23);
    let data: Vec<Vector> = (0..600)
        .map(|_| {
            let (z0, z1) = (normal(&mut rng), normal(&mut rng));
            [truth[0] + l00 * z0, truth[1] + l10 * z0 + l11 * z1].into()
        })
        .collect();

    let nm_config: NelderMeadConfig = NelderMeadConfig::default()
        .with_parameter_names(NAMES)
        .with_transform(PositiveDefinite);
    let nm = NelderMead::default().process_with_default_callbacks(
        &GaussianFit,
        &data,
        [0.5, 1.0, 0.8, 0.1, 0.8],
        nm_config,
    )?;
    println!("Nelder–Mead\n{nm}\n");

    let cg_config: ConjugateGradientConfig = ConjugateGradientConfig::default()
        .with_parameter_names(NAMES)
        .with_transform(PositiveDefinite);
    let fit = ConjugateGradient::<f64>::default().process(
        &GaussianFit,
        &data,
        nm.x.clone(),
        cg_config,
        ConjugateGradient::<f64>::default_callbacks().with_terminator(MaxSteps(40)),
    )?;
    println!("Conjugate gradient\n{fit}\n");

    let center = PositiveDefinite.to_internal(&fit.x);
    let walkers: Vec<Vector> = (0..32)
        .map(|_| {
            let proposal: Vector = (0..5)
                .map(|index| center.get(index) + 0.08 * normal(&mut rng))
                .collect::<Vec<_>>()
                .into();
            PositiveDefinite.to_external(&proposal)
        })
        .collect();
    let ess_config: ESSConfig = ESSConfig::default()
        .with_parameter_names(NAMES)
        .with_moves([ESSMove::gaussian(0.3), ESSMove::differential(0.7)])?
        .with_transform(PositiveDefinite);
    let posterior = ESS::new(Some(23)).process(
        &GaussianFit,
        &data,
        ESSInit::new(walkers)?,
        ess_config,
        Callbacks::empty().with_terminator(MaxSteps(300)),
    )?;
    let chains: Vec<Vec<Vec<f64>>> = posterior
        .chain
        .iter()
        .map(|chain| chain.iter().map(Vector::to_vec).collect())
        .collect();
    let observations: Vec<Vec<f64>> = data.iter().map(Vector::to_vec).collect();

    serde_json::to_writer_pretty(
        File::create(output_path())?,
        &json!({
            "title": "Bivariate Gaussian fit and posterior",
            "parameter_names": NAMES,
            "truth": truth,
            "observations": observations,
            "nelder_mead": nm.x.to_vec(),
            "fit": fit.x.to_vec(),
            "standard_errors": fit.std.to_vec(),
            "chains": chains,
            "burn": 75,
        }),
    )?;
    println!("{posterior}");
    println!("wrote {}", output_path().display());
    Ok(())
}
