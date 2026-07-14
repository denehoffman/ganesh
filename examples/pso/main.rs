//! Animate particle swarm optimization on the two-dimensional Rastrigin function.

use ganesh::{
    algorithms::{
        gradient_free::GradientFreeStatus,
        particles::{PSOConfig, PSO},
    },
    core::{Callbacks, MaxSteps},
    traits::{Algorithm, CostFunction, Observer},
    Vector,
};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::json;
use std::{convert::Infallible, error::Error, fs::File, path::PathBuf, sync::Arc};

#[derive(Serialize)]
struct SwarmFrame {
    step: usize,
    positions: Vec<Vec<f64>>,
    velocities: Vec<Vec<f64>>,
    personal_bests: Vec<Vec<f64>>,
    global_best: Vec<f64>,
    global_best_value: f64,
}

#[derive(Clone, Default)]
struct SwarmHistory(Arc<Mutex<Vec<SwarmFrame>>>);

struct ShowcaseRastrigin;

impl CostFunction for ShowcaseRastrigin {
    fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, Infallible> {
        Ok(
            10.0 + x.get(0).powi(2) - 10.0 * (std::f64::consts::TAU * x.get(0)).cos()
                + x.get(1).powi(2)
                - 10.0 * (std::f64::consts::TAU * x.get(1)).cos(),
        )
    }
}

fn to_external(config: &PSOConfig, value: &Vector) -> Vector {
    config.to_external(value)
}

impl Observer<PSO, ShowcaseRastrigin, GradientFreeStatus, (), Infallible, PSOConfig>
    for SwarmHistory
{
    fn observe(
        &mut self,
        step: usize,
        algorithm: &PSO,
        _: &ShowcaseRastrigin,
        status: &GradientFreeStatus,
        _: &(),
        config: &PSOConfig,
    ) {
        let mut positions = Vec::with_capacity(algorithm.particles().len());
        let mut velocities = Vec::with_capacity(algorithm.particles().len());
        let mut personal_bests = Vec::with_capacity(algorithm.particles().len());

        for particle in algorithm.particles() {
            let position = to_external(config, &particle.x).to_vec();
            let velocity_tip = to_external(config, &particle.x.add(&particle.velocity)).to_vec();
            velocities.push(
                velocity_tip
                    .iter()
                    .zip(&position)
                    .map(|(tip, current)| tip - current)
                    .collect(),
            );
            positions.push(position);
            personal_bests.push(to_external(config, &particle.best_x).to_vec());
        }

        self.0.lock().push(SwarmFrame {
            step,
            positions,
            velocities,
            personal_bests,
            global_best: status.x.to_vec(),
            global_best_value: status.fx,
        });
    }
}

fn output_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/pso/data.json")
}

fn main() -> Result<(), Box<dyn Error>> {
    let history = SwarmHistory::default();
    let config: PSOConfig = PSOConfig::default()
        .with_particles(50)?
        .with_omega(0.8)?
        .with_c1(0.1)?
        .with_c2(0.1)?
        .with_uniform_initialization([(-20.0, 20.0), (-20.0, 20.0)])?;
    let summary = PSO::new(Some(0)).process(
        &ShowcaseRastrigin,
        &(),
        [0.0, 0.0],
        config,
        Callbacks::empty()
            .with_observer(history.clone())
            .with_terminator(MaxSteps(200)),
    )?;

    serde_json::to_writer_pretty(
        File::create(output_path())?,
        &json!({
            "title": "Particle swarm optimization on Rastrigin",
            "plot_bounds": [-10.0, 10.0],
            "initialization_bounds": [-20.0, 20.0],
            "history": &*history.0.lock(),
            "minimum": summary.x.to_vec(),
            "minimum_value": summary.fx,
        }),
    )?;
    println!("{summary}");
    println!("wrote {}", output_path().display());
    Ok(())
}
