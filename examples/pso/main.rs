use fastrand::Rng;
use ganesh::{
    algorithms::particles::{pso::PSOConfig, SwarmPositionInitializer, TrackingSwarmObserver, PSO},
    core::{Callbacks, MaxSteps},
    traits::{Algorithm, CostFunction},
    DVector, Float, PI,
};
use std::{convert::Infallible, error::Error, fs::File, io::BufWriter, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    // Define the function to sample (a multimodal distribution)
    struct Problem;
    // Implement Rastrigin function
    impl CostFunction for Problem {
        type Input = DVector<Float>;
        fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(10.0
                + (x[0].powi(2) - 10.0 * Float::cos(2.0 * PI * x[0]))
                + (x[1].powi(2) - 10.0 * Float::cos(2.0 * PI * x[1])))
        }
    }
    let mut problem = Problem;

    // Create and seed a random number generator
    let mut rng = Rng::new();
    rng.seed(0);

    // Create a tracker to record swarm history
    let tracker = TrackingSwarmObserver::new();

    // Create a particle swarm optimizer algorithm and set some hyperparameters
    // Run the particle swarm optimizer
    let mut pso = PSO::new(2, rng);
    let result = pso.process(
        &mut problem,
        &(),
        PSOConfig::default()
            .with_c1(0.1)
            .with_c2(0.1)
            .with_omega(0.8)
            .setup_swarm(|swarm| {
                swarm
                    .with_position_initializer(SwarmPositionInitializer::RandomInLimits(vec![
                        (-20.0, 20.0),
                        (-20.0, 20.0),
                    ]))
                    .with_n_particles(50)
            }),
        Callbacks::empty()
            .with_observer(tracker.clone())
            .with_terminator(MaxSteps(200)),
    )?;

    println!("{}", result);

    // Export the results to a Python .pkl file to visualize via matplotlib
    let mut writer = BufWriter::new(File::create(Path::new("data.pkl"))?);
    serde_pickle::to_writer(&mut writer, &tracker, Default::default())?;
    Ok(())
}
