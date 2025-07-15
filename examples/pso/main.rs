use std::convert::Infallible;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use fastrand::Rng;
use ganesh::algorithms::particles::SwarmPositionInitializer;
use ganesh::algorithms::particles::TrackingSwarmObserver;
use ganesh::algorithms::particles::PSO;
use ganesh::core::Engine;
use ganesh::traits::Configurable;
use ganesh::traits::CostFunction;
use ganesh::Float;
use ganesh::PI;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Define the function to sample (a multimodal distribution)
    struct Problem;
    // Implement Rastrigin function
    impl CostFunction<(), Infallible> for Problem {
        fn evaluate(&self, x: &[Float], _user_data: &mut ()) -> Result<Float, Infallible> {
            Ok(10.0
                + (x[0].powi(2) - 10.0 * Float::cos(2.0 * PI * x[0]))
                + (x[1].powi(2) - 10.0 * Float::cos(2.0 * PI * x[1])))
        }
    }
    let problem = Problem;

    // Create and seed a random number generator
    let mut rng = Rng::new();
    rng.seed(0);

    // Create a tracker to record swarm history
    let tracker = TrackingSwarmObserver::build();

    // Create a particle swarm optimizer algorithm and set some hyperparameters
    let mut m = Engine::new(PSO::new(2, rng)).setup_engine(|e| {
        e.setup_algorithm(|a| {
            a.setup_config(|c| {
                c.with_c1(0.1)
                    .with_c2(0.1)
                    .with_omega(0.8)
                    .setup_swarm(|swarm| {
                        swarm
                            .with_position_initializer(SwarmPositionInitializer::RandomInLimits(
                                vec![(-20.0, 20.0), (-20.0, 20.0)],
                            ))
                            .with_n_particles(50)
                    })
            })
        })
        .with_observer(tracker.clone())
        .with_max_steps(200)
    });

    // Run the particle swarm optimizer
    m.process(&problem)?;

    println!("{}", m.result);

    // Export the results to a Python .pkl file to visualize via matplotlib
    let mut writer = BufWriter::new(File::create(Path::new("data.pkl"))?);
    serde_pickle::to_writer(&mut writer, &tracker, Default::default())?;
    Ok(())
}
