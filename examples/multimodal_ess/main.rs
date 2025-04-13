use std::convert::Infallible;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use fastrand::Rng;
use ganesh::core::CtrlCAbortSignal;
use ganesh::legacy::observer::AutocorrelationObserver;
use ganesh::legacy::samplers::ess::{ESSMove, ESS};
use ganesh::legacy::samplers::Sampler;
use ganesh::traits::{AbortSignal, CostFunction};
use ganesh::utils::SampleFloat;
use ganesh::Float;
use nalgebra::DVector;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Define the function to sample (a multimodal distribution)
    struct Problem;
    // Implement Function (Himmelblau's test function)
    impl CostFunction<(), Infallible> for Problem {
        fn evaluate(&self, x: &[Float], _user_data: &mut ()) -> Result<Float, Infallible> {
            Ok(-((x[0].powi(2) + x[1] - 11.0).powi(2) + (x[0] + x[1].powi(2) - 7.0).powi(2)))
        }
    }
    let problem = Problem;

    // Create and seed a random number generator
    let mut rng = Rng::new();
    rng.seed(0);

    // Define the initial state of the (100) walkers (normally distributed in 2 dimensions)
    let x0 = (0..100)
        .map(|_| DVector::from_fn(2, |_, _| rng.normal(0.0, 7.0)))
        .collect();

    // Create a new Ensemble Slice Sampler algorithm which uses Differential steps 20% of the time,
    // Global steps 70% of the time, and Gaussian steps the other 10%.
    // The global step is set with a scale factor of 0.5 on the covariance matrix of each Gaussian
    // mixture cluster, where the default is usually 0.001. This promotes jumps between distant
    // clusters.
    let a = ESS::new(
        [
            ESSMove::gaussian(0.1),
            ESSMove::global(0.7, None, Some(0.5), Some(4)),
            ESSMove::differential(0.2),
        ],
        rng,
    );

    // Terminate if the number of steps exceeds 6 integrated autocorrelation times (IAT) and the
    // difference in IAT is less than 1%
    let aco = AutocorrelationObserver::default()
        .with_verbose(true)
        .with_n_taus_threshold(6)
        .build();

    // Create a new Sampler
    let mut s = Sampler::new(Box::new(a), x0).with_observer(aco.clone());

    // Run a maximum of 4000 steps of the MCMC algorithm
    s.sample(&problem, &mut (), 4000, CtrlCAbortSignal::new().boxed())?;

    // Get the resulting samples (no burn-in)
    let chains = s.get_chains(None, None);

    // Get the integrated autocorrelation times
    let taus = aco.read().taus.clone();

    // Export the results to a Python .pkl file to visualize via matplotlib
    let mut writer = BufWriter::new(File::create(Path::new("data.pkl"))?);
    serde_pickle::to_writer(&mut writer, &(chains, taus), Default::default())?;
    Ok(())
}
