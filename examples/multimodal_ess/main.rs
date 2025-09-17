use fastrand::Rng;
use ganesh::{
    algorithms::mcmc::{ess::ESSConfig, AutocorrelationTerminator, ESSMove, ESS},
    core::{utils::SampleFloat, Callbacks, MaxSteps},
    traits::{Algorithm, LogDensity},
    DVector, Float,
};
use std::{convert::Infallible, error::Error, fs::File, io::BufWriter, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    // Define the function to sample (a multimodal distribution)
    struct Problem;
    // Implement Function (Himmelblau's test function)
    impl LogDensity for Problem {
        fn log_density(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(-((x[0].powi(2) + x[1] - 11.0).powi(2) + (x[0] + x[1].powi(2) - 7.0).powi(2)))
        }
    }
    let problem = Problem;

    // Create and seed a random number generator
    let mut rng = Rng::new();
    rng.seed(0);

    // Define the initial state of the (100) walkers (normally distributed in 2 dimensions)
    let x0: Vec<DVector<Float>> = (0..100)
        .map(|_| DVector::from_fn(2, |_, _| rng.normal(0.0, 7.0)))
        .collect();

    // Terminate if the number of steps exceeds 6 integrated autocorrelation times (IAT) and the
    // difference in IAT is less than 1%
    let aco = AutocorrelationTerminator::default()
        .with_verbose(true)
        .with_n_taus_threshold(6)
        .build();

    let mut sampler = ESS::default();
    // Create a new Ensemble Slice Sampler algorithm which uses Differential steps 20% of the time,
    // Global steps 70% of the time, and Gaussian steps the other 10%.
    // The global step is set with a scale factor of 0.5 on the covariance matrix of each Gaussian
    // mixture cluster, where the default is usually 0.001. This promotes jumps between distant
    // clusters.
    // Run a maximum of 8000 steps of the MCMC algorithm
    let result = sampler.process(
        &problem,
        &(),
        ESSConfig::new(x0.clone()).with_moves([
            ESSMove::gaussian(0.1),
            ESSMove::global(0.7, None, Some(0.5), Some(4)),
            ESSMove::differential(0.2),
        ]),
        Callbacks::empty()
            .with_terminator(aco.clone())
            .with_terminator(MaxSteps(8000)),
    )?;

    // Get the resulting samples (no burn-in)
    let chains = result.chain;

    // Get the integrated autocorrelation times
    let taus = aco.lock().taus.clone();

    // Export the results to a Python .pkl file to visualize via matplotlib
    let mut writer = BufWriter::new(File::create(Path::new("data.pkl"))?);
    serde_pickle::to_writer(&mut writer, &(chains, taus), Default::default())?;
    Ok(())
}
