use fastrand::Rng;
use ganesh::{
    algorithms::mcmc::{ess::ESSConfig, AutocorrelationTerminator, ESSMove, ESS},
    core::{utils::SampleFloat, Callbacks, MaxSteps},
    traits::{Algorithm, LogDensity},
    DMatrix, DVector, Float,
};
use std::{convert::Infallible, error::Error, fs::File, io::BufWriter, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    // Define the function to sample (a multinormal distribution)
    struct Problem;
    // Implement Function (args is the inverse of the covariance matrix)
    // NOTE: this is just proportional to the log of the multinormal!
    impl LogDensity<DMatrix<Float>> for Problem {
        type Input = DVector<Float>;
        fn log_density(
            &self,
            x: &DVector<Float>,
            args: &DMatrix<Float>,
        ) -> Result<Float, Infallible> {
            Ok(-0.5 * x.dot(&(args * x)))
        }
    }
    let mut problem = Problem;

    // Create and seed a random number generator
    let mut rng = Rng::new();
    rng.seed(0);

    // Define the initial state of the (100) walkers (normally distributed in 5 dimensions)
    let x0: Vec<DVector<Float>> = (0..100)
        .map(|_| DVector::from_fn(5, |_, _| rng.normal(0.0, 4.0)))
        .collect();

    // Generate a random (inverse) covariance matrix (scaling on off-diagonals makes for
    // nicer-looking results)
    let cov_inv = DMatrix::from_fn(5, 5, |i, j| if i == j { 1.0 } else { 0.1 } / rng.float());
    println!("Σ⁻¹ = \n{}", cov_inv);

    let aco = AutocorrelationTerminator::default()
        .with_verbose(true)
        .build();

    let mut sampler = ESS::default();

    // Create a new Ensemble Slice Sampler algorithm which uses Differential steps 90% of the time
    // and Gaussian steps the other 10%
    // Run a maximum of 1000 steps of the MCMC algorithm
    let result = sampler.process(
        &mut problem,
        &cov_inv,
        ESSConfig::new(x0.clone()).with_moves([
            ESSMove::gaussian(0.1),
            ESSMove::global(0.7, None, Some(0.5), Some(4)),
            ESSMove::differential(0.2),
        ]),
        Callbacks::empty()
            .with_terminator(aco.clone())
            .with_terminator(MaxSteps(1000)),
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
