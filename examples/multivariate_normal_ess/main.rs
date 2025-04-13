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
use nalgebra::{DMatrix, DVector};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Define the function to sample (a multinormal distribution)
    struct Problem;
    // Implement Function (user_data is the inverse of the covariance matrix)
    // NOTE: this is just proportional to the log of the multinormal!
    impl CostFunction<DMatrix<Float>, Infallible> for Problem {
        fn evaluate(
            &self,
            x: &[Float],
            user_data: &mut DMatrix<Float>,
        ) -> Result<Float, Infallible> {
            Ok(-0.5
                * DVector::from_row_slice(x).dot(&(&*user_data * DVector::from_column_slice(x))))
        }
    }
    let problem = Problem;

    // Create and seed a random number generator
    let mut rng = Rng::new();
    rng.seed(0);

    // Define the initial state of the (100) walkers (normally distributed in 5 dimensions)
    let x0 = (0..100)
        .map(|_| DVector::from_fn(5, |_, _| rng.normal(0.0, 4.0)))
        .collect();

    // Generate a random (inverse) covariance matrix (scaling on off-diagonals makes for
    // nicer-looking results)
    let mut cov_inv = DMatrix::from_fn(5, 5, |i, j| if i == j { 1.0 } else { 0.1 } / rng.float());
    println!("Σ⁻¹ = \n{}", cov_inv);

    // Create a new Ensemble Slice Sampler algorithm which uses Differential steps 90% of the time
    // and Gaussian steps the other 10%
    let a = ESS::new([ESSMove::gaussian(0.1), ESSMove::differential(0.9)], rng);

    let aco = AutocorrelationObserver::default()
        .with_verbose(true)
        .build();

    // Create a new Sampler
    let mut s = Sampler::new(Box::new(a), x0).with_observer(aco.clone());

    // Run a maximum of 1000 steps of the MCMC algorithm
    s.sample(
        &problem,
        &mut cov_inv,
        1000,
        CtrlCAbortSignal::new().boxed(),
    )?;

    // Get the resulting samples (no burn-in)
    let chains = s.get_chains(None, None);

    // Get the integrated autocorrelation times
    let taus = aco.read().taus.clone();

    // Export the results to a Python .pkl file to visualize via matplotlib
    let mut writer = BufWriter::new(File::create(Path::new("data.pkl"))?);
    serde_pickle::to_writer(&mut writer, &(chains, taus), Default::default())?;
    Ok(())
}
