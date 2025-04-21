use std::convert::Infallible;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use fastrand::Rng;
use ganesh::algorithms::mcmc::ESS;
use ganesh::algorithms::mcmc::{AutocorrelationObserver, ESSMove};
use ganesh::core::Engine;
use ganesh::traits::CostFunction;
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
    let x0: Vec<DVector<Float>> = (0..100)
        .map(|_| DVector::from_fn(5, |_, _| rng.normal(0.0, 4.0)))
        .collect();

    // Generate a random (inverse) covariance matrix (scaling on off-diagonals makes for
    // nicer-looking results)
    let cov_inv = DMatrix::from_fn(5, 5, |i, j| if i == j { 1.0 } else { 0.1 } / rng.float());
    println!("Σ⁻¹ = \n{}", cov_inv);

    // Create a new Ensemble Slice Sampler algorithm which uses Differential steps 90% of the time
    // and Gaussian steps the other 10%
    let a = ESS::new([ESSMove::gaussian(0.1), ESSMove::differential(0.9)], rng);

    let aco = AutocorrelationObserver::default()
        .with_verbose(true)
        .build();

    // Create a new Sampler
    let mut m = Engine::new(a).setup(|m| {
        m.with_observer(aco.clone())
            .on_status(|s| s.with_walkers(x0.clone()))
            .with_user_data(cov_inv.clone())
    });

    // Run a maximum of 1000 steps of the MCMC algorithm
    m.process(&problem)?;

    // Get the resulting samples (no burn-in)
    let chains = m.result.chain;

    // Get the integrated autocorrelation times
    let taus = aco.read().taus.clone();

    // Export the results to a Python .pkl file to visualize via matplotlib
    let mut writer = BufWriter::new(File::create(Path::new("data.pkl"))?);
    serde_pickle::to_writer(&mut writer, &(chains, taus), Default::default())?;
    Ok(())
}
