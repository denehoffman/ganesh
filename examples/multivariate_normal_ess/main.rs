use std::collections::HashMap;
use std::convert::Infallible;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use fastrand::Rng;
use ganesh::algorithms::mcmc::ess::{ESStep, ESS};
use ganesh::algorithms::mcmc::Sampler;
use ganesh::{Float, Function, SampleFloat};
use nalgebra::{DMatrix, DVector};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Define the function to sample (a multinormal distribution)
    struct Problem;
    // Implement Function (user_data is the inverse of the covariance matrix)
    // NOTE: this is just proportional to the log of the multinormal!
    impl Function<DMatrix<Float>, Infallible> for Problem {
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
    let a = ESS::new(&[(ESStep::Gaussian, 0.1), (ESStep::Differential, 0.9)], rng);

    // Create a new Sampler and set it up to run 1000 steps per walker
    let mut s = Sampler::new(&a, x0, 5).with_max_steps(1000);

    // Run the MCMC
    s.sample(&problem, &mut cov_inv)?;

    // Get the resulting samples, discarding the first 200 (burn-in)
    let flat_chain = s.get_flat_chain(Some(200), None);

    // Export the results to a Python .pkl file to visualize via matplotlib
    let mut map = HashMap::new();
    map.insert("flat chain", flat_chain);
    let mut writer = BufWriter::new(File::create(Path::new("data.pkl"))?);
    serde_pickle::to_writer(&mut writer, &map, Default::default())?;
    Ok(())
}
