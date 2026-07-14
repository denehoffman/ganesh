//! The minimal Nelder–Mead example shown in the README.

use ganesh::{
    algorithms::gradient_free::NelderMead, test_functions::Rosenbrock, traits::Algorithm,
};
use std::convert::Infallible;

fn main() -> Result<(), Infallible> {
    let problem = Rosenbrock { n: 2 };
    let mut nm: NelderMead = Default::default();
    let result = nm.process_default(&problem, &(), [2.0, 2.0])?;
    println!("{result}");
    Ok(())
}
