use std::convert::Infallible;

use ganesh::algorithms::NelderMead;
use ganesh::observers::DebugObserver;
use ganesh::prelude::*;
use ganesh::test_functions::rosenbrock::Rosenbrock;

fn main() -> Result<(), Infallible> {
    const n: usize = 3;
    let problem = Rosenbrock { n };
    let nm = NelderMead::default();
    let obs = DebugObserver;
    let mut opt = Minimizer::new(nm, n).with_max_steps(10_000_000);
    let x0 = &[5.0; n];
    let status = opt.minimize(&problem, x0, &mut ())?;
    dbg!(&status);
    Ok(())
}
