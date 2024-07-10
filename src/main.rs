use std::convert::Infallible;

use ganesh::algorithms::*;
use ganesh::prelude::*;
use ganesh::test_functions::{Bukin6, GoldsteinPrice, Powell, PowellFletcher, Rosenbrock};

fn main() -> Result<(), Infallible> {
    let func = PowellFletcher;
    let opts = nelder_mead::NelderMeadOptionsBuilder::default()
        .max_iters(300)
        .build()
        .unwrap();
    let mut m = nelder_mead::NelderMead::new(func, &[-1.0, 0.0, 0.0], opts);
    m.minimize(|_| {})?;
    println!("{:#?}", m.best());
    Ok(())
}
