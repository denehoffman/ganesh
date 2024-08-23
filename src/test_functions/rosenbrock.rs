use std::convert::Infallible;

use crate::Function;

pub struct Rosenbrock {
    pub n: usize,
}
impl Function<f64, (), Infallible> for Rosenbrock {
    fn evaluate(&self, x: &[f64], _user_data: &mut ()) -> Result<f64, Infallible> {
        Ok((0..(self.n - 1))
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum())
    }
}
