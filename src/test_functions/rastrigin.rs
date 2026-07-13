use crate::{
    traits::{CostFunction, Gradient, LegacyCostFunction, LegacyGradient},
    DVector, Float, LinearAlgebra, RealScalar, Vector, PI,
};
use std::convert::Infallible;

/// The Rastrigin function, a non-convex function with a single minimum but many local minima.
///
/// ```math
/// f(\vec{x}) = 10n + \sum_{i=1}^n (x_i^2 - 10cos(2\pi x_i))
/// ```
pub struct Rastrigin {
    /// The number of dimensions of the function (must be >= 2).
    pub n: usize,
}
impl LegacyCostFunction for Rastrigin {
    fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
        #[allow(clippy::suboptimal_flops)]
        Ok(10.0 * self.n as Float
            + (0..self.n)
                .map(|i| x[i].powi(2) - 10.0 * Float::cos(2.0 * PI * x[i]))
                .sum::<Float>())
    }
}
impl LegacyGradient for Rastrigin {}

impl<T, B> CostFunction<T, B> for Rastrigin
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn evaluate(&self, x: &Vector<T, B>, _args: &()) -> Result<T, Infallible> {
        let ten = T::literal(10.0);
        let two_pi = T::literal(2.0 * std::f64::consts::PI);
        Ok(ten * T::literal(self.n as f64)
            + (0..self.n)
                .map(|i| x.get(i).powi(2) - ten * (two_pi * x.get(i)).cos())
                .fold(T::zero(), |sum, value| sum + value))
    }
}

impl<T, B> Gradient<T, B> for Rastrigin
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
}
