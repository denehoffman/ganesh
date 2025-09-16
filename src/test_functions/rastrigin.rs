use crate::{
    traits::{CostFunction, FiniteDifferenceGradient, Gradient},
    DVector, Float, PI,
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
impl CostFunction for Rastrigin {
    type Input = DVector<Float>;
    fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
        #[allow(clippy::suboptimal_flops)]
        Ok(10.0 * self.n as Float
            + (0..self.n)
                .map(|i| x[i].powi(2) - 10.0 * Float::cos(2.0 * PI * x[i]))
                .sum::<Float>())
    }
}
impl Gradient for Rastrigin {
    fn gradient(&self, x: &Self::Input, args: &()) -> Result<DVector<Float>, Infallible> {
        self.cfd_gradient(x, args)
    }

    fn hessian(&self, x: &Self::Input, args: &()) -> Result<nalgebra::DMatrix<Float>, Infallible> {
        self.cfd_hessian(x, args)
    }
}
