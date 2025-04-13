use std::convert::Infallible;

use crate::traits::CostFunction;
use crate::Float;

/// The Rosenbrock function, a non-convex function with a single minimum.
///
/// ```math
/// f(\vec{x}) = \sum_{i=1}^{n-1} \left[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right]
/// ```
/// where $`n \geq 2`$. This function has a minimum at $`f(\vec{1}) = 0`$.
pub struct Rosenbrock {
    /// The number of dimensions of the function (must be >= 2).
    pub n: usize,
}
impl CostFunction<(), Infallible> for Rosenbrock {
    fn evaluate(&self, x: &[Float], _user_data: &mut ()) -> Result<Float, Infallible> {
        #[allow(clippy::suboptimal_flops)]
        Ok((0..(self.n - 1))
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum())
    }
}

/// The negative Rosenbrock function, a non-convex function with a single maximum.
///
/// This function can be used to test MCMC methods which maximize probability rather than minimize
/// function value.
///
/// ```math
/// f(\vec{x}) = -\sum_{i=1}^{n-1} \left[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right]
/// ```
/// where $`n \geq 2`$. This function has a maximum at $`f(\vec{1}) = 0`$.
pub struct NegativeRosenbrock {
    /// The number of dimensions of the function (must be >= 2).
    pub n: usize,
}
impl CostFunction<(), Infallible> for NegativeRosenbrock {
    fn evaluate(&self, x: &[Float], _user_data: &mut ()) -> Result<Float, Infallible> {
        #[allow(clippy::suboptimal_flops)]
        Ok(-(0..(self.n - 1))
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum::<Float>())
    }
}
