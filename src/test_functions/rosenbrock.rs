use crate::{
    traits::{CostFunction, Gradient, LogDensity},
    DVector, Float,
};
use std::convert::Infallible;

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
impl CostFunction for Rosenbrock {
    type Input = DVector<Float>;
    fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
        #[allow(clippy::suboptimal_flops)]
        Ok((0..(self.n - 1))
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum())
    }
}
impl Gradient for Rosenbrock {}

impl LogDensity for Rosenbrock {
    type Input = DVector<Float>;

    fn log_density(&self, x: &Self::Input, _args: &()) -> Result<Float, Infallible> {
        #[allow(clippy::suboptimal_flops)]
        Ok(-Float::ln(
            (0..(self.n - 1))
                .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
                .sum::<Float>(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rosenbrock_evaluate_at_minimum() {
        let f = Rosenbrock { n: 2 };
        let x = DVector::from_vec(vec![1.0, 1.0]);
        let val = f.evaluate(&x, &()).unwrap();
        assert_eq!(val, 0.0); // global minimum
    }

    #[test]
    fn test_rosenbrock_evaluate_known_point() {
        let f = Rosenbrock { n: 2 };
        let x = DVector::from_vec(vec![0.0, 0.0]);
        let val = f.evaluate(&x, &()).unwrap();
        // f(0,0) = 100*(0-0)^2 + (1-0)^2 = 1
        assert_eq!(val, 1.0);
    }

    #[test]
    fn test_rosenbrock_evaluate_three_dimensions() {
        let f = Rosenbrock { n: 3 };
        let x = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let val = f.evaluate(&x, &()).unwrap();
        // two terms, each zero at (1,1), so total = 0
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_rosenbrock_log_density_at_minimum() {
        let f = Rosenbrock { n: 2 };
        let x = DVector::from_vec(vec![1.0, 1.0]);
        // at the minimum, f = 0, so log_density = ln(0) → ∞
        // note there is an additional minus sign to flip the sign
        // of the Rosenbrock function to provide a maximum rather than
        // a minimum
        let val = f.log_density(&x, &()).unwrap();
        assert!(val.is_infinite() && val.is_sign_positive());
    }

    #[test]
    fn test_rosenbrock_log_density_known_point() {
        let f = Rosenbrock { n: 2 };
        let x = DVector::from_vec(vec![0.0, 0.0]);
        let val = f.log_density(&x, &()).unwrap();
        // f(0,0) = 1, so log_density = -ln(1) = 0
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_rosenbrock_log_density_increasing_cost_decreases_density() {
        let f = Rosenbrock { n: 2 };
        let x1 = DVector::from_vec(vec![0.0, 0.0]); // cost = 1, log_density = 0
        let x2 = DVector::from_vec(vec![2.0, 2.0]); // higher cost, lower log_density
        let ld1 = f.log_density(&x1, &()).unwrap();
        let ld2 = f.log_density(&x2, &()).unwrap();
        assert!(ld2 < ld1);
    }
}
