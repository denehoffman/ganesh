use crate::{
    traits::{
        CostFunction, GenericCostFunction, GenericGradient, Gradient, LegacyCostFunction,
        LegacyGradient, LegacyLogDensity, LogDensity,
    },
    DVector, Float, LinearAlgebra, RealScalar, Vector,
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
impl LegacyCostFunction for Rosenbrock {
    fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
        #[allow(clippy::suboptimal_flops)]
        Ok((0..(self.n - 1))
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum())
    }
}
impl LegacyGradient for Rosenbrock {}
impl<T, B> CostFunction<T, B> for Rosenbrock
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn evaluate(&self, x: &Vector<T, B>, _args: &()) -> Result<T, Infallible> {
        let hundred = T::literal(100.0);
        Ok((0..(self.n - 1))
            .map(|i| {
                hundred * (x.get(i + 1) - x.get(i).powi(2)).powi(2) + (T::one() - x.get(i)).powi(2)
            })
            .fold(T::zero(), |sum, value| sum + value))
    }
}
impl<T, B> Gradient<T, B> for Rosenbrock
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
}
impl GenericCostFunction for Rosenbrock {
    type Input = DVector<Float>;

    fn evaluate_generic(&self, x: &Self::Input, args: &()) -> Result<Float, Infallible> {
        <Self as LegacyCostFunction>::evaluate(self, x, args)
    }
}
impl GenericGradient for Rosenbrock {
    fn gradient_generic(&self, x: &Self::Input, args: &()) -> Result<DVector<Float>, Infallible> {
        <Self as LegacyGradient>::gradient(self, x, args)
    }

    fn hessian_generic(
        &self,
        x: &Self::Input,
        args: &(),
    ) -> Result<nalgebra::DMatrix<Float>, Infallible> {
        <Self as LegacyGradient>::hessian(self, x, args)
    }
}

impl LegacyLogDensity for Rosenbrock {
    fn log_density(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
        #[allow(clippy::suboptimal_flops)]
        Ok(-Float::ln(
            (0..(self.n - 1))
                .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
                .sum::<Float>(),
        ))
    }
}

impl<T, B> LogDensity<T, B> for Rosenbrock
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn log_density(&self, x: &Vector<T, B>, args: &()) -> Result<T, Infallible> {
        Ok(-<Self as CostFunction<T, B>>::evaluate(self, x, args)?.ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rosenbrock_evaluate_at_minimum() {
        let f = Rosenbrock { n: 2 };
        let x = DVector::from_vec(vec![1.0, 1.0]);
        let val = LegacyCostFunction::evaluate(&f, &x, &()).unwrap();
        assert_eq!(val, 0.0); // global minimum
    }

    #[test]
    fn test_rosenbrock_evaluate_known_point() {
        let f = Rosenbrock { n: 2 };
        let x = DVector::from_vec(vec![0.0, 0.0]);
        let val = LegacyCostFunction::evaluate(&f, &x, &()).unwrap();
        // f(0,0) = 100*(0-0)^2 + (1-0)^2 = 1
        assert_eq!(val, 1.0);
    }

    #[test]
    fn test_rosenbrock_evaluate_three_dimensions() {
        let f = Rosenbrock { n: 3 };
        let x = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let val = LegacyCostFunction::evaluate(&f, &x, &()).unwrap();
        // two terms, each zero at (1,1), so total = 0
        assert_eq!(val, 0.0);
    }

    #[test]
    fn generic_problem_supports_f32_and_f64_together() {
        let f = Rosenbrock { n: 2 };
        let x32 = Vector::<f32>::from_vec(vec![1.0, 1.0]);
        let x64 = Vector::<f64>::from_vec(vec![1.0, 1.0]);
        assert_eq!(
            <Rosenbrock as CostFunction<f32>>::evaluate(&f, &x32, &()).unwrap(),
            0.0
        );
        assert_eq!(
            <Rosenbrock as CostFunction<f64>>::evaluate(&f, &x64, &()).unwrap(),
            0.0
        );
    }

    #[test]
    fn test_rosenbrock_log_density_at_minimum() {
        let f = Rosenbrock { n: 2 };
        let x = DVector::from_vec(vec![1.0, 1.0]);
        // at the minimum, f = 0, so log_density = ln(0) → ∞
        // note there is an additional minus sign to flip the sign
        // of the Rosenbrock function to provide a maximum rather than
        // a minimum
        let val = LegacyLogDensity::log_density(&f, &x, &()).unwrap();
        assert!(val.is_infinite() && val.is_sign_positive());
    }

    #[test]
    fn test_rosenbrock_log_density_known_point() {
        let f = Rosenbrock { n: 2 };
        let x = DVector::from_vec(vec![0.0, 0.0]);
        let val = LegacyLogDensity::log_density(&f, &x, &()).unwrap();
        // f(0,0) = 1, so log_density = -ln(1) = 0
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_rosenbrock_log_density_increasing_cost_decreases_density() {
        let f = Rosenbrock { n: 2 };
        let x1 = DVector::from_vec(vec![0.0, 0.0]); // cost = 1, log_density = 0
        let x2 = DVector::from_vec(vec![2.0, 2.0]); // higher cost, lower log_density
        let ld1 = LegacyLogDensity::log_density(&f, &x1, &()).unwrap();
        let ld2 = LegacyLogDensity::log_density(&f, &x2, &()).unwrap();
        assert!(ld2 < ld1);
    }
}
