use crate::{
    traits::{CostFunction, Gradient, LogDensity},
    LinearAlgebra, RealScalar, Vector,
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
    #[allow(clippy::option_if_let_else)]
    fn gradient(&self, x: &Vector<T, B>, _args: &()) -> Result<Vector<T, B>, Infallible> {
        let two = T::literal(2.0);
        let two_hundred = T::literal(200.0);
        let four_hundred = T::literal(400.0);
        let gradient = if let Some(values) = x.as_slice() {
            (0..self.n)
                .map(|index| {
                    if index == 0 {
                        -four_hundred * values[0] * (values[1] - values[0].powi(2))
                            - two * (T::one() - values[0])
                    } else if index == self.n - 1 {
                        two_hundred * (values[index] - values[index - 1].powi(2))
                    } else {
                        two_hundred * (values[index] - values[index - 1].powi(2))
                            - four_hundred
                                * values[index]
                                * (values[index + 1] - values[index].powi(2))
                            - two * (T::one() - values[index])
                    }
                })
                .collect()
        } else {
            (0..self.n)
                .map(|index| {
                    if index == 0 {
                        -four_hundred * x.get(0) * (x.get(1) - x.get(0).powi(2))
                            - two * (T::one() - x.get(0))
                    } else if index == self.n - 1 {
                        two_hundred * (x.get(index) - x.get(index - 1).powi(2))
                    } else {
                        two_hundred * (x.get(index) - x.get(index - 1).powi(2))
                            - four_hundred
                                * x.get(index)
                                * (x.get(index + 1) - x.get(index).powi(2))
                            - two * (T::one() - x.get(index))
                    }
                })
                .collect()
        };
        Ok(Vector::<T, B>::from_vec(gradient))
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
        let x = Vector::<f64>::from_vec(vec![1.0, 1.0]);
        let val = CostFunction::evaluate(&f, &x, &()).unwrap();
        assert_eq!(val, 0.0); // global minimum
    }

    #[test]
    fn test_rosenbrock_evaluate_known_point() {
        let f = Rosenbrock { n: 2 };
        let x = Vector::<f64>::from_vec(vec![0.0, 0.0]);
        let val = CostFunction::evaluate(&f, &x, &()).unwrap();
        // f(0,0) = 100*(0-0)^2 + (1-0)^2 = 1
        assert_eq!(val, 1.0);
    }

    #[test]
    fn test_rosenbrock_evaluate_three_dimensions() {
        let f = Rosenbrock { n: 3 };
        let x = Vector::<f64>::from_vec(vec![1.0, 1.0, 1.0]);
        let val = CostFunction::evaluate(&f, &x, &()).unwrap();
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
        let x = Vector::<f64>::from_vec(vec![1.0, 1.0]);
        // at the minimum, f = 0, so log_density = ln(0) → ∞
        // note there is an additional minus sign to flip the sign
        // of the Rosenbrock function to provide a maximum rather than
        // a minimum
        let val = LogDensity::log_density(&f, &x, &()).unwrap();
        assert!(val.is_infinite() && val.is_sign_positive());
    }

    #[test]
    fn test_rosenbrock_log_density_known_point() {
        let f = Rosenbrock { n: 2 };
        let x = Vector::<f64>::from_vec(vec![0.0, 0.0]);
        let val = LogDensity::log_density(&f, &x, &()).unwrap();
        // f(0,0) = 1, so log_density = -ln(1) = 0
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_rosenbrock_log_density_increasing_cost_decreases_density() {
        let f = Rosenbrock { n: 2 };
        let x1 = Vector::<f64>::from_vec(vec![0.0, 0.0]); // cost = 1, log_density = 0
        let x2 = Vector::<f64>::from_vec(vec![2.0, 2.0]); // higher cost, lower log_density
        let ld1 = LogDensity::log_density(&f, &x1, &()).unwrap();
        let ld2 = LogDensity::log_density(&f, &x2, &()).unwrap();
        assert!(ld2 < ld1);
    }
}
