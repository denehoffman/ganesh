use std::convert::Infallible;

use nalgebra::{DMatrix, DVector};

use crate::Float;

/// A trait which describes a function $`f(\mathbb{R}^n) \to \mathbb{R}`$
///
/// Such a function may also take a `user_data: &mut U` field which can be used to pass external
/// arguments to the function during minimization, or can be modified by the function itself.
///
/// The `CostFunction` trait takes a generic `U` representing the type of user data/arguments
/// and a generic `E` representing any possible errors that might be returned during function
/// execution.
///
pub trait CostFunction<U = (), E = Infallible> {
    /// The input space consumed by the cost function.
    type Input;
    /// The evaluation of the function at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// [`std::convert::Infallible`] if the function evaluation never fails.
    fn evaluate(&self, x: &Self::Input, user_data: &mut U) -> Result<Float, E>;
}

/// A trait which defines the gradient of a function $`f(\mathbb{R}^n) \to \mathbb{R}`$
///
/// Such a function may also take a `user_data: &mut U` field which can be used to pass external
/// arguments to the function during minimization, or can be modified by the function itself.
///
/// The `Gradient` trait takes a  generic `U` representing the type of user data/arguments
/// and a generic `E` representing any possible errors that might be returned during function
/// execution.
///
pub trait Gradient<U = (), E = Infallible>: CostFunction<U, E, Input = DVector<Float>> {
    /// The evaluation of the gradient at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    fn gradient(&self, x: &Self::Input, user_data: &mut U) -> Result<DVector<Float>, E> {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        // This is technically the best step size for the gradient, cbrt(eps) * x_i (or just
        // cbrt(eps) if x_i = 0)
        let h: DVector<Float> = x
            .iter()
            .map(|&xi| Float::cbrt(Float::EPSILON) * (xi.abs() + 1.0))
            .collect::<Vec<_>>()
            .into();
        for i in 0..n {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h[i];
            x_minus[i] -= h[i];
            let f_plus = self.evaluate(&x_plus, user_data)?;
            let f_minus = self.evaluate(&x_minus, user_data)?;
            grad[i] = (f_plus - f_minus) / (2.0 * h[i]);
        }
        Ok(grad)
    }

    /// The evaluation of the hessian at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    fn hessian(&self, x: &Self::Input, user_data: &mut U) -> Result<DMatrix<Float>, E> {
        let h: DVector<Float> = x
            .iter()
            .map(|&xi| Float::cbrt(Float::EPSILON) * (xi.abs() + 1.0))
            .collect::<Vec<_>>()
            .into();
        let mut res = DMatrix::zeros(x.len(), x.len());
        let mut g_plus = DMatrix::zeros(x.len(), x.len());
        let mut g_minus = DMatrix::zeros(x.len(), x.len());
        // g+ and g- are such that
        // g+[(i, j)] = g[i](x + h_je_j) and
        // g-[(i, j)] = g[i](x - h_je_j)
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += h[i];
            x_minus[i] -= h[i];
            g_plus.set_column(i, &self.gradient(&x_plus, user_data)?);
            g_minus.set_column(i, &self.gradient(&x_minus, user_data)?);
            for j in 0..=i {
                if i == j {
                    res[(i, j)] = (g_plus[(i, j)] - g_minus[(i, j)]) / (2.0 * h[i]);
                } else {
                    res[(i, j)] = ((g_plus[(i, j)] - g_minus[(i, j)]) / (4.0 * h[j]))
                        + ((g_plus[(j, i)] - g_minus[(j, i)]) / (4.0 * h[i]));
                    res[(j, i)] = res[(i, j)];
                }
            }
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use approx::assert_relative_eq;

    use crate::{
        traits::{CostFunction, Gradient},
        DVector, Float,
    };

    struct TestFunction;
    impl CostFunction for TestFunction {
        type Input = DVector<Float>;
        fn evaluate(&self, x: &DVector<Float>, _: &mut ()) -> Result<Float, Infallible> {
            Ok(x[0].powi(2) + x[1].powi(2) + 1.0)
        }
    }
    impl Gradient for TestFunction {}

    #[test]
    fn test_cost_function() {
        let x: DVector<Float> = DVector::from_vec(vec![1.0, 2.0]);
        let y = TestFunction.evaluate(&x, &mut ()).unwrap();
        assert_eq!(y, 6.0);
    }

    #[test]
    fn test_cost_function_gradient() {
        let x: DVector<Float> = DVector::from_vec(vec![1.0, 2.0]);
        let dy = TestFunction.gradient(&x, &mut ()).unwrap();
        assert_relative_eq!(dy[0], 2.0, epsilon = Float::EPSILON.sqrt());
        assert_relative_eq!(dy[1], 4.0, epsilon = Float::EPSILON.sqrt());
    }

    #[test]
    fn test_cost_function_hessian() {
        let x: DVector<Float> = DVector::from_vec(vec![1.0, 2.0]);
        let hessian = TestFunction.hessian(&x, &mut ()).unwrap();
        assert_relative_eq!(hessian[(0, 0)], 2.0, epsilon = Float::EPSILON.cbrt());
        assert_relative_eq!(hessian[(1, 1)], 2.0, epsilon = Float::EPSILON.cbrt());
        assert_relative_eq!(hessian[(0, 1)], 0.0, epsilon = Float::EPSILON.cbrt());
        assert_relative_eq!(hessian[(1, 0)], 0.0, epsilon = Float::EPSILON.cbrt());
    }

    #[test]
    fn test_cost_function_covariance_and_std() -> Result<(), Infallible> {
        use crate::utils::hessian_to_covariance;
        let x: DVector<Float> = DVector::from_vec(vec![1.0, 2.0]);
        let hessian = TestFunction.hessian(&x, &mut ())?;
        #[allow(clippy::unwrap_used)]
        let cov = hessian_to_covariance(&hessian).unwrap();
        assert_relative_eq!(cov[(0, 0)], 0.5, epsilon = Float::EPSILON.cbrt());
        assert_relative_eq!(cov[(1, 1)], 0.5, epsilon = Float::EPSILON.cbrt());
        Ok(())
    }
}
