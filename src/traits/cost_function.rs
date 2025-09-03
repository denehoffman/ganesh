use crate::{DMatrix, DVector, Float};
use std::convert::Infallible;

/// A trait which describes a function $`f(\mathbb{R}^n) \to \mathbb{R}`$
///
/// Such a function may also take a `user_data: &mut U` field which can be used to pass external
/// arguments to the function during minimization or can be modified by the function itself.
///
/// The [`CostFunction`] trait takes a generic `U` representing the type of user data/arguments
/// and a generic `E` representing any possible errors that might be returned during function
/// execution.
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
/// arguments to the function during minimization or can be modified by the function itself.
///
/// The [`Gradient`] trait takes a  generic `U` representing the type of user data/arguments
/// and a generic `E` representing any possible errors that might be returned during function
/// execution.
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
        let mut xw = x.clone();
        for i in 0..n {
            let xi = x[i];
            let hi = Float::cbrt(Float::EPSILON) * (xi.abs() + 1.0);
            xw[i] = xi + hi;
            let f_plus = self.evaluate(&xw, user_data)?;
            xw[i] = xi - hi;
            let f_minus = self.evaluate(&xw, user_data)?;
            xw[i] = xi;
            grad[i] = (f_plus - f_minus) / (2.0 * hi);
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
        let n = x.len();
        let h: DVector<Float> = x
            .iter()
            .map(|&xi| Float::cbrt(Float::EPSILON) * (xi.abs() + 1.0))
            .collect::<Vec<_>>()
            .into();
        let mut hess = DMatrix::zeros(n, n);
        let mut g_plus = DMatrix::zeros(n, n);
        let mut g_minus = DMatrix::zeros(n, n);
        let mut xw = x.clone();
        // g+ and g- are such that
        // g+[(i, j)] = g[i](x + h_je_j) and
        // g-[(i, j)] = g[i](x - h_je_j)
        for i in 0..x.len() {
            let xi = xw[i];
            xw[i] = xi + h[i];
            g_plus.set_column(i, &self.gradient(&xw, user_data)?);
            xw[i] = xi - h[i];
            g_minus.set_column(i, &self.gradient(&xw, user_data)?);
            xw[i] = xi;
            hess[(i, i)] = (g_plus[(i, i)] - g_minus[(i, i)]) / (2.0 * h[i]);
            for j in 0..i {
                hess[(i, j)] = ((g_plus[(i, j)] - g_minus[(i, j)]) / (4.0 * h[j]))
                    + ((g_plus[(j, i)] - g_minus[(j, i)]) / (4.0 * h[i]));
                hess[(j, i)] = hess[(i, j)];
            }
        }
        Ok(hess)
    }
}

/// A trait which describes a function $`f(\mathbb{R}^n) \to \mathbb{R}`$ representing the natural
/// logarithm of a probability density function.
///
/// Such a function may also take a `user_data: &mut U` field which can be used to pass external
/// arguments to the function during sampling or can be modified by the function itself.
///
/// The [`LogDensity`] trait takes a generic `U` representing the type of user data/arguments
/// and a generic `E` representing any possible errors that might be returned during function
/// execution.
pub trait LogDensity<U = (), E = Infallible> {
    /// The input space consumed by the log-density function
    type Input;
    /// The log of the evaluation of the density function at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// [`std::convert::Infallible`] if the function evaluation never fails.
    fn log_density(&self, x: &Self::Input, user_data: &mut U) -> Result<Float, E>;
}

#[cfg(test)]
mod tests {
    use crate::{
        traits::{CostFunction, Gradient},
        DVector, Float,
    };
    use approx::assert_relative_eq;
    use std::convert::Infallible;

    struct TestFunction;
    impl CostFunction for TestFunction {
        type Input = DVector<Float>;
        fn evaluate(&self, x: &DVector<Float>, _: &mut ()) -> Result<Float, Infallible> {
            #[allow(clippy::suboptimal_flops)]
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
    fn test_cost_function_covariance_and_std() {
        use crate::core::utils::hessian_to_covariance;
        let x: DVector<Float> = DVector::from_vec(vec![1.0, 2.0]);
        let hessian = TestFunction.hessian(&x, &mut ()).unwrap();
        let cov = hessian_to_covariance(&hessian).unwrap();
        assert_relative_eq!(cov[(0, 0)], 0.5, epsilon = Float::EPSILON.cbrt());
        assert_relative_eq!(cov[(1, 1)], 0.5, epsilon = Float::EPSILON.cbrt());
    }
}
