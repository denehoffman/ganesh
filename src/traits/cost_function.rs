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
        let mut hessian = DMatrix::zeros(n, n);
        let mut xw = x.clone();
        for i in 0..n {
            let xi = x[i];
            let hi = h[i];
            let inv2hi = 1.0 / (2.0 * hi);
            let inv4hi = 0.5 * inv2hi;
            xw[i] = xi + hi;
            let g_plus = self.gradient(&xw, user_data)?;
            xw[i] = xi - hi;
            let g_minus = self.gradient(&xw, user_data)?;
            xw[i] = xi;
            hessian[(i, i)] = (g_plus[i] - g_minus[i]) * inv2hi;
            for j in 0..i {
                let m = (g_plus[j] - g_minus[j]) * inv4hi;
                hessian[(i, j)] += m;
                hessian[(j, i)] = hessian[(i, j)];
            }
            for j in (i + 1)..n {
                let m = (g_plus[j] - g_minus[j]) * inv4hi;
                hessian[(j, i)] = m;
            }
        }
        Ok(hessian)
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
