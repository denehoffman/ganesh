//! Scalar- and linear-algebra-generic numerical problem traits.

use crate::core::{LinearAlgebra, Matrix, NalgebraProvider, RealScalar, Vector};
use std::convert::Infallible;

/// Objective value, gradient, and Hessian returned together.
pub type ValueGradientHessian<T, B> = (T, Vector<T, B>, Matrix<T, B>);
/// Gradient and Hessian returned together.
pub type GradientHessian<T, B> = (Vector<T, B>, Matrix<T, B>);

/// A scalar- and linear-algebra-generic objective function.
pub trait ScalarCostFunction<
    T: RealScalar = f64,
    B: LinearAlgebra<T> = NalgebraProvider,
    U = (),
    E = Infallible,
>
{
    /// Evaluate the objective at `x`.
    ///
    /// # Errors
    /// Returns an error when objective evaluation fails.
    fn evaluate(&self, x: &Vector<T, B>, args: &U) -> Result<T, E>;
}

/// A scalar- and linear-algebra-generic objective with derivatives.
pub trait ScalarGradient<
    T: RealScalar = f64,
    B: LinearAlgebra<T> = NalgebraProvider,
    U = (),
    E = Infallible,
>: ScalarCostFunction<T, B, U, E>
{
    /// Evaluate the gradient, using central finite differences by default.
    ///
    /// # Errors
    /// Returns an error when objective evaluation fails.
    fn gradient(&self, x: &Vector<T, B>, args: &U) -> Result<Vector<T, B>, E> {
        let mut work = x.clone();
        let mut gradient = Vector::<T, B>::zeros(x.len());
        let two = T::literal(2.0);
        for i in 0..x.len() {
            let xi = work.get(i);
            let step = T::epsilon().cbrt() * (xi.abs() + T::one());
            work.set(i, xi + step);
            let plus = self.evaluate(&work, args)?;
            work.set(i, xi - step);
            let minus = self.evaluate(&work, args)?;
            work.set(i, xi);
            gradient.set(i, (plus - minus) / (two * step));
        }
        Ok(gradient)
    }

    /// Evaluate the Hessian by centrally differencing gradients.
    ///
    /// # Errors
    /// Returns an error when derivative evaluation fails.
    fn hessian(&self, x: &Vector<T, B>, args: &U) -> Result<Matrix<T, B>, E> {
        let n = x.len();
        let mut work = x.clone();
        let mut hessian = Matrix::<T, B>::zeros(n, n);
        let two = T::literal(2.0);
        for column in 0..n {
            let xi = work.get(column);
            let step = T::epsilon().cbrt() * (xi.abs() + T::one());
            work.set(column, xi + step);
            let plus = self.gradient(&work, args)?;
            work.set(column, xi - step);
            let minus = self.gradient(&work, args)?;
            work.set(column, xi);
            for row in 0..n {
                hessian.set(row, column, (plus.get(row) - minus.get(row)) / (two * step));
            }
        }
        // Numerical noise can make the finite-difference Hessian slightly asymmetric.
        for row in 0..n {
            for column in 0..row {
                let value = (hessian.get(row, column) + hessian.get(column, row)) / two;
                hessian.set(row, column, value);
                hessian.set(column, row, value);
            }
        }
        Ok(hessian)
    }

    /// Evaluate the objective and gradient.
    ///
    /// # Errors
    /// Returns an error when either evaluation fails.
    fn evaluate_with_gradient(&self, x: &Vector<T, B>, args: &U) -> Result<(T, Vector<T, B>), E> {
        Ok((self.evaluate(x, args)?, self.gradient(x, args)?))
    }

    /// Evaluate the gradient and Hessian.
    ///
    /// # Errors
    /// Returns an error when either evaluation fails.
    fn gradient_with_hessian(
        &self,
        x: &Vector<T, B>,
        args: &U,
    ) -> Result<GradientHessian<T, B>, E> {
        Ok((self.gradient(x, args)?, self.hessian(x, args)?))
    }

    /// Evaluate the objective, gradient, and Hessian.
    ///
    /// # Errors
    /// Returns an error when any evaluation fails.
    fn evaluate_with_gradient_and_hessian(
        &self,
        x: &Vector<T, B>,
        args: &U,
    ) -> Result<ValueGradientHessian<T, B>, E> {
        let (value, gradient) = self.evaluate_with_gradient(x, args)?;
        Ok((value, gradient, self.hessian(x, args)?))
    }
}

/// A scalar- and linear-algebra-generic log probability density.
pub trait ScalarLogDensity<
    T: RealScalar = f64,
    B: LinearAlgebra<T> = NalgebraProvider,
    U = (),
    E = Infallible,
>
{
    /// Evaluate the log density at `x`.
    ///
    /// # Errors
    /// Returns an error when density evaluation fails.
    fn log_density(&self, x: &Vector<T, B>, args: &U) -> Result<T, E>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Quadratic;

    impl<T, B> ScalarCostFunction<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn evaluate(&self, x: &Vector<T, B>, _: &()) -> Result<T, Infallible> {
            Ok(x.dot(x) + T::one())
        }
    }

    impl<T, B> ScalarGradient<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
    }

    fn check<T: RealScalar + nalgebra::RealField>() {
        let x = Vector::<T>::from_vec(vec![<T as RealScalar>::one(), T::literal(2.0)]);
        let gradient = Quadratic.gradient(&x, &()).unwrap();
        let hessian = Quadratic.hessian(&x, &()).unwrap();
        let tolerance = RealScalar::sqrt(<T as RealScalar>::epsilon()) * T::literal(20.0);
        assert!(RealScalar::abs(gradient.get(0) - T::literal(2.0)) < tolerance);
        assert!(RealScalar::abs(gradient.get(1) - T::literal(4.0)) < tolerance);
        assert!(RealScalar::abs(hessian.get(0, 0) - T::literal(2.0)) < RealScalar::cbrt(tolerance));
    }

    #[test]
    fn finite_differences_support_both_native_scalars() {
        check::<f32>();
        check::<f64>();
    }
}
