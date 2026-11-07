//! Scalar- and linear-algebra-generic numerical problem traits.

use crate::core::{LinearAlgebra, Matrix, NalgebraProvider, RealScalar, Vector};
use std::convert::Infallible;

/// Objective value and gradient returned together.
pub type ValueGradient<T, B> = (T, Vector<T, B>);

/// Objective value, gradient, and Hessian returned together.
pub type ValueGradientHessian<T, B> = (T, Vector<T, B>, Matrix<T, B>);
/// Gradient and Hessian returned together.
pub type GradientHessian<T, B> = (Vector<T, B>, Matrix<T, B>);

/// The outcome of evaluating a candidate that the algorithm has yet to accept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Trial<V> {
    /// All requested quantities are usable.
    Valid(V),
    /// The candidate is inadmissible. The message describes the rejected evaluation.
    Rejected(String),
}

impl<V> Trial<V> {
    /// Transform a valid result while preserving a rejection.
    pub fn map<W>(self, map: impl FnOnce(V) -> W) -> Trial<W> {
        match self {
            Self::Valid(value) => Trial::Valid(map(value)),
            Self::Rejected(reason) => Trial::Rejected(reason),
        }
    }

    pub(crate) fn validate(self, finite: impl FnOnce(&V) -> bool) -> Self {
        match self {
            Self::Valid(value) if !finite(&value) => {
                Self::Rejected("nonfinite trial evaluation".into())
            }
            outcome => outcome,
        }
    }
}

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

    /// Describe an evaluation error that makes only this trial inadmissible.
    /// Errors are fatal by default. This hook is used only for trial evaluations.
    fn trial_rejection(&self, _error: &E) -> Option<String> {
        None
    }

    /// Evaluate an uncommitted candidate objective.
    ///
    /// # Errors
    /// Returns any error that the problem has not classified as trial rejection.
    fn trial_value(&self, x: &Vector<T, B>, args: &U) -> Result<Trial<T>, E> {
        classify_trial(
            self.evaluate(x, args),
            |error| self.trial_rejection(error),
            |v| v.is_finite(),
        )
    }
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

    /// Evaluate the gradient of an uncommitted candidate.
    ///
    /// # Errors
    /// Returns any error that the problem has not classified as trial rejection.
    fn trial_gradient(&self, x: &Vector<T, B>, args: &U) -> Result<Trial<Vector<T, B>>, E> {
        classify_trial(
            self.gradient(x, args),
            |error| self.trial_rejection(error),
            finite_vector,
        )
    }

    /// Evaluate the objective and gradient of an uncommitted candidate.
    ///
    /// # Errors
    /// Returns any error that the problem has not classified as trial rejection.
    fn trial_value_gradient(
        &self,
        x: &Vector<T, B>,
        args: &U,
    ) -> Result<Trial<ValueGradient<T, B>>, E> {
        classify_trial(
            self.evaluate_with_gradient(x, args),
            |error| self.trial_rejection(error),
            |(v, g)| v.is_finite() && finite_vector(g),
        )
    }

    /// Evaluate the derivatives of an uncommitted candidate.
    ///
    /// # Errors
    /// Returns any error that the problem has not classified as trial rejection.
    fn trial_gradient_hessian(
        &self,
        x: &Vector<T, B>,
        args: &U,
    ) -> Result<Trial<GradientHessian<T, B>>, E> {
        classify_trial(
            self.gradient_with_hessian(x, args),
            |error| self.trial_rejection(error),
            |(g, h)| finite_vector(g) && finite_matrix(h),
        )
    }
}

fn classify_trial<V, E>(
    result: Result<V, E>,
    rejection: impl FnOnce(&E) -> Option<String>,
    finite: impl FnOnce(&V) -> bool,
) -> Result<Trial<V>, E> {
    match result {
        Ok(value) => Ok(Trial::Valid(value).validate(finite)),
        Err(error) => {
            rejection(&error).map_or_else(|| Err(error), |reason| Ok(Trial::Rejected(reason)))
        }
    }
}

pub(crate) fn finite_vector<T: RealScalar, B: LinearAlgebra<T>>(value: &Vector<T, B>) -> bool {
    (0..value.len()).all(|index| value.get(index).is_finite())
}

pub(crate) fn finite_matrix<T: RealScalar, B: LinearAlgebra<T>>(value: &Matrix<T, B>) -> bool {
    (0..value.rows()).all(|row| (0..value.cols()).all(|col| value.get(row, col).is_finite()))
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
