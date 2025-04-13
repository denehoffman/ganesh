use nalgebra::{DMatrix, DVector};

use crate::{core::Bound, Float};

/// A trait which describes a function $`f(\mathbb{R}^n) \to \mathbb{R}`$
///
/// Such a function may also take a `user_data: &mut UD` field which can be used to pass external
/// arguments to the function during minimization, or can be modified by the function itself.
///
/// The `Function` trait takes a generic `T` which represents a numeric scalar, a generic `U`
/// representing the type of user data/arguments, and a generic `E` representing any possible
/// errors that might be returned during function execution.
///
/// There is also a default implementation of a gradient function which uses a central
/// finite-difference method to evaluate derivatives. If an exact gradient is known, it can be used
/// to speed up gradient-dependent algorithms.
pub trait CostFunction<U, E> {
    /// The evaluation of the function at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    fn evaluate(&self, x: &[Float], user_data: &mut U) -> Result<Float, E>;
    /// The evaluation of the function at a point `x` with the given arguments/user data. This
    /// function assumes `x` is an internal, unbounded vector, but performs a coordinate transform
    /// to bound `x` when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    fn evaluate_bounded(
        &self,
        x: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
    ) -> Result<Float, E> {
        self.evaluate(Bound::to_bounded(x, bounds).as_slice(), user_data)
    }
    /// The evaluation of the gradient at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn gradient(&self, x: &[Float], user_data: &mut U) -> Result<DVector<Float>, E> {
        let n = x.len();
        let x = DVector::from_column_slice(x);
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
            let f_plus = self.evaluate(x_plus.as_slice(), user_data)?;
            let f_minus = self.evaluate(x_minus.as_slice(), user_data)?;
            grad[i] = (f_plus - f_minus) / (2.0 * h[i]);
        }
        Ok(grad)
    }
    /// The evaluation of the gradient at a point `x` with the given arguments/user data. This
    /// function assumes `x` is an internal, unbounded vector, but performs a coordinate transform
    /// to bound `x` when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn gradient_bounded(
        &self,
        x: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
    ) -> Result<DVector<Float>, E> {
        self.gradient(Bound::to_bounded(x, bounds).as_slice(), user_data)
    }

    /// The evaluation of the hessian at a point `x` with the given arguments/user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn hessian(&self, x: &[Float], user_data: &mut U) -> Result<DMatrix<Float>, E> {
        let x = DVector::from_column_slice(x);
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
            g_plus.set_column(i, &self.gradient(x_plus.as_slice(), user_data)?);
            g_minus.set_column(i, &self.gradient(x_minus.as_slice(), user_data)?);
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
    /// The evaluation of the hessian at a point `x` with the given arguments/user data. This
    /// function assumes `x` is an internal, unbounded vector, but performs a coordinate transform
    /// to bound `x` when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn hessian_bounded(
        &self,
        x: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
    ) -> Result<DMatrix<Float>, E> {
        self.hessian(Bound::to_bounded(x, bounds).as_slice(), user_data)
    }
}
