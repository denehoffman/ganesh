use nalgebra::{DMatrix, DVector, RealField};
use num::Float;

use crate::{Algorithm, Bound, Function, Status};

use super::line_search::{LineSearch, StrongWolfeLineSearch};

/// A terminator for the [`BFGS`] [`Algorithm`] which causes termination when the change in the
/// function evaluation becomes smaller than the given absolute tolerance. In such a case, the [`Status`]
/// of the [`Minimizer`](`crate::Minimizer`) will be set as converged with the message "GRADIENT
/// CONVERGED".
pub struct BFGSFTerminator<T> {
    tol_f_abs: T,
}
impl<T> BFGSFTerminator<T>
where
    T: RealField,
{
    fn update_convergence(&self, fx_current: T, fx_previous: T, status: &mut Status<T>) {
        if (fx_previous - fx_current).abs() < self.tol_f_abs {
            status.set_converged();
            status.update_message("F_EVAL CONVERGED");
        }
    }
}

/// A terminator for the [`BFGS`] [`Algorithm`] which causes termination when the magnitude of the
/// gradient vector becomes smaller than the given absolute tolerance. In such a case, the [`Status`]
/// of the [`Minimizer`](`crate::Minimizer`) will be set as converged with the message "GRADIENT
/// CONVERGED".
pub struct BFGSGTerminator<T> {
    tol_g_abs: T,
}
impl<T> BFGSGTerminator<T>
where
    T: RealField,
{
    fn update_convergence(&self, gradient: &DVector<T>, status: &mut Status<T>) {
        if gradient.dot(gradient).sqrt() < self.tol_g_abs {
            status.set_converged();
            status.update_message("GRADIENT CONVERGED");
        }
    }
}

/// The BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm.
///
/// This minimization [`Algorithm`] is a quasi-Newton minimizer which approximates the inverse of
/// the Hessian matrix using the BFGS update step. This algorithm is described in detail in Chapter
/// 6 of "Numerical Optimization"[^1] (pages 136-143).
///
/// [^1]: [Numerical Optimization. Springer New York, 2006. doi: 10.1007/978-0-387-40065-5.](https://doi.org/10.1007/978-0-387-40065-5)

#[allow(clippy::upper_case_acronyms)]
pub struct BFGS<T, U, E> {
    status: Status<T>,
    x: DVector<T>,
    g: DVector<T>,
    p: DVector<T>,
    h_inv: DMatrix<T>,
    f_previous: T,
    terminator_f: BFGSFTerminator<T>,
    terminator_g: BFGSGTerminator<T>,
    line_search: Box<dyn LineSearch<T, U, E>>,
}

impl<T, U, E> BFGS<T, U, E>
where
    T: Float + RealField,
{
    /// Set the termination condition concerning the function values.
    pub const fn with_terminator_f(mut self, term: BFGSFTerminator<T>) -> Self {
        self.terminator_f = term;
        self
    }

    /// Set the termination condition concerning the gradient values.
    pub const fn with_terminator_g(mut self, term: BFGSGTerminator<T>) -> Self {
        self.terminator_g = term;
        self
    }
    /// Set the line search local method for local optimization of step size. Defaults to a line
    /// search which satisfies the strong Wolfe conditions, [`StrongWolfeLineSearch`]. Note that in
    /// general, this should only use [`LineSearch`] algorithms which satisfy the Wolfe conditions.
    /// Using the Armijo condition alone will lead to slower convergence.
    pub fn with_line_search<LS: LineSearch<T, U, E> + 'static>(mut self, line_search: LS) -> Self {
        self.line_search = Box::new(line_search);
        self
    }
}

impl<T, U, E> Default for BFGS<T, U, E>
where
    T: Float + RealField + Default,
{
    fn default() -> Self {
        Self {
            status: Default::default(),
            x: Default::default(),
            g: Default::default(),
            p: Default::default(),
            h_inv: Default::default(),
            f_previous: T::infinity(),
            terminator_f: BFGSFTerminator {
                tol_f_abs: Float::cbrt(T::epsilon()),
            },
            terminator_g: BFGSGTerminator {
                tol_g_abs: Float::cbrt(T::epsilon()),
            },
            line_search: Box::new(StrongWolfeLineSearch::default()),
        }
    }
}

impl<T, U, E> BFGS<T, U, E>
where
    T: Float + RealField,
{
    fn update_h_inv(&mut self, n: usize, s: &DVector<T>, y: &DVector<T>) {
        let rho = Float::recip(y.dot(s));
        let m_left = DMatrix::identity(n, n) - (s * y.transpose()).scale(rho);
        let m_right = DMatrix::identity(n, n) - (y * s.transpose()).scale(rho);
        let m_add = (s * s.transpose()).scale(rho);
        self.h_inv = (m_left * &self.h_inv * m_right) + m_add;
    }
}

impl<T, U, E> Algorithm<T, U, E> for BFGS<T, U, E>
where
    T: RealField + Float,
{
    fn initialize(
        &mut self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.h_inv = DMatrix::identity(x0.len(), x0.len());
        self.x = DVector::from_vec(Bound::to_unbounded(x0, bounds));
        self.g = DVector::from_vec(func.gradient_bounded(self.x.as_slice(), bounds, user_data)?);
        self.status.inc_n_g_evals();
        self.status.update_position((
            Bound::to_bounded(self.x.as_slice(), bounds),
            func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?,
        ));
        self.status.inc_n_f_evals();
        Ok(())
    }

    fn step(
        &mut self,
        _i_step: usize,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.p = -&self.h_inv * &self.g;
        let (valid, alpha, f_kp1, g_kp1) = self.line_search.search(
            &self.x,
            &self.p,
            None,
            func,
            bounds,
            user_data,
            &mut self.status,
        )?;
        if valid {
            let s = self.p.scale(alpha);
            let grad_kp1_vec = DVector::from_vec(g_kp1);
            let y = &grad_kp1_vec - &self.g;
            let n = self.x.len();
            self.update_h_inv(n, &s, &y);
            self.x += s;
            self.g = grad_kp1_vec;
            self.status
                .update_position((Bound::to_bounded(self.x.as_slice(), bounds), f_kp1));
        } else {
            self.status.set_converged();
        }
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<bool, E> {
        let f_current = func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?;
        self.terminator_f
            .update_convergence(f_current, self.f_previous, &mut self.status);
        self.f_previous = f_current;
        self.terminator_g
            .update_convergence(&self.g, &mut self.status);
        Ok(self.status.converged)
    }

    fn get_status(&self) -> &Status<T> {
        &self.status
    }
}
