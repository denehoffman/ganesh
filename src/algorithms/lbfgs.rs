use std::collections::VecDeque;

use nalgebra::{DVector, RealField};
use num::Float;

use crate::{Algorithm, Bound, Function, Status};

use super::line_search::{LineSearch, StrongWolfeLineSearch};

/// A terminator for the [`LBFGS`] [`Algorithm`] which causes termination when the magnitude of the
/// gradient vector becomes smaller than the given absolute tolerance. In such a case, the [`Status`]
/// of the [`Minimizer`](`crate::Minimizer`) will be set as converged with the message "GRADIENT
/// CONVERGED".
pub struct LBFGSGTerminator<T> {
    tol_g_abs: T,
}
impl<T> LBFGSGTerminator<T>
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

/// The L-BFGS (Limited memory Broyden-Fletcher-Goldfarb-Shanno) algorithm.
///
/// This minimization [`Algorithm`] is a quasi-Newton minimizer which approximates the inverse of
/// the Hessian matrix using the L-BFGS update step. The BFGS algorithm is described in detail in Chapter
/// 6 of "Numerical Optimization"[^1] (pages 136-143).
///
/// [^1]: [Numerical Optimization. Springer New York, 2006. doi: 10.1007/978-0-387-40065-5.](https://doi.org/10.1007/978-0-387-40065-5)

#[allow(clippy::upper_case_acronyms)]
pub struct LBFGS<T, U, E> {
    status: Status<T>,
    x: DVector<T>,
    g: DVector<T>,
    p: DVector<T>,
    terminator_g: LBFGSGTerminator<T>,
    line_search: Box<dyn LineSearch<T, U, E>>,
    m: usize,
    y_store: VecDeque<DVector<T>>,
    s_store: VecDeque<DVector<T>>,
}

impl<T, U, E> LBFGS<T, U, E>
where
    T: Float + RealField,
{
    /// Set the termination condition concerning the gradient values.
    pub const fn with_terminator_g(mut self, term: LBFGSGTerminator<T>) -> Self {
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

impl<T, U, E> Default for LBFGS<T, U, E>
where
    T: Float + RealField + Default,
{
    fn default() -> Self {
        Self {
            status: Default::default(),
            x: Default::default(),
            g: Default::default(),
            p: Default::default(),
            terminator_g: LBFGSGTerminator {
                tol_g_abs: Float::cbrt(T::epsilon()),
            },
            line_search: Box::new(StrongWolfeLineSearch::default()),
            m: 10,
            y_store: VecDeque::default(),
            s_store: VecDeque::default(),
        }
    }
}

impl<T, U, E> LBFGS<T, U, E>
where
    T: RealField + Float,
{
    fn g_approx(&self) -> DVector<T> {
        let m = self.s_store.len();
        let mut q = self.g.clone();
        let mut a = vec![T::zero(); m];
        let rho = DVector::from_fn(m, |j, _| T::one() / self.y_store[j].dot(&self.s_store[j]));
        for i in 0..m {
            a[m - 1 - i] = self.s_store[m - 1 - i].dot(&q).scale(rho[m - 1 - i]);
            q -= self.y_store[m - 1 - i].scale(a[m - 1 - i]);
        }
        q = q.scale(
            self.s_store[m - 1].dot(&self.y_store[m - 1])
                / self.y_store[m - 1].dot(&self.y_store[m - 1]),
        );
        for i in 0..m {
            let beta = rho[i] * self.y_store[i].dot(&q);
            q += self.s_store[i].scale(a[i] - beta);
        }
        q
    }
}

impl<T, U, E> Algorithm<T, U, E> for LBFGS<T, U, E>
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
        i_step: usize,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        if i_step == 0 {
            self.p = -self.g.clone();
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
                let dx = self.p.scale(alpha);
                self.s_store.push_back(dx.clone());
                let grad_kp1_vec = DVector::from_vec(g_kp1);
                self.y_store.push_back(&grad_kp1_vec - &self.g);
                self.x += dx;
                self.g = grad_kp1_vec;
                self.status
                    .update_position((Bound::to_bounded(self.x.as_slice(), bounds), f_kp1));
            } else {
                self.status.set_converged();
            }
        } else {
            self.p = -self.g_approx();
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
                let dx = self.p.scale(alpha);
                self.s_store.push_back(dx.clone());
                let grad_kp1_vec = DVector::from_vec(g_kp1);
                self.y_store.push_back(&grad_kp1_vec - &self.g);
                if self.s_store.len() > self.m {
                    self.s_store.pop_front();
                    self.y_store.pop_front();
                }
                self.x += dx;
                self.g = grad_kp1_vec;
                self.status
                    .update_position((Bound::to_bounded(self.x.as_slice(), bounds), f_kp1));
            } else {
                self.status.set_converged();
            }
        }
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        _func: &dyn Function<T, U, E>,
        _bounds: Option<&Vec<Bound<T>>>,
        _user_data: &mut U,
    ) -> Result<bool, E> {
        self.terminator_g
            .update_convergence(&self.g, &mut self.status);
        Ok(self.status.converged)
    }

    fn get_status(&self) -> &Status<T> {
        &self.status
    }
}