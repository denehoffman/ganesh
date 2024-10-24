use std::collections::VecDeque;

use nalgebra::{DVector, RealField, Scalar};
use num::Float;

use crate::{convert, Algorithm, Bound, Function, Status};

use super::line_search::{LineSearch, StrongWolfeLineSearch};

/// A terminator for the [`LBFGS`] [`Algorithm`] which causes termination when the change in the
/// function evaluation becomes smaller than the given absolute tolerance. In such a case, the [`Status`]
/// of the [`Minimizer`](`crate::Minimizer`) will be set as converged with the message "GRADIENT
/// CONVERGED".
#[derive(Clone)]
pub struct LBFGSFTerminator<T> {
    /// Absolute tolerance $`\varepsilon`$.
    pub tol_f_abs: T,
}
impl<T> LBFGSFTerminator<T>
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

/// A terminator for the [`LBFGS`] [`Algorithm`] which causes termination when the magnitude of the
/// gradient vector becomes smaller than the given absolute tolerance. In such a case, the [`Status`]
/// of the [`Minimizer`](`crate::Minimizer`) will be set as converged with the message "GRADIENT
/// CONVERGED".
#[derive(Clone)]
pub struct LBFGSGTerminator<T> {
    /// Absolute tolerance $`\varepsilon`$.
    pub tol_g_abs: T,
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

/// Error modes for [`LBFGS`] [`Algorithm`].
#[derive(Default, Clone)]
pub enum LBFGSErrorMode {
    /// Computes the exact Hessian matrix via finite differences.
    #[default]
    ExactHessian,
}

/// The L-BFGS (Limited memory Broyden-Fletcher-Goldfarb-Shanno) algorithm.
///
/// This minimization [`Algorithm`] is a quasi-Newton minimizer which approximates the inverse of
/// the Hessian matrix using the L-BFGS update step. The BFGS algorithm is described in detail in Chapter
/// 6 of "Numerical Optimization"[^1] (pages 136-143).
///
/// [^1]: [Numerical Optimization. Springer New York, 2006. doi: 10.1007/978-0-387-40065-5.](https://doi.org/10.1007/978-0-387-40065-5)

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone)]
pub struct LBFGS<T: Scalar, U, E> {
    x: DVector<T>,
    g: DVector<T>,
    f_previous: T,
    terminator_f: LBFGSFTerminator<T>,
    terminator_g: LBFGSGTerminator<T>,
    line_search: Box<dyn LineSearch<T, U, E>>,
    m: usize,
    y_store: VecDeque<DVector<T>>,
    s_store: VecDeque<DVector<T>>,
    max_step: T,
    error_mode: LBFGSErrorMode,
}

impl<T, U, E> LBFGS<T, U, E>
where
    T: Float + RealField,
{
    /// Set the termination condition concerning the function values.
    pub const fn with_terminator_f(mut self, term: LBFGSFTerminator<T>) -> Self {
        self.terminator_f = term;
        self
    }
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
    /// Set the number of stored L-BFGS updator steps. A larger value might improve performance
    /// while sacrificing memory usage (default = `10`).
    pub const fn with_memory_limit(mut self, limit: usize) -> Self {
        self.m = limit;
        self
    }
    /// Set the mode for caluclating parameter errors at the end of the fit. Defaults to
    /// recalculating an exact finite-difference Hessian.
    pub const fn with_error_mode(mut self, error_mode: LBFGSErrorMode) -> Self {
        self.error_mode = error_mode;
        self
    }
}

impl<T, U, E> Default for LBFGS<T, U, E>
where
    T: Float + RealField + Default,
{
    fn default() -> Self {
        Self {
            x: Default::default(),
            g: Default::default(),
            f_previous: T::infinity(),
            terminator_f: LBFGSFTerminator {
                tol_f_abs: Float::sqrt(T::epsilon()),
            },
            terminator_g: LBFGSGTerminator {
                tol_g_abs: Float::cbrt(T::epsilon()),
            },
            line_search: Box::new(StrongWolfeLineSearch::default()),
            m: 10,
            y_store: VecDeque::default(),
            s_store: VecDeque::default(),
            max_step: convert!(1e8, T),
            error_mode: Default::default(),
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
        if m < 1 {
            return q;
        }
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
    T: RealField + Float + Default,
    U: Clone,
    E: Clone,
{
    fn initialize(
        &mut self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
        status: &mut Status<T>,
    ) -> Result<(), E> {
        self.f_previous = T::infinity();
        self.x = Bound::to_unbounded(x0, bounds);
        self.g = func.gradient_bounded(self.x.as_slice(), bounds, user_data)?;
        status.inc_n_g_evals();
        status.update_position((
            Bound::to_bounded(self.x.as_slice(), bounds),
            func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?,
        ));
        status.inc_n_f_evals();
        Ok(())
    }

    fn step(
        &mut self,
        _i_step: usize,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
        status: &mut Status<T>,
    ) -> Result<(), E> {
        let d = -self.g_approx();
        let (valid, alpha, f_kp1, g_kp1) = self.line_search.search(
            &self.x,
            &d,
            Some(self.max_step),
            func,
            bounds,
            user_data,
            status,
        )?;
        if valid {
            let dx = d.scale(alpha);
            let grad_kp1_vec = g_kp1;
            let dg = &grad_kp1_vec - &self.g;
            let sy = dx.dot(&dg);
            let yy = dg.dot(&dg);
            if sy > T::epsilon() * yy {
                self.s_store.push_back(dx.clone());
                self.y_store.push_back(dg);
                if self.s_store.len() > self.m {
                    self.s_store.pop_front();
                    self.y_store.pop_front();
                }
            }
            self.x += dx;
            self.g = grad_kp1_vec;
            status.update_position((Bound::to_bounded(self.x.as_slice(), bounds), f_kp1));
        } else {
            // reboot
            self.s_store.clear();
            self.y_store.clear();
        }
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
        status: &mut Status<T>,
    ) -> Result<bool, E> {
        let f_current = func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?;
        self.terminator_f
            .update_convergence(f_current, self.f_previous, status);
        self.f_previous = f_current;
        self.terminator_g.update_convergence(&self.g, status);
        Ok(status.converged)
    }

    fn postprocessing(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
        status: &mut Status<T>,
    ) -> Result<(), E> {
        match self.error_mode {
            LBFGSErrorMode::ExactHessian => {
                let hessian = func.hessian_bounded(self.x.as_slice(), bounds, user_data)?;
                status.set_hess(&hessian);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use float_cmp::assert_approx_eq;

    use crate::{prelude::*, test_functions::Rosenbrock};

    use super::LBFGS;

    #[test]
    fn test_lbfgs() -> Result<(), Infallible> {
        let algo = LBFGS::default();
        let mut m = Minimizer::new(&algo, 2);
        let problem = Rosenbrock { n: 2 };
        m.minimize(&problem, &[-2.0, 2.0], &mut ())?;
        assert!(m.status.converged);
        assert_approx_eq!(f64, m.status.fx, 0.0, epsilon = 1e-10);
        m.minimize(&problem, &[2.0, 2.0], &mut ())?;
        assert!(m.status.converged);
        assert_approx_eq!(f64, m.status.fx, 0.0, epsilon = 1e-8);
        m.minimize(&problem, &[2.0, -2.0], &mut ())?;
        assert!(m.status.converged);
        assert_approx_eq!(f64, m.status.fx, 0.0, epsilon = 1e-10);
        m.minimize(&problem, &[-2.0, -2.0], &mut ())?;
        assert!(m.status.converged);
        assert_approx_eq!(f64, m.status.fx, 0.0, epsilon = 1e-9);
        m.minimize(&problem, &[0.0, 0.0], &mut ())?;
        assert!(m.status.converged);
        assert_approx_eq!(f64, m.status.fx, 0.0, epsilon = 1e-10);
        m.minimize(&problem, &[1.0, 1.0], &mut ())?;
        assert!(m.status.converged);
        assert_approx_eq!(f64, m.status.fx, 0.0, epsilon = 1e-10);
        Ok(())
    }
}
