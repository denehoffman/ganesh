use std::collections::VecDeque;

use nalgebra::DVector;

use crate::{Algorithm, Bound, Float, Function, Status};

use super::line_search::{LineSearch, StrongWolfeLineSearch};

/// A terminator for the [`LBFGS`] [`Algorithm`] which causes termination when the change in the
/// function evaluation becomes smaller than the given absolute tolerance. In such a case, the [`Status`]
/// of the [`Minimizer`](`crate::Minimizer`) will be set as converged with the message "GRADIENT
/// CONVERGED".
#[derive(Clone)]
pub struct LBFGSFTerminator {
    /// Absolute tolerance $`\varepsilon`$.
    pub tol_f_abs: Float,
}
impl LBFGSFTerminator {
    fn update_convergence(&self, fx_current: Float, fx_previous: Float, status: &mut Status) {
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
pub struct LBFGSGTerminator {
    /// Absolute tolerance $`\varepsilon`$.
    pub tol_g_abs: Float,
}
impl LBFGSGTerminator {
    fn update_convergence(&self, gradient: &DVector<Float>, status: &mut Status) {
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
pub struct LBFGS<U, E> {
    x: DVector<Float>,
    g: DVector<Float>,
    f_previous: Float,
    terminator_f: LBFGSFTerminator,
    terminator_g: LBFGSGTerminator,
    line_search: Box<dyn LineSearch<U, E>>,
    m: usize,
    y_store: VecDeque<DVector<Float>>,
    s_store: VecDeque<DVector<Float>>,
    max_step: Float,
    error_mode: LBFGSErrorMode,
}

impl<U, E> LBFGS<U, E> {
    /// Set the termination condition concerning the function values.
    pub const fn with_terminator_f(mut self, term: LBFGSFTerminator) -> Self {
        self.terminator_f = term;
        self
    }
    /// Set the termination condition concerning the gradient values.
    pub const fn with_terminator_g(mut self, term: LBFGSGTerminator) -> Self {
        self.terminator_g = term;
        self
    }
    /// Set the line search local method for local optimization of step size. Defaults to a line
    /// search which satisfies the strong Wolfe conditions, [`StrongWolfeLineSearch`]. Note that in
    /// general, this should only use [`LineSearch`] algorithms which satisfy the Wolfe conditions.
    /// Using the Armijo condition alone will lead to slower convergence.
    pub fn with_line_search<LS: LineSearch<U, E> + 'static>(mut self, line_search: LS) -> Self {
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

impl<U, E> Default for LBFGS<U, E> {
    fn default() -> Self {
        Self {
            x: Default::default(),
            g: Default::default(),
            f_previous: Float::INFINITY,
            terminator_f: LBFGSFTerminator {
                tol_f_abs: Float::sqrt(Float::EPSILON),
            },
            terminator_g: LBFGSGTerminator {
                tol_g_abs: Float::cbrt(Float::EPSILON),
            },
            line_search: Box::<StrongWolfeLineSearch>::default(),
            m: 10,
            y_store: VecDeque::default(),
            s_store: VecDeque::default(),
            max_step: 1e8,
            error_mode: Default::default(),
        }
    }
}

impl<U, E> LBFGS<U, E> {
    fn g_approx(&self) -> DVector<Float> {
        let m = self.s_store.len();
        let mut q = self.g.clone();
        if m < 1 {
            return q;
        }
        let mut a = vec![0.0; m];
        let rho = DVector::from_fn(m, |j, _| 1.0 / self.y_store[j].dot(&self.s_store[j]));
        for i in 0..m {
            a[m - 1 - i] = self.s_store[m - 1 - i].dot(&q) * rho[m - 1 - i];
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

impl<U, E> Algorithm<U, E> for LBFGS<U, E> {
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
        x0: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        self.f_previous = Float::INFINITY;
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
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
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
            if sy > Float::EPSILON * yy {
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
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
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
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
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

    use approx::assert_relative_eq;

    use crate::{test_functions::Rosenbrock, Float, Minimizer};

    use super::LBFGS;

    #[test]
    fn test_lbfgs() -> Result<(), Infallible> {
        let algo = LBFGS::default();
        let mut m = Minimizer::new(Box::new(algo), 2);
        let problem = Rosenbrock { n: 2 };
        m.minimize(&problem, &[-2.0, 2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(&problem, &[2.0, 2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(&problem, &[2.0, -2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(&problem, &[-2.0, -2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(&problem, &[0.0, 0.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(&problem, &[1.0, 1.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }
}
