use std::collections::VecDeque;

use nalgebra::{DMatrix, DVector};

use super::Algorithm;
use crate::{Bound, Float, Function, Status};

use super::line_search::{LineSearch, StrongWolfeLineSearch};

/// A terminator for the [`LBFGSB`] [`Algorithm`] which causes termination when the change in the
/// function evaluation becomes smaller than the given absolute tolerance. In such a case, the [`Status`]
/// of the [`Minimizer`](`crate::Minimizer`) will be set as converged with the message "GRADIENT
/// CONVERGED".
#[derive(Clone)]
pub struct LBFGSBFTerminator;
impl LBFGSBFTerminator {
    fn update_convergence(
        &self,
        fx_current: Float,
        fx_previous: Float,
        status: &mut Status,
        eps_abs: Float,
    ) {
        if (fx_previous - fx_current).abs() < eps_abs {
            status.set_converged();
            status.update_message("F_EVAL CONVERGED");
        }
    }
}

/// A terminator for the [`LBFGSB`] [`Algorithm`] which causes termination when the magnitude of the
/// gradient vector becomes smaller than the given absolute tolerance. In such a case, the [`Status`]
/// of the [`Minimizer`](`crate::Minimizer`) will be set as converged with the message "GRADIENT
/// CONVERGED".
#[derive(Clone)]
pub struct LBFGSBGTerminator;
impl LBFGSBGTerminator {
    fn update_convergence(&self, gradient: &DVector<Float>, status: &mut Status, eps_abs: Float) {
        if gradient.dot(gradient).sqrt() < eps_abs {
            status.set_converged();
            status.update_message("GRADIENT CONVERGED");
        }
    }
}

/// Error modes for [`LBFGSB`] [`Algorithm`].
#[derive(Default, Clone)]
pub enum LBFGSBErrorMode {
    /// Computes the exact Hessian matrix via finite differences.
    #[default]
    ExactHessian,
    /// Skip Hessian computation (use this when error evaluation is not important).
    Skip,
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
pub struct LBFGSB<U, E> {
    x: DVector<Float>,
    g: DVector<Float>,
    l: DVector<Float>,
    u: DVector<Float>,
    m_mat: DMatrix<Float>,
    w_mat: DMatrix<Float>,
    theta: Float,
    f_previous: Float,
    terminator_f: LBFGSBFTerminator,
    terminator_g: LBFGSBGTerminator,
    eps_f_abs: Float,
    eps_g_abs: Float,
    tol_g_abs: Float,
    line_search: Box<dyn LineSearch<U, E>>,
    m: usize,
    y_store: VecDeque<DVector<Float>>,
    s_store: VecDeque<DVector<Float>>,
    max_step: Float,
    error_mode: LBFGSBErrorMode,
}

impl<U, E> LBFGSB<U, E> {
    /// Set the termination condition concerning the function values.
    pub const fn with_terminator_f(mut self, term: LBFGSBFTerminator) -> Self {
        self.terminator_f = term;
        self
    }
    /// Set the termination condition concerning the gradient values.
    pub const fn with_terminator_g(mut self, term: LBFGSBGTerminator) -> Self {
        self.terminator_g = term;
        self
    }
    /// Set the absolute f-convergence tolerance (default = `MACH_EPS^(1/2)`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\epsilon <= 0`$.
    pub fn with_eps_f_abs(mut self, value: Float) -> Self {
        assert!(value > 0.0);
        self.eps_f_abs = value;
        self
    }
    /// Set the absolute g-convergence tolerance (default = `MACH_EPS^(1/3)`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\epsilon <= 0`$.
    pub fn with_eps_g_abs(mut self, value: Float) -> Self {
        assert!(value > 0.0);
        self.eps_g_abs = value;
        self
    }
    /// Set the value $`\varepsilon_g`$ for which $`||g_\text{proj}||_{\inf} < \varepsilon_g`$ will
    /// successfully terminate the algorithm (default = `1e-5`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\varepsilon <= 0`$.
    pub fn with_tol_g_abs(mut self, tol: Float) -> Self {
        assert!(tol > 0.0);
        self.tol_g_abs = tol;
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
    pub const fn with_error_mode(mut self, error_mode: LBFGSBErrorMode) -> Self {
        self.error_mode = error_mode;
        self
    }
}

impl<U, E> Default for LBFGSB<U, E> {
    fn default() -> Self {
        Self {
            x: Default::default(),
            g: Default::default(),
            l: Default::default(),
            u: Default::default(),
            m_mat: Default::default(),
            w_mat: Default::default(),
            theta: 1.0,
            f_previous: Float::INFINITY,
            terminator_f: LBFGSBFTerminator,
            terminator_g: LBFGSBGTerminator,
            eps_f_abs: Float::sqrt(Float::EPSILON),
            eps_g_abs: Float::cbrt(Float::EPSILON),
            tol_g_abs: Float::cbrt(Float::EPSILON),
            line_search: Box::<StrongWolfeLineSearch>::default(),
            m: 10,
            y_store: VecDeque::default(),
            s_store: VecDeque::default(),
            max_step: 1e8,
            error_mode: Default::default(),
        }
    }
}

impl<U, E> LBFGSB<U, E> {
    /// For Equation 6.1
    fn get_inf_norm_projected_gradient(&self) -> Float {
        let x_minus_g = &self.x - &self.g;
        (0..x_minus_g.len())
            .map(|i| {
                if self.x[i] - self.g[i] < self.l[i] {
                    Float::abs(self.l[i] - self.x[i])
                } else if self.x[i] - self.g[i] > self.u[i] {
                    Float::abs(self.u[i] - self.x[i])
                } else {
                    Float::abs(self.g[i])
                }
            })
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or(0.0)
    }
    /// Equations 3.3, 3.4, 3.5, 3.6
    #[allow(clippy::expect_used)]
    fn update_w_mat_m_mat(&mut self) {
        let m = self.s_store.len();
        let n = self.x.len();
        let s_mat = DMatrix::from_fn(n, m, |i, j| self.s_store[j][i]);
        let y_mat = DMatrix::from_fn(n, m, |i, j| self.y_store[j][i]);

        // W
        self.w_mat = DMatrix::zeros(n, 2 * m);
        let mut y_view = self.w_mat.view_mut((0, 0), (n, m));
        y_view += &y_mat;
        let mut theta_s_view = self.w_mat.view_mut((0, m), (n, m));
        theta_s_view += s_mat.scale(self.theta);
        let theta_s_tr_s = (s_mat.transpose() * &s_mat).scale(self.theta);

        // M
        let s_tr_y = s_mat.transpose() * &y_mat;
        let d_vec = s_tr_y.diagonal();
        let mut l_mat = s_tr_y.lower_triangle();
        l_mat.set_diagonal(&DVector::from_element(m, 0.0));
        let mut m_mat_inv = DMatrix::zeros(2 * m, 2 * m);
        let mut d_view = m_mat_inv.view_mut((0, 0), (m, m));
        d_view.set_diagonal(&(-&d_vec));
        let mut l_view = m_mat_inv.view_mut((m, 0), (m, m));
        l_view += &l_mat;
        let mut l_tr_view = m_mat_inv.view_mut((0, m), (m, m));
        l_tr_view += l_mat.transpose();
        let mut theta_s_tr_s_view = m_mat_inv.view_mut((m, m), (m, m));
        theta_s_tr_s_view += theta_s_tr_s;
        self.m_mat = m_mat_inv
            .try_inverse()
            .expect("Error: Something has gone horribly wrong, inversion of M failed!");
    }
    fn get_xcp_c_free_indices(&self) -> (DVector<Float>, DVector<Float>, Vec<usize>) {
        // Equations 4.1 and 4.2
        let (t, mut d): (DVector<Float>, DVector<Float>) = (0..self.g.len())
            .map(|i| {
                let ti = if self.g[i] < 0.0 {
                    (self.x[i] - self.u[i]) / self.g[i]
                } else if self.g[i] > 0.0 {
                    (self.x[i] - self.l[i]) / self.g[i]
                } else {
                    Float::INFINITY
                };
                let di = if ti < Float::EPSILON { 0.0 } else { -self.g[i] };
                (ti, di)
            })
            .unzip();
        let mut x_cp = self.x.clone();
        let mut free_indices: Vec<usize> = (0..t.len()).filter(|&i| t[i] > 0.0).collect();
        free_indices.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
        let free_indices = VecDeque::from(free_indices);
        let mut t_old = 0.0;
        let mut i_free = 0;
        let mut b = free_indices[0];
        let mut t_b = t[b];
        let mut dt_b = t_b - t_old;

        let mut p = self.w_mat.transpose() * &d;
        let mut c = DVector::zeros(p.len());
        let mut df = -d.dot(&d);
        let mut ddf = (-self.theta).mul_add(df, -p.dot(&(&self.m_mat * &p)));
        let mut dt_min = -df / ddf;

        while dt_min >= dt_b && i_free < free_indices.len() {
            // b is the index of the smallest positive nonzero element of t, so d_b is never zero!
            x_cp[b] = if d[b] > 0.0 { self.u[b] } else { self.l[b] };
            let z_b = x_cp[b] - self.x[b];
            c += p.scale(dt_b);
            let g_b = self.g[b];
            let w_b_tr = self.w_mat.row(b);
            df += dt_b.mul_add(
                ddf,
                g_b * (self.theta.mul_add(z_b, g_b) - w_b_tr.transpose().dot(&(&self.m_mat * &c))),
            );
            ddf -= g_b
                * self.theta.mul_add(
                    g_b,
                    (-2.0 as Float).mul_add(
                        w_b_tr.transpose().dot(&(&self.m_mat * &p)),
                        -(g_b * w_b_tr.transpose().dot(&(&self.m_mat * w_b_tr.transpose()))),
                    ),
                );
            // min here
            p += w_b_tr.transpose().scale(g_b);
            d[b] = 0.0;
            dt_min = -df / ddf;
            t_old = t_b;
            i_free += 1;
            if i_free < free_indices.len() {
                b = free_indices[i_free];
                t_b = t[b];
                dt_b = t_b - t_old;
            } else {
                t_b = Float::INFINITY;
            }
        }
        dt_min = Float::max(dt_min, 0.0);
        t_old += dt_min;
        // for i in free_indices.iter() {
        for i in 0..self.x.len() {
            if t[i] >= t_b {
                x_cp[i] += t_old * d[i];
            }
        }
        let free_indices = (0..free_indices.len())
            .filter(|&i| x_cp[i] < self.u[i] && x_cp[i] > self.l[i])
            .collect();
        c += p.scale(dt_min);
        // let vec_free_indices = (0..x_cp.len())
        //     .filter(|&i| x_cp[i] < self.u[i] && x_cp[i] > self.l[i])
        //     .collect();
        (x_cp, c, free_indices)
    }
    // Direct primal method (page 1199, equations 5.4), returns x_bar such that the search
    // direction is d = x_bar - x
    #[allow(clippy::expect_used)]
    fn direct_primal_min(
        &self,
        x_cp: &DVector<Float>,
        c: &DVector<Float>,
        free_indices: &[usize],
    ) -> DVector<Float> {
        let z_mat = DMatrix::from_fn(self.x.len(), free_indices.len(), |i, j| {
            if i == free_indices[j] {
                1.0
            } else {
                0.0
            }
        });
        let r_hat_c = z_mat.transpose()
            * (&self.g + (x_cp - &self.x).scale(self.theta) - &self.w_mat * &self.m_mat * c);
        let w_tr_z_mat = self.w_mat.transpose() * &z_mat;
        let n_mat = DMatrix::identity(self.m_mat.shape().0, self.m_mat.shape().1)
            - (&self.m_mat * (&w_tr_z_mat * w_tr_z_mat.transpose())).unscale(self.theta);
        let n_mat_inv = n_mat
            .try_inverse()
            .expect("Error: Something has gone horribly wrong, inversion of N^{-1} failed!");
        let v = n_mat_inv * &self.m_mat * &self.w_mat.transpose() * &z_mat * &r_hat_c;
        let d_hat_u =
            -(r_hat_c + (w_tr_z_mat.transpose() * v).unscale(self.theta)).unscale(self.theta);
        // The minus sign here is missing in equation 5.11, this is a typo!
        let mut alpha_star = 1.0;
        for i in 0..free_indices.len() {
            let i_free = free_indices[i];
            alpha_star = if d_hat_u[i] > 0.0 {
                Float::min(alpha_star, (self.u[i_free] - x_cp[i_free]) / d_hat_u[i])
            } else if d_hat_u[i] < 0.0 {
                Float::min(alpha_star, (self.l[i_free] - x_cp[i_free]) / d_hat_u[i])
            } else {
                alpha_star
            }
        }
        let mut x_bar = x_cp.clone();
        let d_hat_star = d_hat_u.scale(alpha_star);
        let z_d_hat_star = &z_mat * d_hat_star;
        for i in free_indices {
            x_bar[*i] += z_d_hat_star[*i]
        }
        x_bar
    }
    fn compute_step_direction(&self) -> DVector<Float> {
        let (xcp, c, free_indices) = self.get_xcp_c_free_indices();
        let x_bar = if free_indices.is_empty() {
            xcp
        } else {
            self.direct_primal_min(&xcp, &c, &free_indices)
        };
        x_bar - &self.x
    }
    fn compute_max_step(&self, d: &DVector<Float>) -> Float {
        let mut max_step = self.max_step;
        for i in 0..self.x.len() {
            max_step = if d[i] > 0.0 {
                Float::min(max_step, (self.u[i] - self.x[i]) / d[i])
            } else if d[i] < 0.0 {
                Float::min(max_step, (self.l[i] - self.x[i]) / d[i])
            } else {
                max_step
            }
        }
        max_step
    }
}

impl<U, E> Algorithm<U, E> for LBFGSB<U, E> {
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
        x0: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        self.f_previous = Float::INFINITY;
        self.theta = 1.0;
        self.l = DVector::from_element(x0.len(), Float::NEG_INFINITY);
        self.u = DVector::from_element(x0.len(), Float::INFINITY);
        if let Some(bounds_vec) = bounds {
            for (i, bound) in bounds_vec.iter().enumerate() {
                match bound {
                    Bound::NoBound => {}
                    Bound::LowerBound(lb) => self.l[i] = *lb,
                    Bound::UpperBound(ub) => self.u[i] = *ub,
                    Bound::LowerAndUpperBound(lb, ub) => {
                        self.l[i] = *lb;
                        self.u[i] = *ub;
                    }
                }
            }
        }
        self.x = DVector::from_fn(x0.len(), |i, _| {
            if x0[i] < self.l[i] {
                self.l[i]
            } else if x0[i] > self.u[i] {
                self.u[i]
            } else {
                x0[i]
            }
        });
        self.g = func.gradient(self.x.as_slice(), user_data)?;
        status.inc_n_g_evals();
        status.update_position((self.x.clone(), func.evaluate(self.x.as_slice(), user_data)?));
        status.inc_n_f_evals();
        self.w_mat = DMatrix::zeros(self.x.len(), 1);
        self.m_mat = DMatrix::zeros(1, 1);
        Ok(())
    }

    fn step(
        &mut self,
        _i_step: usize,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        let d = self.compute_step_direction();
        let max_step = self.compute_max_step(&d);
        let (valid, alpha, f_kp1, g_kp1) =
            self.line_search
                .search(&self.x, &d, Some(max_step), func, user_data, status)?;
        if valid {
            let dx = d.scale(alpha);
            let grad_kp1_vec = g_kp1;
            let dg = &grad_kp1_vec - &self.g;
            let sy = dx.dot(&dg);
            let yy = dg.dot(&dg);
            if sy > Float::EPSILON * yy {
                self.s_store.push_back(dx.clone());
                self.y_store.push_back(dg);
                self.theta = yy / sy;
                if self.s_store.len() > self.m {
                    self.s_store.pop_front();
                    self.y_store.pop_front();
                }
                self.update_w_mat_m_mat();
            }
            self.x += dx;
            self.g = grad_kp1_vec;
            status.update_position((self.x.clone(), f_kp1));
        } else {
            // reboot
            self.s_store.clear();
            self.y_store.clear();
            self.w_mat = DMatrix::zeros(self.x.len(), 1);
            self.m_mat = DMatrix::zeros(1, 1);
            self.theta = 1.0;
        }
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<bool, E> {
        let f_current = func.evaluate(self.x.as_slice(), user_data)?;
        self.terminator_f
            .update_convergence(f_current, self.f_previous, status, self.eps_f_abs);
        self.f_previous = f_current;
        self.terminator_g
            .update_convergence(&self.g, status, self.eps_g_abs);
        if self.get_inf_norm_projected_gradient() < self.tol_g_abs {
            status.set_converged();
            status.update_message("PROJECTED GRADIENT WITHIN TOLERANCE");
        }
        Ok(status.converged)
    }

    fn postprocessing(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        match self.error_mode {
            LBFGSBErrorMode::ExactHessian => {
                let hessian = func.hessian(self.x.as_slice(), user_data)?;
                status.set_hess(&hessian);
            }
            LBFGSBErrorMode::Skip => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use approx::assert_relative_eq;

    use crate::{
        abort_signal::CtrlCAbortSignal, test_functions::Rosenbrock, traits::AbortSignal, Float,
        Minimizer,
    };

    use super::LBFGSB;

    #[test]
    fn test_lbfgsb() -> Result<(), Infallible> {
        let algo = LBFGSB::default();
        let mut m = Minimizer::new(Box::new(algo), 2);
        let problem = Rosenbrock { n: 2 };
        m.minimize(
            &problem,
            &[-2.0, 2.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(
            &problem,
            &[2.0, 2.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(
            &problem,
            &[2.0, -2.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(
            &problem,
            &[-2.0, -2.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(
            &problem,
            &[0.0, 0.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(
            &problem,
            &[1.0, 1.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }

    #[test]
    fn test_bounded_lbfgsb() -> Result<(), Infallible> {
        let algo = LBFGSB::default();
        let mut m = Minimizer::new(Box::new(algo), 2).with_bounds(vec![(-4.0, 4.0), (-4.0, 4.0)]);
        let problem = Rosenbrock { n: 2 };
        m.minimize(
            &problem,
            &[-2.0, 2.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(
            &problem,
            &[2.0, 2.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(
            &problem,
            &[2.0, -2.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(
            &problem,
            &[-2.0, -2.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(
            &problem,
            &[0.0, 0.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.minimize(
            &problem,
            &[1.0, 1.0],
            &mut (),
            CtrlCAbortSignal::new().boxed(),
        )?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }
}
