use std::{cmp::Ordering, collections::VecDeque};

use nalgebra::{DMatrix, DVector, RealField};
use num::Float;

use crate::{convert, Algorithm, Bound, Function, Status};

use super::line_search::{LineSearch, StrongWolfeLineSearch};

/// The L-BFGS (Limited memory Broyden-Fletcher-Goldfarb-Shanno) algorithm.
///
/// This minimization [`Algorithm`] is a quasi-Newton minimizer which approximates the inverse of
/// the Hessian matrix using the L-BFGS update step. The BFGS algorithm is described in detail in Chapter
/// 6 of "Numerical Optimization"[^1] (pages 136-143).
///
/// [^1]: [Numerical Optimization. Springer New York, 2006. doi: 10.1007/978-0-387-40065-5.](https://doi.org/10.1007/978-0-387-40065-5)

#[allow(clippy::upper_case_acronyms)]
pub struct LBFGSB<T, U, E> {
    status: Status<T>,
    x: DVector<T>,
    g: DVector<T>,
    l: DVector<T>,
    u: DVector<T>,
    m_mat: DMatrix<T>,
    w_mat: DMatrix<T>,
    theta: T,
    g_tolerance: T,
    line_search: Box<dyn LineSearch<T, U, E>>,
    m: usize,
    y_store: VecDeque<DVector<T>>,
    s_store: VecDeque<DVector<T>>,
    max_step: T,
}

impl<T, U, E> LBFGSB<T, U, E>
where
    T: Float + RealField,
{
    /// Set the value $`\varepsilon_g`$ for which $`||g_\text{proj}||_{\inf} < \varepsilon_g`$ will
    /// successfully terminate the algorithm (default = `1e-5`).
    pub const fn with_g_tolerance(mut self, tol: T) -> Self {
        self.g_tolerance = tol;
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

impl<T, U, E> Default for LBFGSB<T, U, E>
where
    T: Float + RealField + Default,
{
    fn default() -> Self {
        Self {
            status: Default::default(),
            x: Default::default(),
            g: Default::default(),
            l: Default::default(),
            u: Default::default(),
            m_mat: Default::default(),
            w_mat: Default::default(),
            theta: T::one(),
            g_tolerance: convert!(1e-5, T),
            line_search: Box::new(StrongWolfeLineSearch::default()),
            m: 10,
            y_store: VecDeque::default(),
            s_store: VecDeque::default(),
            max_step: convert!(1e8, T),
        }
    }
}

impl<T, U, E> LBFGSB<T, U, E>
where
    T: RealField + Float,
{
    /// For Equation 6.1
    fn get_inf_norm_projected_gradient(&self) -> T {
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
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .unwrap_or_else(T::zero)
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
        l_mat.set_diagonal(&DVector::from_element(m, T::zero()));
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
    fn get_xcp_c_free_indices(&self) -> (DVector<T>, DVector<T>, Vec<usize>) {
        // Equations 4.1 and 4.2
        let (t, mut d): (DVector<T>, DVector<T>) = (0..self.g.len())
            .map(|i| {
                let ti = if self.g[i] < T::zero() {
                    (self.x[i] - self.u[i]) / self.g[i]
                } else if self.g[i] > T::zero() {
                    (self.x[i] - self.l[i]) / self.g[i]
                } else {
                    T::infinity()
                };
                let di = if ti < T::epsilon() {
                    T::zero()
                } else {
                    -self.g[i]
                };
                (ti, di)
            })
            .collect();
        let mut x_cp = self.x.clone();
        let mut free_indices: Vec<usize> = (0..t.len()).filter(|&i| t[i] > T::zero()).collect();
        free_indices.sort_by(|&a, &b| t[a].partial_cmp(&t[b]).unwrap_or(Ordering::Equal));
        let free_indices = VecDeque::from(free_indices);
        let mut t_old = T::zero();
        let mut i_free = 0;
        let mut b = free_indices[0];
        let mut t_b = t[b];
        let mut dt_b = t_b - t_old;

        let mut p = self.w_mat.transpose() * &d;
        let mut c = DVector::zeros(p.len());
        let mut df = -d.dot(&d);
        let mut ddf = -self.theta * df - p.dot(&(&self.m_mat * &p));
        let mut dt_min = -df / ddf;

        while dt_min >= dt_b && i_free < free_indices.len() {
            // b is the index of the smallest positive nonzero element of t, so d_b is never zero!
            x_cp[b] = if d[b] > T::zero() {
                self.u[b]
            } else {
                self.l[b]
            };
            let z_b = x_cp[b] - self.x[b];
            c += p.scale(dt_b);
            let g_b = self.g[b];
            let w_b_tr = self.w_mat.row(b);
            df += dt_b * ddf + g_b * (g_b + self.theta * z_b - w_b_tr.dot(&(&self.m_mat * &c)));
            ddf -= g_b
                * (self.theta * g_b
                    - convert!(2, T) * w_b_tr.dot(&(&self.m_mat * &p))
                    - g_b * w_b_tr.dot(&(&self.m_mat * w_b_tr.transpose())));
            // min here
            p += w_b_tr.transpose().scale(g_b);
            d[b] = T::zero();
            dt_min = -df / ddf;
            t_old = t_b;
            i_free += 1;
            if i_free < free_indices.len() {
                b = free_indices[i_free];
                t_b = t[b];
                dt_b = t_b - t_old;
            } else {
                t_b = T::infinity();
            }
        }
        dt_min = Float::max(dt_min, T::zero());
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
        x_cp: &DVector<T>,
        c: &DVector<T>,
        free_indices: &[usize],
    ) -> DVector<T> {
        let z_mat = DMatrix::from_fn(self.x.len(), free_indices.len(), |i, j| {
            if i == free_indices[j] {
                T::one()
            } else {
                T::zero()
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
        let mut alpha_star = T::one();
        for i in 0..free_indices.len() {
            let i_free = free_indices[i];
            alpha_star = if d_hat_u[i] > T::zero() {
                Float::min(alpha_star, (self.u[i_free] - x_cp[i_free]) / d_hat_u[i])
            } else if d_hat_u[i] < T::zero() {
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
    fn compute_step_direction(&self) -> DVector<T> {
        let (xcp, c, free_indices) = self.get_xcp_c_free_indices();
        let x_bar = if free_indices.is_empty() {
            xcp
        } else {
            self.direct_primal_min(&xcp, &c, &free_indices)
        };
        x_bar - &self.x
    }
    fn compute_max_step(&self, d: &DVector<T>) -> T {
        let mut max_step = self.max_step;
        for i in 0..self.x.len() {
            max_step = if d[i] > T::zero() {
                Float::min(max_step, (self.u[i] - self.x[i]) / d[i])
            } else if d[i] < T::zero() {
                Float::min(max_step, (self.l[i] - self.x[i]) / d[i])
            } else {
                max_step
            }
        }
        max_step
    }
}

impl<T, U, E> Algorithm<T, U, E> for LBFGSB<T, U, E>
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
        self.l = DVector::from_element(x0.len(), T::neg_infinity());
        self.u = DVector::from_element(x0.len(), T::infinity());
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
        self.g = DVector::from_vec(func.gradient(self.x.as_slice(), user_data)?);
        self.status.inc_n_g_evals();
        self.status.update_position((
            self.x.data.as_vec().to_vec(),
            func.evaluate(self.x.as_slice(), user_data)?,
        ));
        self.status.inc_n_f_evals();
        // self.update_w_mat_m_mat();
        self.w_mat = DMatrix::zeros(self.x.len(), 1);
        self.m_mat = DMatrix::zeros(1, 1);
        Ok(())
    }

    fn step(
        &mut self,
        _i_step: usize,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        let d = self.compute_step_direction();
        let max_step = self.compute_max_step(&d);
        let (_valid, alpha, f_kp1, g_kp1) = self.line_search.search(
            &self.x,
            &d,
            Some(max_step),
            func,
            bounds,
            user_data,
            &mut self.status,
        )?;
        let dx = d.scale(alpha);
        let grad_kp1_vec = DVector::from_vec(g_kp1);
        let dg = &grad_kp1_vec - &self.g;
        let sy = dx.dot(&dg);
        let yy = dg.dot(&dg);
        if sy > T::epsilon() * yy {
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
        self.status
            .update_position((self.x.data.as_vec().to_vec(), f_kp1));
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        _func: &dyn Function<T, U, E>,
        _bounds: Option<&Vec<Bound<T>>>,
        _user_data: &mut U,
    ) -> Result<bool, E> {
        if self.get_inf_norm_projected_gradient() < self.g_tolerance {
            self.status.set_converged();
            self.status
                .update_message("PROJECTED GRADIENT WITHIN TOLERANCE");
        }
        Ok(self.status.converged)
    }

    fn get_status(&self) -> &Status<T> {
        &self.status
    }
}