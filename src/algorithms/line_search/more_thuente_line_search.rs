use crate::{
    algorithms::gradient::GradientStatus,
    core::Bounds,
    traits::{linesearch::LineSearchOutput, Boundable, Gradient, LineSearch},
    DVector, Float,
};

/// A line search which implements Algorithms 3.5 and 3.6 from Nocedal and Wright's book "Numerical
/// Optimization"[^1] (pages 60-61).
///
/// This algorithm upholds the strong Wolfe conditions. This is the algorithm originally described by Moré and Thuente in [^2].
///
/// [^1]: [Numerical Optimization. Springer New York, 2006. doi: 10.1007/978-0-387-40065-5.](https://doi.org/10.1007/978-0-387-40065-5)
///
/// [^2]: [J. J. Moré and D. J. Thuente, "Line search algorithms with guaranteed sufficient decrease," ACM Trans. Math. Softw., vol. 20, no. 3, pp. 286–307, Sept. 1994, doi: 10.1145/192115.192132.](https://doi.org/10.1145/192115.192132)
#[derive(Clone)]
pub struct MoreThuenteLineSearch {
    max_iters: usize,
    max_zoom: usize,
    c1: Float,
    c2: Float,
}

impl Default for MoreThuenteLineSearch {
    fn default() -> Self {
        Self {
            max_iters: 100,
            max_zoom: 100,
            c1: 1e-4,
            c2: 0.9,
        }
    }
}

impl MoreThuenteLineSearch {
    /// Set the maximum allowed iterations of the algorithm (defaults to 100).
    pub const fn with_max_iterations(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }
    /// Set the maximum allowed iterations of the internal zoom algorithm (defaults to 100).
    pub const fn with_max_zoom(mut self, max_zoom: usize) -> Self {
        self.max_zoom = max_zoom;
        self
    }
    /// Set the first control parameter, used in the Armijo condition evaluation (defaults to
    /// 1e-4).
    ///
    /// # Panics
    ///
    /// This method will panic if the condition $`0 < c_1 < c_2 < 1`$ is not met.
    pub fn with_c1(mut self, c1: Float) -> Self {
        assert!(0.0 < c1);
        assert!(c1 < self.c2);
        self.c1 = c1;
        self
    }
    /// Set the second control parameter, used in the second Wolfe condition (defaults to 0.9).
    ///
    /// # Panics
    ///
    /// This method will panic if the condition $`0 < c_1 < c_2 < 1`$ is not met.
    pub fn with_c2(mut self, c2: Float) -> Self {
        assert!(1.0 > c2);
        assert!(c2 > self.c1);
        self.c2 = c2;
        self
    }
}

impl MoreThuenteLineSearch {
    fn f_eval<U, E>(
        &self,
        func: &dyn Gradient<U, E>,
        x: &DVector<Float>,
        bounds: Option<&Bounds>,
        args: &U,
        status: &mut GradientStatus,
    ) -> Result<Float, E> {
        status.inc_n_f_evals();
        func.evaluate(&x.constrain_to(bounds), args)
    }
    fn g_eval<U, E>(
        &self,
        func: &dyn Gradient<U, E>,
        x: &DVector<Float>,
        bounds: Option<&Bounds>,
        args: &U,
        status: &mut GradientStatus,
    ) -> Result<DVector<Float>, E> {
        status.inc_n_g_evals();
        func.gradient(&x.constrain_to(bounds), args)
    }
    #[allow(clippy::too_many_arguments)]
    fn zoom<U, E>(
        &self,
        func: &dyn Gradient<U, E>,
        x0: &DVector<Float>,
        bounds: Option<&Bounds>,
        args: &U,
        f0: Float,
        g0: &DVector<Float>,
        p: &DVector<Float>,
        alpha_lo: Float,
        alpha_hi: Float,
        status: &mut GradientStatus,
    ) -> Result<Result<LineSearchOutput, LineSearchOutput>, E> {
        let mut alpha_lo = alpha_lo;
        let mut alpha_hi = alpha_hi;
        let dphi0 = g0.dot(p);
        let mut i = 0;
        loop {
            let alpha_i = (alpha_lo + alpha_hi) / 2.0;
            let x = x0 + p.scale(alpha_i);
            let f_i = self.f_eval(func, &x, bounds, args, status)?;
            let x_lo = x0 + p.scale(alpha_lo);
            let f_lo = self.f_eval(func, &x_lo, bounds, args, status)?;
            let valid = if (f_i > (self.c1 * alpha_i).mul_add(dphi0, f0)) || (f_i >= f_lo) {
                alpha_hi = alpha_i;
                false
            } else {
                let g_i = self.g_eval(func, &x, bounds, args, status)?;
                let dphi = g_i.dot(p);
                if Float::abs(dphi) <= -self.c2 * dphi0 {
                    return Ok(Ok(LineSearchOutput {
                        alpha: alpha_i,
                        fx: f_i,
                        g: g_i,
                    }));
                }
                if dphi * (alpha_hi - alpha_lo) >= 0.0 {
                    alpha_hi = alpha_lo;
                }
                alpha_lo = alpha_i;
                true
            };
            i += 1;
            if i > self.max_zoom {
                let g_i = self.g_eval(func, &x, bounds, args, status)?;
                if valid {
                    return Ok(Ok(LineSearchOutput {
                        alpha: alpha_i,
                        fx: f_i,
                        g: g_i,
                    }));
                } else {
                    return Ok(Err(LineSearchOutput {
                        alpha: alpha_i,
                        fx: f_i,
                        g: g_i,
                    }));
                }
            }
        }
    }
}

impl<U, E> LineSearch<GradientStatus, U, E> for MoreThuenteLineSearch {
    fn search(
        &mut self,
        x0: &DVector<Float>,
        p: &DVector<Float>,
        max_step: Option<Float>,
        problem: &dyn Gradient<U, E>,
        bounds: Option<&Bounds>,
        args: &U,
        status: &mut GradientStatus,
    ) -> Result<Result<LineSearchOutput, LineSearchOutput>, E> {
        let f0 = self.f_eval(problem, x0, bounds, args, status)?;
        let g0 = self.g_eval(problem, x0, bounds, args, status)?;
        let alpha_max = max_step.unwrap_or(1.0); // TODO: 1e5?
        let mut alpha_im1 = 0.0;
        let mut alpha_i = 1.0;
        let mut f_im1 = f0;
        let dphi0 = g0.dot(p);
        let mut i = 0;
        loop {
            let x = x0 + p.scale(alpha_i);
            let f_i = self.f_eval(problem, &x, bounds, args, status)?;
            if (f_i > self.c1.mul_add(dphi0, f0)) || (i > 1 && f_i >= f_im1) {
                return self.zoom(
                    problem, x0, bounds, args, f0, &g0, p, alpha_im1, alpha_i, status,
                );
            }
            let g_i = self.g_eval(problem, &x, bounds, args, status)?;
            let dphi = g_i.dot(p);
            if Float::abs(dphi) <= self.c2 * Float::abs(dphi0) {
                return Ok(Ok(LineSearchOutput {
                    alpha: alpha_i,
                    fx: f_i,
                    g: g_i,
                }));
            }
            if dphi >= 0.0 {
                return self.zoom(
                    problem, x0, bounds, args, f0, &g0, p, alpha_i, alpha_im1, status,
                );
            }
            alpha_im1 = alpha_i;
            f_im1 = f_i;
            alpha_i += 0.8 * (alpha_max - alpha_i);
            i += 1;
            if i > self.max_iters {
                return Ok(Err(LineSearchOutput {
                    alpha: alpha_i,
                    fx: f_i,
                    g: g_i,
                }));
            }
        }
    }
}
