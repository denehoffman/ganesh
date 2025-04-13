use nalgebra::DVector;

use crate::{
    core::Status,
    traits::{CostFunction, LineSearch},
    Float,
};

/// A line search which implements Algorithms 3.5 and 3.6 from Nocedal and Wright's book "Numerical
/// Optimization"[^1] (pages 60-61). This algorithm upholds the strong Wolfe conditions.
///
/// [^1]: [Numerical Optimization. Springer New York, 2006. doi: 10.1007/978-0-387-40065-5.](https://doi.org/10.1007/978-0-387-40065-5)
#[derive(Clone)]
pub struct StrongWolfeLineSearch {
    max_iters: usize,
    max_zoom: usize,
    c1: Float,
    c2: Float,
    use_bounds: bool,
}

impl Default for StrongWolfeLineSearch {
    fn default() -> Self {
        Self {
            max_iters: 100,
            max_zoom: 100,
            c1: 1e-4,
            c2: 0.9,
            use_bounds: false,
        }
    }
}

impl StrongWolfeLineSearch {
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
    /// Use the bounded forms of the function evaluators, transforming the function inputs in a
    /// nonlinear way to convert between external and internal parameters. See
    /// [`Bound`](`crate::Bound`) for more
    /// details.
    pub const fn with_bounds_transformation(mut self) -> Self {
        self.use_bounds = true;
        self
    }
}

impl StrongWolfeLineSearch {
    fn f_eval<U, E>(
        &self,
        func: &dyn CostFunction<U, E>,
        x: &DVector<Float>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<Float, E> {
        status.inc_n_f_evals();
        func.evaluate(x.as_slice(), user_data)
    }
    fn g_eval<U, E>(
        &self,
        func: &dyn CostFunction<U, E>,
        x: &DVector<Float>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<DVector<Float>, E> {
        status.inc_n_g_evals();
        func.gradient(x.as_slice(), user_data).map(DVector::from)
    }
    #[allow(clippy::too_many_arguments)]
    fn zoom<U, E>(
        &self,
        func: &dyn CostFunction<U, E>,
        x0: &DVector<Float>,
        user_data: &mut U,
        f0: Float,
        g0: &DVector<Float>,
        p: &DVector<Float>,
        alpha_lo: Float,
        alpha_hi: Float,
        status: &mut Status,
    ) -> Result<(bool, Float, Float, DVector<Float>), E> {
        let mut alpha_lo = alpha_lo;
        let mut alpha_hi = alpha_hi;
        let dphi0 = g0.dot(p);
        let mut i = 0;
        loop {
            let alpha_i = (alpha_lo + alpha_hi) / 2.0;
            let x = x0 + p.scale(alpha_i);
            let f_i = self.f_eval(func, &x, user_data, status)?;
            let x_lo = x0 + p.scale(alpha_lo);
            let f_lo = self.f_eval(func, &x_lo, user_data, status)?;
            let valid = if (f_i > (self.c1 * alpha_i).mul_add(dphi0, f0)) || (f_i >= f_lo) {
                alpha_hi = alpha_i;
                false
            } else {
                let g_i = self.g_eval(func, &x, user_data, status)?;
                let dphi = g_i.dot(p);
                if Float::abs(dphi) <= -self.c2 * dphi0 {
                    return Ok((true, alpha_i, f_i, g_i));
                }
                if dphi * (alpha_hi - alpha_lo) >= 0.0 {
                    alpha_hi = alpha_lo;
                }
                alpha_lo = alpha_i;
                true
            };
            i += 1;
            if i > self.max_zoom {
                let g_i = self.g_eval(func, &x, user_data, status)?;
                return Ok((valid, alpha_i, f_i, g_i));
            }
        }
    }
}

impl<U, E> LineSearch<U, E> for StrongWolfeLineSearch {
    fn search(
        &mut self,
        x0: &DVector<Float>,
        p: &DVector<Float>,
        max_step: Option<Float>,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(bool, Float, Float, DVector<Float>), E> {
        let f0 = self.f_eval(func, x0, user_data, status)?;
        let g0 = self.g_eval(func, x0, user_data, status)?;
        let alpha_max = max_step.map_or(1.0, |alpha_max| alpha_max);
        let mut alpha_im1 = 0.0;
        let mut alpha_i = 1.0;
        let mut f_im1 = f0;
        let dphi0 = g0.dot(p);
        let mut i = 0;
        loop {
            let x = x0 + p.scale(alpha_i);
            let f_i = self.f_eval(func, &x, user_data, status)?;
            if (f_i > self.c1.mul_add(dphi0, f0)) || (i > 1 && f_i >= f_im1) {
                return self.zoom(func, x0, user_data, f0, &g0, p, alpha_im1, alpha_i, status);
            }
            let g_i = self.g_eval(func, &x, user_data, status)?;
            let dphi = g_i.dot(p);
            if Float::abs(dphi) <= self.c2 * Float::abs(dphi0) {
                return Ok((true, alpha_i, f_i, g_i));
            }
            if dphi >= 0.0 {
                return self.zoom(func, x0, user_data, f0, &g0, p, alpha_i, alpha_im1, status);
            }
            alpha_im1 = alpha_i;
            f_im1 = f_i;
            alpha_i += 0.8 * (alpha_max - alpha_i);
            i += 1;
            if i > self.max_iters {
                return Ok((false, alpha_i, f_i, g_i));
            }
        }
    }
}
