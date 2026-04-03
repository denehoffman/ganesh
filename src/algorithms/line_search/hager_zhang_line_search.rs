use crate::{
    algorithms::gradient::GradientStatus,
    core::Bounds,
    error::{GaneshError, GaneshResult},
    traits::{linesearch::LineSearchOutput, Gradient, LineSearch},
    DVector, Float,
};

/// The line search algorithm from Hager and Zhang (2006)[^1].
///
/// This is intended to be used with a conjugate gradient algorithm, but as it satisfies the strong Wolfe conditions, it can be used
/// with the [`LBFGSB`](`crate::algorithms::gradient::LBFGSB`) algorithm as well, although it tends to result in more function/gradient
/// evaluations than the [`MoreThuenteLineSearch`](`crate::algorithms::line_search::MoreThuenteLineSearch`).
///
/// [^1]: [W. W. Hager and H. Zhang, “A New Conjugate Gradient Method with Guaranteed Descent and an Efficient Line Search,” SIAM J. Optim., vol. 16, no. 1, pp. 170–192, Jan. 2005, doi: 10.1137/030601880.](https://doi.org/10.1137/030601880)

#[derive(Clone)]
pub struct HagerZhangLineSearch {
    delta: Float,
    sigma: Float,
    epsilon: Float,
    theta: Float,
    gamma: Float,
    max_iters: usize,
    max_bisects: usize,
}

impl Default for HagerZhangLineSearch {
    fn default() -> Self {
        Self {
            delta: 0.1,
            sigma: 0.9,
            epsilon: Float::EPSILON.cbrt(), // 10^-6 in the paper
            theta: 0.5,
            gamma: 0.66,
            max_iters: 100,
            max_bisects: 50,
        }
    }
}

impl HagerZhangLineSearch {
    /// Set the maximum allowed iterations of the algorithm (defaults to 100).
    pub const fn with_max_iterations(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }
    /// Set the parameter $`\delta`$ used in the Armijo condition evaluation (defaults to 0.1).
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `delta` does not satisfy `0 < delta < sigma`.
    pub fn with_delta(mut self, delta: Float) -> GaneshResult<Self> {
        if !(0.0 < delta && delta < self.sigma) {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch requires 0 < delta < sigma".to_string(),
            ));
        }
        self.delta = delta;
        Ok(self)
    }
    /// Set the parameter $`\sigma`$ used in the second Wolfe condition (defaults to 0.9).
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `sigma` does not satisfy `delta < sigma < 1`.
    pub fn with_sigma(mut self, sigma: Float) -> GaneshResult<Self> {
        if !(self.delta < sigma && sigma < 1.0) {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch requires delta < sigma < 1".to_string(),
            ));
        }
        self.sigma = sigma;
        Ok(self)
    }
    /// Set the parameter $`\delta`$ used in the Armijo condition evaluation (defaults to 0.1) and the parameter $`\sigma`$ used in the second Wolfe condition (defaults to 0.9) simultaneously.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `delta` and `sigma` do not satisfy `0 < delta < sigma < 1`.
    pub fn with_delta_sigma(mut self, delta: Float, sigma: Float) -> GaneshResult<Self> {
        if !(0.0 < delta && delta < sigma && sigma < 1.0) {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch requires 0 < delta < sigma < 1".to_string(),
            ));
        }
        self.delta = delta;
        self.sigma = sigma;
        Ok(self)
    }
    /// Set the tolerance parameter $`\epsilon`$ used in the approximate Wolfe termination
    /// conditions (defaults to `MACH_EPS^(1/3)`).
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `epsilon` is not strictly positive.
    pub fn with_epsilon(mut self, epsilon: Float) -> GaneshResult<Self> {
        if epsilon <= 0.0 {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch epsilon must be > 0".to_string(),
            ));
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Set the parameter $`\theta`$ used in interval updates (defaults to 0.5).
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `theta` is not in the interval `(0, 1)`.
    pub fn with_theta(mut self, theta: Float) -> GaneshResult<Self> {
        if !(0.0 < theta && theta < 1.0) {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch theta must be in (0, 1)".to_string(),
            ));
        }
        self.theta = theta;
        Ok(self)
    }

    /// Set the parameter $`\gamma`$ which determines when a bisection is performed (defaults to 0.66).
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `gamma` is not in the interval `(0, 1)`.
    pub fn with_gamma(mut self, gamma: Float) -> GaneshResult<Self> {
        if !(0.0 < gamma && gamma < 1.0) {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch gamma must be in (0, 1)".to_string(),
            ));
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// Set the maximum number of bisections allowed in the interval update step (defaults to 50).
    pub const fn with_max_bisects(mut self, max_bisects: usize) -> Self {
        self.max_bisects = max_bisects;
        self
    }
}

impl HagerZhangLineSearch {
    fn f_eval<U, E>(
        &self,
        func: &dyn Gradient<U, E>,
        x: &DVector<Float>,
        args: &U,
        status: &mut GradientStatus,
    ) -> Result<Float, E> {
        status.inc_n_f_evals();
        func.evaluate(x, args)
    }
    fn g_eval<U, E>(
        &self,
        func: &dyn Gradient<U, E>,
        x: &DVector<Float>,
        args: &U,
        status: &mut GradientStatus,
    ) -> Result<DVector<Float>, E> {
        status.inc_n_g_evals();
        func.gradient(x, args)
    }
    fn f_g_eval<U, E>(
        &self,
        func: &dyn Gradient<U, E>,
        x: &DVector<Float>,
        args: &U,
        status: &mut GradientStatus,
    ) -> Result<(Float, DVector<Float>), E> {
        status.inc_n_f_evals();
        status.inc_n_g_evals();
        func.evaluate_with_gradient(x, args)
    }
}

impl<U, E> LineSearch<GradientStatus, U, E> for HagerZhangLineSearch {
    fn search(
        &mut self,
        x0: &DVector<Float>,
        p: &DVector<Float>,
        max_step: Option<Float>,
        problem: &dyn Gradient<U, E>,
        _bounds: Option<&Bounds>,
        args: &U,
        status: &mut GradientStatus,
    ) -> Result<Result<LineSearchOutput, LineSearchOutput>, E> {
        let phi = |alpha: Float, st: &mut GradientStatus| -> Result<Float, E> {
            self.f_eval(problem, &(x0 + p.scale(alpha)), args, st)
        };
        let dphi_vec = |alpha: Float, st: &mut GradientStatus| -> Result<DVector<Float>, E> {
            self.g_eval(problem, &(x0 + p.scale(alpha)), args, st)
        };
        let phi_dphi_vec =
            |alpha: Float, st: &mut GradientStatus| -> Result<(Float, DVector<Float>), E> {
                self.f_g_eval(problem, &(x0 + p.scale(alpha)), args, st)
            };
        let dphi = |dphi_vec: &DVector<Float>| -> Float { dphi_vec.dot(p) };
        let secant = |a: Float, dphi_a: Float, b: Float, dphi_b: Float| -> Float {
            a.mul_add(dphi_b, -(b * dphi_a)) / (dphi_b - dphi_a)
        };
        let (phi_0, g_0) = phi_dphi_vec(0.0, status)?;
        let dphi_0 = dphi(&g_0);
        let epsilon_k = self.epsilon * phi_0.abs();
        let update =
            |a: Float, b: Float, c: Float, st: &mut GradientStatus| -> Result<(Float, Float), E> {
                if c <= a || c >= b {
                    // U0
                    return Ok((a, b));
                }
                let dphi_c = dphi(&dphi_vec(c, st)?);
                if dphi_c >= 0.0 {
                    // U1
                    Ok((a, c))
                } else {
                    let phi_c = phi(c, st)?;
                    if phi_c <= phi_0 + epsilon_k {
                        // U2
                        Ok((c, b))
                    } else {
                        // U3
                        let mut a_hat = a;
                        let mut b_hat = c;
                        let mut i = 0;
                        loop {
                            let d = (1.0 - self.theta).mul_add(a_hat, self.theta * b_hat);
                            let dphi_d = dphi(&dphi_vec(d, st)?);
                            if dphi_d >= 0.0 || i >= self.max_bisects {
                                // U3a
                                return Ok((a_hat, d));
                            } else {
                                let phi_d = phi(d, st)?;
                                if phi_d <= phi_0 + epsilon_k {
                                    // U3b
                                    a_hat = d;
                                } else {
                                    // U3c
                                    b_hat = d;
                                }
                            }
                            i += 1;
                        }
                    }
                }
            };
        let secant_2 = |a: Float,
                        dphi_a: Float,
                        b: Float,
                        dphi_b: Float,
                        st: &mut GradientStatus|
         -> Result<(Float, Float), E> {
            let c = secant(a, dphi_a, b, dphi_b);
            let (a_star, b_star) = update(a, b, c, st)?;
            if c == a_star {
                update(
                    a_star,
                    b_star,
                    secant(a, dphi_a, a_star, dphi(&dphi_vec(a_star, st)?)),
                    st,
                )
            } else if c == b_star {
                update(
                    a_star,
                    b_star,
                    secant(b, dphi_b, b_star, dphi(&dphi_vec(b_star, st)?)),
                    st,
                )
            } else {
                Ok((a_star, b_star))
            }
        };
        // T1: 2.2 & 2.3
        // 2.2:
        // phi(a_k) - phi(0) <= delta a_k phi'(0)
        // 2.3:
        // phi'(a_k) >= sigma phi'(0)
        let check_t1 = |alpha: Float, phi_alpha: Float, dphi_alpha: Float| -> bool {
            let c1 = phi_alpha - phi_0 <= self.delta * alpha * dphi_0;
            let c2 = dphi_alpha >= self.sigma * dphi_0;
            c1 && c2
        };
        // T2: 4.1 & phi(a_k) <= phi(0) + e_k
        // 4.1: (2 delta - 1) phi'(0) >= phi'(a_k) >= sigma phi'(0)
        let check_t2 = |phi_alpha: Float, dphi_alpha: Float| -> bool {
            let c1 = Float::mul_add(2.0, self.delta, -1.0) * dphi_0 >= dphi_alpha;
            let c2 = dphi_alpha >= self.sigma * dphi_0;
            let c3 = phi_alpha <= phi_0 + epsilon_k;
            c1 && c2 && c3
        };
        let check = |alpha: Float, phi_alpha: Float, dphi_alpha: Float| -> bool {
            if check_t1(alpha, phi_alpha, dphi_alpha) {
                return true;
            }
            check_t2(phi_alpha, dphi_alpha)
        };
        // TODO: start with [a, b] such that phi(a) <= phi(0) + e_k, phi'(a) < 0, and phi'(b) >= 0
        // Hager and Zhang talk about some methods for doing this, but I can't find many
        // implementations of this line search where they actually go through the trouble.
        let mut a_k = Float::EPSILON;
        let mut b_k = max_step.unwrap_or(1e5);
        let mut i = 0;
        loop {
            let (phi_a_k, g_a_k) = phi_dphi_vec(a_k, status)?;
            let dphi_a_k = dphi(&g_a_k);
            if check(a_k, phi_a_k, dphi_a_k) {
                return Ok(Ok(LineSearchOutput {
                    alpha: a_k,
                    fx: phi_a_k,
                    g: g_a_k,
                }));
            }
            let (phi_b_k, g_b_k) = phi_dphi_vec(b_k, status)?;
            let dphi_b_k = dphi(&g_b_k);
            if check(b_k, phi_b_k, dphi_b_k) {
                return Ok(Ok(LineSearchOutput {
                    alpha: b_k,
                    fx: phi_b_k,
                    g: g_b_k,
                }));
            }
            let (mut a, mut b) = secant_2(a_k, dphi_a_k, b_k, dphi_b_k, status)?;
            if b - a > self.gamma * (b_k - a_k) {
                let c = (a + b) / 2.0;
                (a, b) = update(a, b, c, status)?;
            }
            (a_k, b_k) = (a, b);
            if i > self.max_iters {
                return Ok(Err(LineSearchOutput {
                    alpha: a_k,
                    fx: phi_a_k,
                    g: g_a_k,
                }));
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_delta_sets_value() {
        let ls = HagerZhangLineSearch::default().with_delta(0.2).unwrap();
        assert_eq!(ls.delta, 0.2);
    }

    #[test]
    fn with_sigma_sets_value() {
        let ls = HagerZhangLineSearch::default().with_sigma(0.7).unwrap();
        assert_eq!(ls.sigma, 0.7);
    }

    #[test]
    fn with_delta_sigma_sets_both() {
        let ls = HagerZhangLineSearch::default()
            .with_delta_sigma(0.05, 0.8)
            .unwrap();
        assert_eq!(ls.delta, 0.05);
        assert_eq!(ls.sigma, 0.8);
    }

    #[test]
    fn with_epsilon_sets_value() {
        let ls = HagerZhangLineSearch::default().with_epsilon(1e-8).unwrap();
        assert_eq!(ls.epsilon, 1e-8);
        assert!(ls.epsilon > 0.0);
    }

    #[test]
    fn with_theta_sets_value() {
        let ls = HagerZhangLineSearch::default().with_theta(0.6).unwrap();
        assert_eq!(ls.theta, 0.6);
        assert!(0.0 < ls.theta && ls.theta < 1.0);
    }

    #[test]
    fn with_gamma_sets_value() {
        let ls = HagerZhangLineSearch::default().with_gamma(0.7).unwrap();
        assert_eq!(ls.gamma, 0.7);
        assert!(0.0 < ls.gamma && ls.gamma < 1.0);
    }

    #[test]
    fn with_max_bisects_sets_value() {
        let ls = HagerZhangLineSearch::default().with_max_bisects(7);
        assert_eq!(ls.max_bisects, 7);
    }

    #[test]
    fn with_delta_errors_when_nonpositive() {
        assert!(HagerZhangLineSearch::default().with_delta(0.0).is_err());
    }

    #[test]
    fn with_delta_errors_when_not_less_than_sigma() {
        let ls = HagerZhangLineSearch::default().with_sigma(0.4).unwrap();
        assert!(ls.with_delta(0.5).is_err());
    }

    #[test]
    fn with_sigma_errors_when_not_less_than_one() {
        assert!(HagerZhangLineSearch::default().with_sigma(1.0).is_err());
    }

    #[test]
    fn with_sigma_errors_when_not_greater_than_delta() {
        let ls = HagerZhangLineSearch::default().with_delta(0.2).unwrap();
        assert!(ls.with_sigma(0.1).is_err());
    }

    #[test]
    fn with_delta_sigma_errors_when_bad_ordering() {
        assert!(HagerZhangLineSearch::default()
            .with_delta_sigma(0.5, 0.2)
            .is_err());
    }

    #[test]
    fn with_delta_sigma_errors_when_sigma_not_less_than_one() {
        assert!(HagerZhangLineSearch::default()
            .with_delta_sigma(0.2, 1.0)
            .is_err());
    }

    #[test]
    fn with_delta_sigma_errors_when_delta_not_positive() {
        assert!(HagerZhangLineSearch::default()
            .with_delta_sigma(0.0, 0.5)
            .is_err());
    }

    #[test]
    fn with_epsilon_errors_when_nonpositive() {
        assert!(HagerZhangLineSearch::default().with_epsilon(0.0).is_err());
    }

    #[test]
    fn with_theta_errors_when_out_of_range() {
        assert!(HagerZhangLineSearch::default().with_theta(1.0).is_err());
    }

    #[test]
    fn with_gamma_errors_when_out_of_range() {
        assert!(HagerZhangLineSearch::default().with_gamma(0.0).is_err());
    }
}
