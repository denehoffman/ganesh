use crate::{
    algorithms::gradient::GradientStatus,
    core::Bounds,
    error::{GaneshError, GaneshResult},
    traits::{linesearch::LineSearchOutput, Gradient, LineSearch},
    DVector, Float,
};

/// A minimal line search algorithm which satisfies the Armijo condition. This is equivalent to
/// Algorithm 3.1 from Nocedal and Wright's book "Numerical Optimization"[^1] (page 37).
///
/// [^1]: [Numerical Optimization. Springer New York, 2006. doi: 10.1007/978-0-387-40065-5.](https://doi.org/10.1007/978-0-387-40065-5)
#[derive(Clone)]
pub struct BacktrackingLineSearch {
    rho: Float,
    c: Float,
}
impl Default for BacktrackingLineSearch {
    fn default() -> Self {
        Self { rho: 0.5, c: 1e-4 }
    }
}
impl BacktrackingLineSearch {
    /// Set the backtracking factor $`\rho`$ (default = `0.5`).
    ///
    /// On each unsuccessful Armijo check, the step is scaled by $`\rho`$.
    ///
    /// Returns a config error if $`0 \ge \rho`$ or $`\rho \ge 1`$.
    pub fn with_rho(mut self, rho: Float) -> GaneshResult<Self> {
        if !(0.0 < rho && rho < 1.0) {
            return Err(GaneshError::ConfigError(
                "BacktrackingLineSearch requires 0 < rho < 1".to_string(),
            ));
        }
        self.rho = rho;
        Ok(self)
    }

    /// Set the Armijo parameter $`c`$ (default = `1e-4`).
    ///
    /// The Armijo condition is $`\phi(\alpha) \le \phi(0) + c\,\alpha\,\phi'(0)`$.
    ///
    /// Returns a config error if $`0 \ge c`$ or $`c \ge 1`$.
    pub fn with_c(mut self, c: Float) -> GaneshResult<Self> {
        if !(0.0 < c && c < 1.0) {
            return Err(GaneshError::ConfigError(
                "BacktrackingLineSearch requires 0 < c < 1".to_string(),
            ));
        }
        self.c = c;
        Ok(self)
    }
}

impl<U, E> LineSearch<GradientStatus, U, E> for BacktrackingLineSearch {
    fn search(
        &mut self,
        x: &DVector<Float>,
        p: &DVector<Float>,
        max_step: Option<Float>,
        problem: &dyn Gradient<U, E>,
        _bounds: Option<&Bounds>,
        args: &U,
        status: &mut GradientStatus,
    ) -> Result<Result<LineSearchOutput, LineSearchOutput>, E> {
        let mut alpha_i = max_step.map_or(1.0, |max_alpha| max_alpha);
        let phi = |alpha: Float, args: &U, st: &mut GradientStatus| -> Result<Float, E> {
            st.inc_n_f_evals();
            problem.evaluate(&(x + p.scale(alpha)), args)
        };
        status.inc_n_f_evals();
        status.inc_n_g_evals();
        let (phi_0, g_0) = problem.evaluate_with_gradient(x, args)?;
        let mut phi_alpha_i = phi(alpha_i, args, status)?;
        let dphi_0 = g_0.dot(p);
        loop {
            let armijo = phi_alpha_i <= (self.c * alpha_i).mul_add(dphi_0, phi_0);
            if armijo {
                status.inc_n_g_evals();
                let g_alpha_i = problem.gradient(&(x + p.scale(alpha_i)), args)?;
                return Ok(Ok(LineSearchOutput {
                    alpha: alpha_i,
                    fx: phi_alpha_i,
                    g: g_alpha_i,
                }));
            }
            alpha_i *= self.rho;
            phi_alpha_i = phi(alpha_i, args, status)?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_rho_sets_value() {
        let ls = BacktrackingLineSearch::default().with_rho(0.7).unwrap();
        assert_eq!(ls.rho, 0.7);
    }

    #[test]
    fn with_c_sets_value() {
        let ls = BacktrackingLineSearch::default().with_c(1e-3).unwrap();
        assert_eq!(ls.c, 1e-3);
    }

    #[test]
    fn with_rho_errors_when_out_of_range_low() {
        assert!(BacktrackingLineSearch::default().with_rho(0.0).is_err());
    }

    #[test]
    fn with_rho_errors_when_out_of_range_high() {
        assert!(BacktrackingLineSearch::default().with_rho(1.0).is_err());
    }

    #[test]
    fn with_c_errors_when_out_of_range_low() {
        assert!(BacktrackingLineSearch::default().with_c(0.0).is_err());
    }

    #[test]
    fn with_c_errors_when_out_of_range_high() {
        assert!(BacktrackingLineSearch::default().with_c(1.0).is_err());
    }
}
