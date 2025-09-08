use crate::{
    algorithms::gradient::GradientStatus,
    core::Bounds,
    traits::{linesearch::LineSearchOutput, Boundable, Gradient, LineSearch},
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

impl<U, E> LineSearch<GradientStatus, U, E> for BacktrackingLineSearch {
    fn search(
        &mut self,
        x: &DVector<Float>,
        p: &DVector<Float>,
        max_step: Option<Float>,
        problem: &dyn Gradient<U, E>,
        bounds: Option<&Bounds>,
        args: &U,
        status: &mut GradientStatus,
    ) -> Result<Result<LineSearchOutput, LineSearchOutput>, E> {
        let mut alpha_i = max_step.map_or(1.0, |max_alpha| max_alpha);
        let phi = |alpha: Float, args: &U, st: &mut GradientStatus| -> Result<Float, E> {
            st.inc_n_f_evals();
            problem.evaluate(&(x + p.scale(alpha)).constrain_to(bounds), args)
        };
        let dphi = |alpha: Float, args: &U, st: &mut GradientStatus| -> Result<Float, E> {
            st.inc_n_g_evals();
            Ok(problem
                .gradient(&(x + p.scale(alpha)).constrain_to(bounds), args)?
                .dot(p))
        };
        let phi_0 = phi(0.0, args, status)?;
        let mut phi_alpha_i = phi(alpha_i, args, status)?;
        let dphi_0 = dphi(0.0, args, status)?;
        loop {
            let armijo = phi_alpha_i <= (self.c * alpha_i).mul_add(dphi_0, phi_0);
            if armijo {
                let g_alpha_i =
                    problem.gradient(&(x + p.scale(alpha_i)).constrain_to(bounds), args)?;
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
