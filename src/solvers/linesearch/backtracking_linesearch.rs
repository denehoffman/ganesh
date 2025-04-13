use nalgebra::DVector;

use crate::{
    core::GradientStatus,
    traits::{CostFunction, LineSearch},
    Float,
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
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        status: &mut GradientStatus,
    ) -> Result<(bool, Float, Float, DVector<Float>), E> {
        let mut alpha_i = max_step.map_or(1.0, |max_alpha| max_alpha);
        let phi = |alpha: Float, ud: &mut U, st: &mut GradientStatus| -> Result<Float, E> {
            st.inc_n_f_evals();
            func.evaluate((x + p.scale(alpha)).as_slice(), ud)
        };
        let dphi = |alpha: Float, ud: &mut U, st: &mut GradientStatus| -> Result<Float, E> {
            st.inc_n_g_evals();
            Ok(func.gradient((x + p.scale(alpha)).as_slice(), ud)?.dot(p))
        };
        let phi_0 = phi(0.0, user_data, status)?;
        let mut phi_alpha_i = phi(alpha_i, user_data, status)?;
        let dphi_0 = dphi(0.0, user_data, status)?;
        loop {
            let armijo = phi_alpha_i <= (self.c * alpha_i).mul_add(dphi_0, phi_0);
            if armijo {
                let g_alpha_i = func.gradient((x + p.scale(alpha_i)).as_slice(), user_data)?;
                return Ok((true, alpha_i, phi_alpha_i, g_alpha_i));
            }
            alpha_i *= self.rho;
            phi_alpha_i = phi(alpha_i, user_data, status)?;
        }
    }
}
