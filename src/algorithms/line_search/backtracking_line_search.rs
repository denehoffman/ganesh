use crate::core::{EvalCounts, LinearAlgebra, NalgebraBackend, RealScalar, Vector};
use crate::traits::{BackendLineSearch, BackendLineSearchOutput, Gradient};
use crate::{
    algorithms::gradient::LegacyGradientStatus,
    core::Bounds,
    error::{GaneshError, GaneshResult},
    traits::{LegacyGradient, LegacyLineSearch, LegacyLineSearchOutput},
    DVector, Float,
};
use std::marker::PhantomData;

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
    /// # Errors
    ///
    /// Returns a configuration error if `rho` is not in the interval `(0, 1)`.
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
    /// # Errors
    ///
    /// Returns a configuration error if `c` is not in the interval `(0, 1)`.
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

impl<U, E> LegacyLineSearch<LegacyGradientStatus, U, E> for BacktrackingLineSearch {
    fn search(
        &mut self,
        x: &DVector<Float>,
        p: &DVector<Float>,
        max_step: Option<Float>,
        problem: &dyn LegacyGradient<U, E>,
        _bounds: Option<&Bounds>,
        args: &U,
        status: &mut LegacyGradientStatus,
    ) -> Result<Result<LegacyLineSearchOutput, LegacyLineSearchOutput>, E> {
        let mut alpha_i = max_step.map_or(1.0, |max_alpha| max_alpha);
        let phi = |alpha: Float, args: &U, st: &mut LegacyGradientStatus| -> Result<Float, E> {
            st.evals.record_f();
            problem.evaluate(&(x + p.scale(alpha)), args)
        };
        status.evals.record_fg();
        let (phi_0, g_0) = problem.evaluate_with_gradient(x, args)?;
        let mut phi_alpha_i = phi(alpha_i, args, status)?;
        let dphi_0 = g_0.dot(p);
        loop {
            let armijo = phi_alpha_i <= (self.c * alpha_i).mul_add(dphi_0, phi_0);
            if armijo {
                status.evals.record_g();
                let g_alpha_i = problem.gradient(&(x + p.scale(alpha_i)), args)?;
                return Ok(Ok(LegacyLineSearchOutput {
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

/// Scalar- and backend-generic Armijo backtracking line search.
#[derive(Clone, Debug)]
pub struct BackendBacktrackingLineSearch<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend>
{
    rho: T,
    c: T,
    _backend: PhantomData<B>,
}

impl<T, B> Default for BackendBacktrackingLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            rho: T::literal(0.5),
            c: T::literal(1e-4),
            _backend: PhantomData,
        }
    }
}

impl<T, B> BackendBacktrackingLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Configure the backtracking contraction factor.
    ///
    /// # Errors
    /// Returns a configuration error unless `rho` is in `(0, 1)`.
    pub fn with_rho(mut self, rho: T) -> GaneshResult<Self> {
        if rho <= T::zero() || rho >= T::one() {
            return Err(GaneshError::ConfigError(
                "backtracking rho must be in (0, 1)".to_string(),
            ));
        }
        self.rho = rho;
        Ok(self)
    }

    /// Configure the Armijo sufficient-decrease factor.
    ///
    /// # Errors
    /// Returns a configuration error unless `c` is in `(0, 1)`.
    pub fn with_c(mut self, c: T) -> GaneshResult<Self> {
        if c <= T::zero() || c >= T::one() {
            return Err(GaneshError::ConfigError(
                "backtracking c must be in (0, 1)".to_string(),
            ));
        }
        self.c = c;
        Ok(self)
    }
}

impl<T, B, P, U, E> BackendLineSearch<T, B, P, U, E> for BackendBacktrackingLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: Gradient<T, B, U, E> + ?Sized,
{
    fn search(
        &mut self,
        x: &Vector<T, B>,
        direction: &Vector<T, B>,
        max_step: Option<T>,
        problem: &P,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<Result<BackendLineSearchOutput<T, B>, BackendLineSearchOutput<T, B>>, E> {
        let mut alpha = max_step.unwrap_or_else(T::one);
        let (fx, gradient) = problem.evaluate_with_gradient(x, args)?;
        evals.record_fg();
        let directional_derivative = gradient.dot(direction);
        loop {
            let trial = x.add_scaled(direction, alpha);
            let trial_fx = problem.evaluate(&trial, args)?;
            evals.record_f();
            if trial_fx <= fx + self.c * alpha * directional_derivative {
                let trial_gradient = problem.gradient(&trial, args)?;
                evals.record_g();
                return Ok(Ok(BackendLineSearchOutput {
                    alpha,
                    fx: trial_fx,
                    gradient: trial_gradient,
                }));
            }
            alpha = alpha * self.rho;
            if alpha <= T::epsilon() {
                return Ok(Err(BackendLineSearchOutput {
                    alpha: T::zero(),
                    fx,
                    gradient,
                }));
            }
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

    struct Quadratic;

    impl<T, B> crate::traits::CostFunction<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn evaluate(&self, x: &Vector<T, B>, _: &()) -> Result<T, std::convert::Infallible> {
            Ok(x.dot(x))
        }
    }

    impl<T, B> Gradient<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn gradient(
            &self,
            x: &Vector<T, B>,
            _: &(),
        ) -> Result<Vector<T, B>, std::convert::Infallible> {
            Ok(x.scale(T::literal(2.0)))
        }
    }

    #[test]
    fn backend_backtracking_supports_f32() {
        let mut search = BackendBacktrackingLineSearch::<f32>::default();
        let x = Vector::from_vec(vec![2.0, -1.0]);
        let direction = x.scale(-2.0);
        let mut evals = EvalCounts::default();
        let result = search
            .search(&x, &direction, None, &Quadratic, &(), &mut evals)
            .unwrap()
            .unwrap();
        assert!(result.fx < 5.0);
        assert!(result.alpha > 0.0);
        assert!(evals.f() > 0);
    }
}
