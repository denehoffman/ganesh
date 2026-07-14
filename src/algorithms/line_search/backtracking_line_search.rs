use crate::core::{EvalCounts, LinearAlgebra, NalgebraProvider, RealScalar, Vector};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{Gradient, LineSearch, LineSearchOutput};
use std::marker::PhantomData;

/// Scalar- and linear-algebra-generic Armijo backtracking line search.
#[derive(Clone, Debug)]
pub struct BacktrackingLineSearch<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    rho: T,
    c: T,
    _provider: PhantomData<B>,
}

impl<T, B> Default for BacktrackingLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            rho: T::literal(0.5),
            c: T::literal(1e-4),
            _provider: PhantomData,
        }
    }
}

impl<T, B> BacktrackingLineSearch<T, B>
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

impl<T, B, P, U, E> LineSearch<T, B, P, U, E> for BacktrackingLineSearch<T, B>
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
    ) -> Result<Result<LineSearchOutput<T, B>, LineSearchOutput<T, B>>, E> {
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
                return Ok(Ok(LineSearchOutput {
                    alpha,
                    fx: trial_fx,
                    gradient: trial_gradient,
                }));
            }
            alpha = alpha * self.rho;
            if alpha <= T::epsilon() {
                return Ok(Err(LineSearchOutput {
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
        let ls = BacktrackingLineSearch::<f64>::default()
            .with_rho(0.7)
            .unwrap();
        assert_eq!(ls.rho, 0.7);
    }

    #[test]
    fn with_c_sets_value() {
        let ls = BacktrackingLineSearch::<f64>::default()
            .with_c(1e-3)
            .unwrap();
        assert_eq!(ls.c, 1e-3);
    }

    #[test]
    fn with_rho_errors_when_out_of_range_low() {
        assert!(BacktrackingLineSearch::<f64>::default()
            .with_rho(0.0)
            .is_err());
    }

    #[test]
    fn with_rho_errors_when_out_of_range_high() {
        assert!(BacktrackingLineSearch::<f64>::default()
            .with_rho(1.0)
            .is_err());
    }

    #[test]
    fn with_c_errors_when_out_of_range_low() {
        assert!(BacktrackingLineSearch::<f64>::default()
            .with_c(0.0)
            .is_err());
    }

    #[test]
    fn with_c_errors_when_out_of_range_high() {
        assert!(BacktrackingLineSearch::<f64>::default()
            .with_c(1.0)
            .is_err());
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
    fn provider_backtracking_supports_f32() {
        let mut search = BacktrackingLineSearch::<f32>::default();
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
