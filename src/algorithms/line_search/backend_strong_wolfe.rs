//! Scalar- and backend-generic strong-Wolfe line search.

use crate::core::{EvalCounts, LinearAlgebra, NalgebraBackend, RealScalar, Vector};
use crate::traits::{BackendLineSearch, BackendLineSearchOutput, Gradient};
use std::marker::PhantomData;

/// Backend-generic bracketing and zoom line search satisfying the strong-Wolfe conditions.
#[derive(Clone, Debug)]
pub struct BackendStrongWolfeLineSearch<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend>
{
    /// Armijo sufficient-decrease constant.
    pub c1: T,
    /// Curvature constant.
    pub c2: T,
    /// Maximum bracketing/zoom iterations.
    pub max_iterations: usize,
    _backend: PhantomData<B>,
}

impl<T, B> Default for BackendStrongWolfeLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            c1: T::literal(1e-4),
            c2: T::literal(0.9),
            max_iterations: 64,
            _backend: PhantomData,
        }
    }
}

impl<T, B> BackendStrongWolfeLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn evaluate<P, U, E>(
        x: &Vector<T, B>,
        direction: &Vector<T, B>,
        alpha: T,
        problem: &P,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<BackendLineSearchOutput<T, B>, E>
    where
        P: Gradient<T, B, U, E> + ?Sized,
    {
        let trial = x.add_scaled(direction, alpha);
        let (fx, gradient) = problem.evaluate_with_gradient(&trial, args)?;
        evals.record_fg();
        Ok(BackendLineSearchOutput {
            alpha,
            fx,
            gradient,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn zoom<P, U, E>(
        &self,
        x: &Vector<T, B>,
        direction: &Vector<T, B>,
        mut low: BackendLineSearchOutput<T, B>,
        mut high: BackendLineSearchOutput<T, B>,
        phi0: T,
        dphi0: T,
        problem: &P,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<crate::traits::BackendLineSearchResult<T, B>, E>
    where
        P: Gradient<T, B, U, E> + ?Sized,
    {
        let mut best = if low.fx < high.fx {
            low.clone()
        } else {
            high.clone()
        };
        for _ in 0..self.max_iterations {
            let alpha = (low.alpha + high.alpha) / T::literal(2.0);
            let current = Self::evaluate(x, direction, alpha, problem, args, evals)?;
            if current.fx < best.fx {
                best = current.clone();
            }
            if current.fx > phi0 + self.c1 * alpha * dphi0 || current.fx >= low.fx {
                high = current;
            } else {
                let derivative = current.gradient.dot(direction);
                if derivative.abs() <= -self.c2 * dphi0 {
                    return Ok(Ok(current));
                }
                if derivative * (high.alpha - low.alpha) >= T::zero() {
                    high = low;
                }
                low = current;
            }
            if (high.alpha - low.alpha).abs() <= T::epsilon() {
                break;
            }
        }
        Ok(Err(best))
    }
}

impl<T, B, P, U, E> BackendLineSearch<T, B, P, U, E> for BackendStrongWolfeLineSearch<T, B>
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
        let origin = Self::evaluate(x, direction, T::zero(), problem, args, evals)?;
        let phi0 = origin.fx;
        let dphi0 = origin.gradient.dot(direction);
        if dphi0 >= T::zero() {
            return Ok(Err(origin));
        }
        let maximum = max_step.unwrap_or_else(|| T::literal(1e6));
        let mut previous = origin.clone();
        let mut alpha = if T::one() < maximum {
            T::one()
        } else {
            maximum
        };
        let mut best = origin;
        for iteration in 0..self.max_iterations {
            let current = Self::evaluate(x, direction, alpha, problem, args, evals)?;
            if current.fx < best.fx {
                best = current.clone();
            }
            if current.fx > phi0 + self.c1 * alpha * dphi0
                || (iteration > 0 && current.fx >= previous.fx)
            {
                return self.zoom(
                    x, direction, previous, current, phi0, dphi0, problem, args, evals,
                );
            }
            let derivative = current.gradient.dot(direction);
            if derivative.abs() <= -self.c2 * dphi0 {
                return Ok(Ok(current));
            }
            if derivative >= T::zero() {
                return self.zoom(
                    x, direction, current, previous, phi0, dphi0, problem, args, evals,
                );
            }
            previous = current;
            if alpha >= maximum {
                break;
            }
            let doubled = alpha * T::literal(2.0);
            alpha = if doubled < maximum { doubled } else { maximum };
        }
        Ok(Err(best))
    }
}

/// Generic More-Thuente-compatible strong-Wolfe surface.
pub type BackendMoreThuenteLineSearch<T = f64, B = NalgebraBackend> =
    BackendStrongWolfeLineSearch<T, B>;
/// Generic Hager-Zhang-compatible strong-Wolfe surface.
pub type BackendHagerZhangLineSearch<T = f64, B = NalgebraBackend> =
    BackendStrongWolfeLineSearch<T, B>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CostFunction;
    use std::convert::Infallible;

    struct Quadratic;

    impl<T, B> CostFunction<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn evaluate(&self, x: &Vector<T, B>, _: &()) -> Result<T, Infallible> {
            Ok(x.dot(x))
        }
    }

    impl<T, B> Gradient<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn gradient(&self, x: &Vector<T, B>, _: &()) -> Result<Vector<T, B>, Infallible> {
            Ok(x.scale(T::literal(2.0)))
        }
    }

    #[test]
    fn strong_wolfe_supports_f32() {
        let x: Vector<f32> = Vector::from_vec(vec![2.0_f32, -1.0]);
        let direction = x.scale(-2.0);
        let mut evals = EvalCounts::default();
        let result = BackendStrongWolfeLineSearch::default()
            .search(&x, &direction, None, &Quadratic, &(), &mut evals)
            .unwrap()
            .unwrap();
        assert!(result.fx < 1e-8);
        assert!(evals.f() > 0);
    }
}
