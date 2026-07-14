//! Scalar- and linear-algebra-generic More–Thuente-style strong-Wolfe line search.

use crate::core::{EvalCounts, LinearAlgebra, NalgebraProvider, RealScalar, Vector};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{Gradient, LineSearch, LineSearchOutput, LineSearchResult};
use std::marker::PhantomData;

/// Generic implementation of the Moré–Thuente search and zoom procedure.
#[derive(Clone, Debug)]
pub struct MoreThuenteLineSearch<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    max_iterations: usize,
    max_zoom: usize,
    c1: T,
    c2: T,
    _provider: PhantomData<B>,
}

impl<T, B> Default for MoreThuenteLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            max_iterations: 100,
            max_zoom: 100,
            c1: T::literal(1e-4),
            c2: T::literal(0.9),
            _provider: PhantomData,
        }
    }
}

impl<T, B> MoreThuenteLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Set the maximum number of bracketing iterations.
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the maximum number of zoom iterations.
    pub const fn with_max_zoom(mut self, max_zoom: usize) -> Self {
        self.max_zoom = max_zoom;
        self
    }

    /// Set the Armijo constant.
    ///
    /// # Errors
    /// Returns a configuration error unless `0 < c1 < c2`.
    pub fn with_c1(mut self, c1: T) -> GaneshResult<Self> {
        if c1 <= T::zero() || c1 >= self.c2 {
            return Err(GaneshError::ConfigError(
                "MoreThuenteLineSearch requires 0 < c1 < c2".to_string(),
            ));
        }
        self.c1 = c1;
        Ok(self)
    }

    /// Set the curvature constant.
    ///
    /// # Errors
    /// Returns a configuration error unless `c1 < c2 < 1`.
    pub fn with_c2(mut self, c2: T) -> GaneshResult<Self> {
        if c2 <= self.c1 || c2 >= T::one() {
            return Err(GaneshError::ConfigError(
                "MoreThuenteLineSearch requires c1 < c2 < 1".to_string(),
            ));
        }
        self.c2 = c2;
        Ok(self)
    }

    /// Set both strong-Wolfe constants.
    ///
    /// # Errors
    /// Returns a configuration error unless `0 < c1 < c2 < 1`.
    pub fn with_c1_c2(mut self, c1: T, c2: T) -> GaneshResult<Self> {
        if c1 <= T::zero() || c1 >= c2 || c2 >= T::one() {
            return Err(GaneshError::ConfigError(
                "MoreThuenteLineSearch requires 0 < c1 < c2 < 1".to_string(),
            ));
        }
        self.c1 = c1;
        self.c2 = c2;
        Ok(self)
    }

    fn evaluate_f<P, U, E>(
        problem: &P,
        x: &Vector<T, B>,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<T, E>
    where
        P: Gradient<T, B, U, E> + ?Sized,
    {
        evals.record_f();
        problem.evaluate(x, args)
    }

    fn evaluate_g<P, U, E>(
        problem: &P,
        x: &Vector<T, B>,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<Vector<T, B>, E>
    where
        P: Gradient<T, B, U, E> + ?Sized,
    {
        evals.record_g();
        problem.gradient(x, args)
    }

    fn evaluate_fg<P, U, E>(
        problem: &P,
        x: &Vector<T, B>,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<(T, Vector<T, B>), E>
    where
        P: Gradient<T, B, U, E> + ?Sized,
    {
        evals.record_fg();
        problem.evaluate_with_gradient(x, args)
    }

    #[allow(clippy::too_many_arguments)]
    fn zoom<P, U, E>(
        &self,
        problem: &P,
        x0: &Vector<T, B>,
        args: &U,
        f0: T,
        g0: &Vector<T, B>,
        direction: &Vector<T, B>,
        mut alpha_low: T,
        mut alpha_high: T,
        evals: &mut EvalCounts,
    ) -> Result<LineSearchResult<T, B>, E>
    where
        P: Gradient<T, B, U, E> + ?Sized,
    {
        let dphi0 = g0.dot(direction);
        for iteration in 0..=self.max_zoom {
            let alpha = (alpha_low + alpha_high) / T::literal(2.0);
            let x = x0.add_scaled(direction, alpha);
            let fx = Self::evaluate_f(problem, &x, args, evals)?;
            let x_low = x0.add_scaled(direction, alpha_low);
            let f_low = Self::evaluate_f(problem, &x_low, args, evals)?;
            let valid = if fx > dphi0.mul_add(self.c1 * alpha, f0) || fx >= f_low {
                alpha_high = alpha;
                false
            } else {
                let gradient = Self::evaluate_g(problem, &x, args, evals)?;
                let dphi = gradient.dot(direction);
                if dphi.abs() <= -self.c2 * dphi0 {
                    return Ok(Ok(LineSearchOutput {
                        alpha,
                        fx,
                        gradient,
                    }));
                }
                if dphi * (alpha_high - alpha_low) >= T::zero() {
                    alpha_high = alpha_low;
                }
                alpha_low = alpha;
                true
            };
            if iteration == self.max_zoom {
                let gradient = Self::evaluate_g(problem, &x, args, evals)?;
                let output = LineSearchOutput {
                    alpha,
                    fx,
                    gradient,
                };
                return Ok(if valid { Ok(output) } else { Err(output) });
            }
        }
        unreachable!("bounded zoom loop always returns")
    }
}

impl<T, B, P, U, E> LineSearch<T, B, P, U, E> for MoreThuenteLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: Gradient<T, B, U, E> + ?Sized,
{
    fn search(
        &mut self,
        x0: &Vector<T, B>,
        direction: &Vector<T, B>,
        max_step: Option<T>,
        problem: &P,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<LineSearchResult<T, B>, E> {
        let (f0, g0) = Self::evaluate_fg(problem, x0, args, evals)?;
        let alpha_max = max_step.unwrap_or_else(T::one);
        let mut previous_alpha = T::zero();
        let mut alpha = T::one();
        let mut previous_fx = f0;
        let dphi0 = g0.dot(direction);
        for iteration in 0..=self.max_iterations {
            let x = x0.add_scaled(direction, alpha);
            let fx = Self::evaluate_f(problem, &x, args, evals)?;
            if fx > dphi0.mul_add(self.c1, f0) || (iteration > 1 && fx >= previous_fx) {
                return self.zoom(
                    problem,
                    x0,
                    args,
                    f0,
                    &g0,
                    direction,
                    previous_alpha,
                    alpha,
                    evals,
                );
            }
            let gradient = Self::evaluate_g(problem, &x, args, evals)?;
            let dphi = gradient.dot(direction);
            if dphi.abs() <= self.c2 * dphi0.abs() {
                return Ok(Ok(LineSearchOutput {
                    alpha,
                    fx,
                    gradient,
                }));
            }
            if dphi >= T::zero() {
                return self.zoom(
                    problem,
                    x0,
                    args,
                    f0,
                    &g0,
                    direction,
                    alpha,
                    previous_alpha,
                    evals,
                );
            }
            previous_alpha = alpha;
            previous_fx = fx;
            alpha = (alpha_max - alpha).mul_add(T::literal(0.8), alpha);
            if iteration == self.max_iterations {
                return Ok(Err(LineSearchOutput {
                    alpha,
                    fx,
                    gradient,
                }));
            }
        }
        unreachable!("bounded search loop always returns")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_validation_matches_wolfe_ordering() {
        assert!(MoreThuenteLineSearch::<f64>::default()
            .with_c1_c2(1e-3, 0.8)
            .is_ok());
        assert!(MoreThuenteLineSearch::<f64>::default()
            .with_c1_c2(0.8, 0.1)
            .is_err());
    }
}
