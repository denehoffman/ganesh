//! Scalar- and linear-algebra-generic Hager–Zhang line search.

use crate::core::{EvalCounts, LinearAlgebra, NalgebraProvider, RealScalar, Vector};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{Gradient, LineSearch, LineSearchOutput, LineSearchResult};
use std::marker::PhantomData;

/// Generic Hager–Zhang approximate-Wolfe line search.
#[derive(Clone, Debug)]
pub struct HagerZhangLineSearch<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    delta: T,
    sigma: T,
    epsilon: T,
    theta: T,
    gamma: T,
    max_iterations: usize,
    max_bisects: usize,
    _provider: PhantomData<B>,
}

impl<T, B> Default for HagerZhangLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            delta: T::literal(0.1),
            sigma: T::literal(0.9),
            epsilon: T::epsilon().cbrt(),
            theta: T::literal(0.5),
            gamma: T::literal(0.66),
            max_iterations: 100,
            max_bisects: 50,
            _provider: PhantomData,
        }
    }
}

impl<T, B> HagerZhangLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Set the maximum number of secant iterations.
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the Armijo parameter.
    ///
    /// # Errors
    /// Returns a configuration error unless `0 < delta < sigma`.
    pub fn with_delta(mut self, delta: T) -> GaneshResult<Self> {
        if delta <= T::zero() || delta >= self.sigma {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch requires 0 < delta < sigma".to_string(),
            ));
        }
        self.delta = delta;
        Ok(self)
    }

    /// Set the curvature parameter.
    ///
    /// # Errors
    /// Returns a configuration error unless `delta < sigma < 1`.
    pub fn with_sigma(mut self, sigma: T) -> GaneshResult<Self> {
        if sigma <= self.delta || sigma >= T::one() {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch requires delta < sigma < 1".to_string(),
            ));
        }
        self.sigma = sigma;
        Ok(self)
    }

    /// Set both Wolfe parameters.
    ///
    /// # Errors
    /// Returns a configuration error unless `0 < delta < sigma < 1`.
    pub fn with_delta_sigma(mut self, delta: T, sigma: T) -> GaneshResult<Self> {
        if delta <= T::zero() || delta >= sigma || sigma >= T::one() {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch requires 0 < delta < sigma < 1".to_string(),
            ));
        }
        self.delta = delta;
        self.sigma = sigma;
        Ok(self)
    }

    /// Set the approximate-Wolfe tolerance.
    ///
    /// # Errors
    /// Returns a configuration error unless `epsilon > 0`.
    pub fn with_epsilon(mut self, epsilon: T) -> GaneshResult<Self> {
        if epsilon <= T::zero() {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch epsilon must be > 0".to_string(),
            ));
        }
        self.epsilon = epsilon;
        Ok(self)
    }

    /// Set the interval interpolation parameter.
    ///
    /// # Errors
    /// Returns a configuration error unless `theta` is in `(0, 1)`.
    pub fn with_theta(mut self, theta: T) -> GaneshResult<Self> {
        if theta <= T::zero() || theta >= T::one() {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch theta must be in (0, 1)".to_string(),
            ));
        }
        self.theta = theta;
        Ok(self)
    }

    /// Set the interval-contraction threshold.
    ///
    /// # Errors
    /// Returns a configuration error unless `gamma` is in `(0, 1)`.
    pub fn with_gamma(mut self, gamma: T) -> GaneshResult<Self> {
        if gamma <= T::zero() || gamma >= T::one() {
            return Err(GaneshError::ConfigError(
                "HagerZhangLineSearch gamma must be in (0, 1)".to_string(),
            ));
        }
        self.gamma = gamma;
        Ok(self)
    }

    /// Set the maximum number of interval bisections.
    pub const fn with_max_bisects(mut self, max_bisects: usize) -> Self {
        self.max_bisects = max_bisects;
        self
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_f<P, U, E>(
        problem: &P,
        x0: &Vector<T, B>,
        direction: &Vector<T, B>,
        alpha: T,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<T, E>
    where
        P: Gradient<T, B, U, E> + ?Sized,
    {
        let x = x0.add_scaled(direction, alpha);
        evals.record_f();
        problem.evaluate(&x, args)
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_g<P, U, E>(
        problem: &P,
        x0: &Vector<T, B>,
        direction: &Vector<T, B>,
        alpha: T,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<Vector<T, B>, E>
    where
        P: Gradient<T, B, U, E> + ?Sized,
    {
        let x = x0.add_scaled(direction, alpha);
        evals.record_g();
        problem.gradient(&x, args)
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_fg<P, U, E>(
        problem: &P,
        x0: &Vector<T, B>,
        direction: &Vector<T, B>,
        alpha: T,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<(T, Vector<T, B>), E>
    where
        P: Gradient<T, B, U, E> + ?Sized,
    {
        let x = x0.add_scaled(direction, alpha);
        evals.record_fg();
        problem.evaluate_with_gradient(&x, args)
    }

    fn secant(a: T, dphi_a: T, b: T, dphi_b: T) -> T {
        a.mul_add(dphi_b, -(b * dphi_a)) / (dphi_b - dphi_a)
    }

    #[allow(clippy::too_many_arguments)]
    fn update<P, U, E>(
        &self,
        problem: &P,
        x0: &Vector<T, B>,
        direction: &Vector<T, B>,
        args: &U,
        phi0: T,
        epsilon_k: T,
        a: T,
        b: T,
        c: T,
        evals: &mut EvalCounts,
    ) -> Result<(T, T), E>
    where
        P: Gradient<T, B, U, E> + ?Sized,
    {
        if c <= a || c >= b {
            return Ok((a, b));
        }
        let gradient_c = Self::evaluate_g(problem, x0, direction, c, args, evals)?;
        if gradient_c.dot(direction) >= T::zero() {
            return Ok((a, c));
        }
        let phi_c = Self::evaluate_f(problem, x0, direction, c, args, evals)?;
        if phi_c <= phi0 + epsilon_k {
            return Ok((c, b));
        }
        let mut a_hat = a;
        let mut b_hat = c;
        for iteration in 0..=self.max_bisects {
            let d = a_hat.mul_add(T::one() - self.theta, self.theta * b_hat);
            let gradient_d = Self::evaluate_g(problem, x0, direction, d, args, evals)?;
            if gradient_d.dot(direction) >= T::zero() || iteration == self.max_bisects {
                return Ok((a_hat, d));
            }
            let phi_d = Self::evaluate_f(problem, x0, direction, d, args, evals)?;
            if phi_d <= phi0 + epsilon_k {
                a_hat = d;
            } else {
                b_hat = d;
            }
        }
        unreachable!("bounded bisection loop always returns")
    }

    #[allow(clippy::too_many_arguments)]
    fn secant2<P, U, E>(
        &self,
        problem: &P,
        x0: &Vector<T, B>,
        direction: &Vector<T, B>,
        args: &U,
        phi0: T,
        epsilon_k: T,
        a: T,
        dphi_a: T,
        b: T,
        dphi_b: T,
        evals: &mut EvalCounts,
    ) -> Result<(T, T), E>
    where
        P: Gradient<T, B, U, E> + ?Sized,
    {
        let c = Self::secant(a, dphi_a, b, dphi_b);
        let (a_star, b_star) = self.update(
            problem, x0, direction, args, phi0, epsilon_k, a, b, c, evals,
        )?;
        if c == a_star {
            let gradient = Self::evaluate_g(problem, x0, direction, a_star, args, evals)?;
            self.update(
                problem,
                x0,
                direction,
                args,
                phi0,
                epsilon_k,
                a_star,
                b_star,
                Self::secant(a, dphi_a, a_star, gradient.dot(direction)),
                evals,
            )
        } else if c == b_star {
            let gradient = Self::evaluate_g(problem, x0, direction, b_star, args, evals)?;
            self.update(
                problem,
                x0,
                direction,
                args,
                phi0,
                epsilon_k,
                a_star,
                b_star,
                Self::secant(b, dphi_b, b_star, gradient.dot(direction)),
                evals,
            )
        } else {
            Ok((a_star, b_star))
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn satisfies_wolfe(&self, alpha: T, phi: T, dphi: T, phi0: T, dphi0: T, epsilon_k: T) -> bool {
        let standard = phi - phi0 <= self.delta * alpha * dphi0 && dphi >= self.sigma * dphi0;
        standard
            || ((T::literal(2.0) * self.delta - T::one()) * dphi0 >= dphi
                && dphi >= self.sigma * dphi0
                && phi <= phi0 + epsilon_k)
    }
}

impl<T, B, P, U, E> LineSearch<T, B, P, U, E> for HagerZhangLineSearch<T, B>
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
        let (phi0, gradient0) = Self::evaluate_fg(problem, x0, direction, T::zero(), args, evals)?;
        let dphi0 = gradient0.dot(direction);
        let epsilon_k = self.epsilon * phi0.abs();
        let mut a = T::epsilon();
        let mut b = max_step.unwrap_or_else(|| T::literal(1e5));
        let mut iteration = 0;
        loop {
            let (phi_a, gradient_a) = Self::evaluate_fg(problem, x0, direction, a, args, evals)?;
            let dphi_a = gradient_a.dot(direction);
            if self.satisfies_wolfe(a, phi_a, dphi_a, phi0, dphi0, epsilon_k) {
                return Ok(Ok(LineSearchOutput {
                    alpha: a,
                    fx: phi_a,
                    gradient: gradient_a,
                }));
            }
            let (phi_b, gradient_b) = Self::evaluate_fg(problem, x0, direction, b, args, evals)?;
            let dphi_b = gradient_b.dot(direction);
            if self.satisfies_wolfe(b, phi_b, dphi_b, phi0, dphi0, epsilon_k) {
                return Ok(Ok(LineSearchOutput {
                    alpha: b,
                    fx: phi_b,
                    gradient: gradient_b,
                }));
            }
            let (mut next_a, mut next_b) = self.secant2(
                problem, x0, direction, args, phi0, epsilon_k, a, dphi_a, b, dphi_b, evals,
            )?;
            if next_b - next_a > self.gamma * (b - a) {
                let midpoint = (next_a + next_b) / T::literal(2.0);
                (next_a, next_b) = self.update(
                    problem, x0, direction, args, phi0, epsilon_k, next_a, next_b, midpoint, evals,
                )?;
            }
            (a, b) = (next_a, next_b);
            if iteration > self.max_iterations {
                return Ok(Err(LineSearchOutput {
                    alpha: a,
                    fx: phi_a,
                    gradient: gradient_a,
                }));
            }
            iteration += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_validation_matches_wolfe_ordering() {
        assert!(HagerZhangLineSearch::<f64>::default()
            .with_delta_sigma(0.1, 0.8)
            .is_ok());
        assert!(HagerZhangLineSearch::<f64>::default()
            .with_delta_sigma(0.8, 0.1)
            .is_err());
    }
}
