//! Scalar- and backend-generic limited-memory BFGS with native box projection.

use crate::algorithms::gradient::BackendGradientStatus;
use crate::algorithms::line_search::BackendBacktrackingLineSearch;
use crate::core::{
    BackendMinimizationSummary, Callbacks, LinearAlgebra, Matrix, MaxSteps, NalgebraBackend,
    PseudoInverse, RealScalar, Vector,
};
use crate::error::GaneshResult;
use crate::traits::{
    Algorithm, BackendLineSearch, BackendLineSearchOutput, Gradient, ScalarBound, Status,
};
use std::marker::PhantomData;

/// Configuration for backend-generic L-BFGS-B.
pub struct BackendLBFGSBConfig<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Number of correction pairs retained by the inverse-Hessian approximation.
    pub history_size: usize,
    /// Projected-gradient convergence tolerance.
    pub gradient_tolerance: T,
    /// Optional native parameter bounds.
    pub bounds: Option<Vec<ScalarBound<T>>>,
    /// Backtracking line search.
    pub line_search: BackendBacktrackingLineSearch<T, B>,
}

impl<T, B> Default for BackendLBFGSBConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            history_size: 10,
            gradient_tolerance: T::epsilon().cbrt(),
            bounds: None,
            line_search: BackendBacktrackingLineSearch::default(),
        }
    }
}

impl<T, B> BackendLBFGSBConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Configure native lower/upper bounds.
    ///
    /// # Errors
    /// Returns a configuration error for invalid endpoints.
    pub fn with_bounds<I>(mut self, bounds: I) -> GaneshResult<Self>
    where
        I: IntoIterator<Item = (T, T)>,
    {
        self.bounds = Some(
            bounds
                .into_iter()
                .map(|(lower, upper)| ScalarBound::new(lower, upper))
                .collect::<GaneshResult<Vec<_>>>()?,
        );
        Ok(self)
    }
}

/// Scalar- and backend-generic limited-memory BFGS optimizer with native box projection.
#[derive(Clone, Debug)]
pub struct BackendLBFGSB<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    x: Vector<T, B>,
    fx: T,
    gradient: Vector<T, B>,
    s_history: Vec<Vector<T, B>>,
    y_history: Vec<Vector<T, B>>,
    _backend: PhantomData<B>,
}

impl<T, B> Default for BackendLBFGSB<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            x: Vector::zeros(0),
            fx: T::infinity(),
            gradient: Vector::zeros(0),
            s_history: Vec::new(),
            y_history: Vec::new(),
            _backend: PhantomData,
        }
    }
}

impl<T, B> BackendLBFGSB<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn clip(mut x: Vector<T, B>, bounds: Option<&[ScalarBound<T>]>) -> Vector<T, B> {
        if let Some(bounds) = bounds {
            for (index, bound) in bounds.iter().copied().enumerate() {
                let value = match bound {
                    ScalarBound::Unbounded => x.get(index),
                    ScalarBound::Lower(lower) => {
                        if x.get(index) < lower {
                            lower
                        } else {
                            x.get(index)
                        }
                    }
                    ScalarBound::Upper(upper) => {
                        if x.get(index) > upper {
                            upper
                        } else {
                            x.get(index)
                        }
                    }
                    ScalarBound::Both(lower, upper) => {
                        if x.get(index) < lower {
                            lower
                        } else if x.get(index) > upper {
                            upper
                        } else {
                            x.get(index)
                        }
                    }
                };
                x.set(index, value);
            }
        }
        x
    }

    fn projected_gradient(&self, bounds: Option<&[ScalarBound<T>]>) -> Vector<T, B> {
        let mut projected = self.gradient.clone();
        if let Some(bounds) = bounds {
            for (index, bound) in bounds.iter().copied().enumerate() {
                let at_lower = match bound {
                    ScalarBound::Lower(lower) | ScalarBound::Both(lower, _) => {
                        self.x.get(index) <= lower + T::epsilon().sqrt()
                    }
                    _ => false,
                };
                let at_upper = match bound {
                    ScalarBound::Upper(upper) | ScalarBound::Both(_, upper) => {
                        self.x.get(index) >= upper - T::epsilon().sqrt()
                    }
                    _ => false,
                };
                if (at_lower && projected.get(index) > T::zero())
                    || (at_upper && projected.get(index) < T::zero())
                {
                    projected.set(index, T::zero());
                }
            }
        }
        projected
    }

    fn direction(&self, projected_gradient: &Vector<T, B>) -> Vector<T, B> {
        if self.s_history.is_empty() {
            return projected_gradient.neg();
        }
        let mut q = projected_gradient.clone();
        let mut alphas = vec![T::zero(); self.s_history.len()];
        let mut rhos = vec![T::zero(); self.s_history.len()];
        for index in (0..self.s_history.len()).rev() {
            let sy = self.s_history[index].dot(&self.y_history[index]);
            if sy.abs() <= T::epsilon() {
                continue;
            }
            rhos[index] = T::one() / sy;
            alphas[index] = rhos[index] * self.s_history[index].dot(&q);
            q = q.add_scaled(&self.y_history[index], -alphas[index]);
        }
        let last = self.s_history.len() - 1;
        let yy = self.y_history[last].dot(&self.y_history[last]);
        let gamma = if yy > T::epsilon() {
            self.s_history[last].dot(&self.y_history[last]) / yy
        } else {
            T::one()
        };
        let mut result = q.scale(gamma);
        for index in 0..self.s_history.len() {
            let beta = rhos[index] * self.y_history[index].dot(&result);
            result = result.add_scaled(&self.s_history[index], alphas[index] - beta);
        }
        result.neg()
    }

    fn maximum_step(
        &self,
        direction: &Vector<T, B>,
        bounds: Option<&[ScalarBound<T>]>,
    ) -> Option<T> {
        let bounds = bounds?;
        let mut maximum = T::infinity();
        for (index, bound) in bounds.iter().copied().enumerate() {
            let direction_i = direction.get(index);
            let candidate = match bound {
                ScalarBound::Lower(lower) if direction_i < T::zero() => {
                    Some((lower - self.x.get(index)) / direction_i)
                }
                ScalarBound::Upper(upper) if direction_i > T::zero() => {
                    Some((upper - self.x.get(index)) / direction_i)
                }
                ScalarBound::Both(lower, _) if direction_i < T::zero() => {
                    Some((lower - self.x.get(index)) / direction_i)
                }
                ScalarBound::Both(_, upper) if direction_i > T::zero() => {
                    Some((upper - self.x.get(index)) / direction_i)
                }
                _ => None,
            };
            if let Some(candidate) = candidate {
                if candidate < maximum {
                    maximum = candidate;
                }
            }
        }
        if maximum.is_finite() {
            Some(maximum)
        } else {
            None
        }
    }

    fn project_direction(
        &self,
        mut direction: Vector<T, B>,
        bounds: Option<&[ScalarBound<T>]>,
    ) -> Vector<T, B> {
        if let Some(bounds) = bounds {
            for (index, bound) in bounds.iter().copied().enumerate() {
                let blocked = match bound {
                    ScalarBound::Lower(lower) => {
                        self.x.get(index) <= lower + T::epsilon().sqrt()
                            && direction.get(index) < T::zero()
                    }
                    ScalarBound::Upper(upper) => {
                        self.x.get(index) >= upper - T::epsilon().sqrt()
                            && direction.get(index) > T::zero()
                    }
                    ScalarBound::Both(lower, upper) => {
                        (self.x.get(index) <= lower + T::epsilon().sqrt()
                            && direction.get(index) < T::zero())
                            || (self.x.get(index) >= upper - T::epsilon().sqrt()
                                && direction.get(index) > T::zero())
                    }
                    ScalarBound::Unbounded => false,
                };
                if blocked {
                    direction.set(index, T::zero());
                }
            }
        }
        direction
    }
}

impl<T, B, P, U, E> Algorithm<P, BackendGradientStatus<T, B>, U, E> for BackendLBFGSB<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T> + PseudoInverse<T>,
    P: Gradient<T, B, U, E>,
{
    type Summary = BackendMinimizationSummary<T, B>;
    type Config = BackendLBFGSBConfig<T, B>;
    type Init = Vector<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut BackendGradientStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        if let Some(bounds) = &config.bounds {
            debug_assert_eq!(bounds.len(), init.len());
        }
        self.x = Self::clip(init.clone(), config.bounds.as_deref());
        (self.fx, self.gradient) = problem.evaluate_with_gradient(&self.x, args)?;
        self.s_history.clear();
        self.y_history.clear();
        status.evals.record_fg();
        status.initialize(self.x.clone(), self.fx);
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut BackendGradientStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let projected = self.projected_gradient(config.bounds.as_deref());
        if projected.norm() <= config.gradient_tolerance {
            status
                .set_message()
                .succeed_with_message("PROJECTED GRADIENT CONVERGED");
            return Ok(());
        }
        let direction =
            self.project_direction(self.direction(&projected), config.bounds.as_deref());
        let maximum_step = self.maximum_step(&direction, config.bounds.as_deref());
        let mut line_search = config.line_search.clone();
        if let Ok(BackendLineSearchOutput {
            alpha,
            fx,
            gradient,
        }) = line_search.search(
            &self.x,
            &direction,
            maximum_step,
            problem,
            args,
            &mut status.evals,
        )? {
            let next_x = Self::clip(
                self.x.add_scaled(&direction, alpha),
                config.bounds.as_deref(),
            );
            let s = next_x.sub(&self.x);
            let y = gradient.sub(&self.gradient);
            if s.dot(&y) > T::epsilon() {
                if self.s_history.len() == config.history_size.max(1) {
                    self.s_history.remove(0);
                    self.y_history.remove(0);
                }
                self.s_history.push(s);
                self.y_history.push(y);
            }
            self.x = next_x;
            self.fx = fx;
            self.gradient = gradient;
            status.set_position(self.x.clone(), self.fx);
        } else {
            self.s_history.clear();
            self.y_history.clear();
        }
        Ok(())
    }

    fn postprocessing(
        &mut self,
        problem: &P,
        status: &mut BackendGradientStatus<T, B>,
        args: &U,
        _config: &Self::Config,
    ) -> Result<(), E> {
        let hessian = problem.hessian(&self.x, args)?;
        status.evals.record_h();
        status.set_hess(hessian);
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &BackendGradientStatus<T, B>,
        _args: &U,
        init: &Self::Init,
        _config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        let dimension = status.x.len();
        Ok(BackendMinimizationSummary {
            parameter_names: None,
            message: status.message.clone(),
            x0: init.clone(),
            x: status.x.clone(),
            std: status
                .err
                .clone()
                .unwrap_or_else(|| Vector::zeros(dimension)),
            fx: status.fx,
            evals: status.evals,
            covariance: status
                .cov
                .clone()
                .unwrap_or_else(|| Matrix::identity(dimension)),
        })
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn default_callbacks() -> Callbacks<Self, P, BackendGradientStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty().with_terminator(MaxSteps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CostFunction;
    use std::convert::Infallible;

    struct ShiftedQuadratic;

    impl<T, B> CostFunction<T, B> for ShiftedQuadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn evaluate(&self, x: &Vector<T, B>, _: &()) -> Result<T, Infallible> {
            Ok((x.get(0) - T::literal(3.0)).powi(2) + x.get(1).powi(2))
        }
    }

    impl<T, B> Gradient<T, B> for ShiftedQuadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn gradient(&self, x: &Vector<T, B>, _: &()) -> Result<Vector<T, B>, Infallible> {
            Ok(Vector::from_vec(vec![
                T::literal(2.0) * (x.get(0) - T::literal(3.0)),
                T::literal(2.0) * x.get(1),
            ]))
        }

        fn hessian(&self, _x: &Vector<T, B>, _: &()) -> Result<Matrix<T, B>, Infallible> {
            Ok(Matrix::identity(2).scale(T::literal(2.0)))
        }
    }

    #[test]
    fn lbfgsb_respects_native_f32_bounds() {
        let config = BackendLBFGSBConfig {
            gradient_tolerance: 1e-6,
            ..BackendLBFGSBConfig::default()
        }
        .with_bounds([(0.0_f32, 2.0), (-1.0, 1.0)])
        .unwrap();
        let mut algorithm = BackendLBFGSB::<f32>::default();
        let result = algorithm
            .process(
                &ShiftedQuadratic,
                &(),
                Vector::from_vec(vec![0.5, 0.5]),
                config,
                BackendLBFGSB::<f32>::default_callbacks(),
            )
            .unwrap();
        assert!((result.x.get(0) - 2.0).abs() < 1e-4);
        assert!(
            result.x.get(1).abs() < 5e-4,
            "x={:?}, fx={}",
            result.x,
            result.fx
        );
    }
}
