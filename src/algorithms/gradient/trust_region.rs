use crate::{
    algorithms::gradient::GradientStatus,
    core::{Callbacks, MaxSteps, MinimizationSummary},
    error::{GaneshError, GaneshResult},
    traits::{
        Algorithm, CostFunction, Gradient, Status, SupportsParameterNames, SupportsTransform,
        Terminator, Transform, TransformedProblem,
    },
    DMatrix, DVector, Float,
};
use nalgebra::LU;
use std::ops::ControlFlow;

/// A [`Terminator`] which stops [`TrustRegion`] once the gradient norm is sufficiently small.
#[derive(Clone)]
pub struct TrustRegionGTerminator {
    /// The absolute gradient-norm tolerance.
    pub eps_abs: Float,
}

impl Default for TrustRegionGTerminator {
    fn default() -> Self {
        Self {
            eps_abs: Float::cbrt(Float::EPSILON),
        }
    }
}

impl TrustRegionGTerminator {
    /// Generate a new [`TrustRegionGTerminator`] with a given absolute tolerance.
    pub fn new(eps_abs: Float) -> GaneshResult<Self> {
        if eps_abs <= 0.0 {
            return Err(GaneshError::ConfigError(
                "eps_abs must be greater than 0".to_string(),
            ));
        }
        Ok(Self { eps_abs })
    }
}

impl<P, U, E> Terminator<TrustRegion, P, GradientStatus, U, E, TrustRegionConfig>
    for TrustRegionGTerminator
where
    P: Gradient<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut TrustRegion,
        _problem: &P,
        status: &mut GradientStatus,
        _args: &U,
        _config: &TrustRegionConfig,
    ) -> ControlFlow<()> {
        if algorithm.g.norm() < self.eps_abs {
            status
                .set_message()
                .succeed_with_message("GRADIENT CONVERGED");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// Trust-region subproblem solvers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TrustRegionSubproblem {
    /// The Cauchy point, using the best steepest-descent step inside the trust region.
    ///
    /// [^1]: [J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed. Springer, 2006, ch. 4.](https://doi.org/10.1007/978-0-387-40065-5)
    CauchyPoint,
    /// The Powell dogleg method for the trust-region subproblem.
    ///
    /// [^1]: [J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed. Springer, 2006, ch. 4.](https://doi.org/10.1007/978-0-387-40065-5)
    #[default]
    Dogleg,
}

/// Configuration for the [`TrustRegion`] algorithm.
#[derive(Clone)]
pub struct TrustRegionConfig {
    x0: DVector<Float>,
    parameter_names: Option<Vec<String>>,
    transform: Option<Box<dyn Transform>>,
    subproblem: TrustRegionSubproblem,
    initial_radius: Float,
    max_radius: Float,
    eta: Float,
}

impl TrustRegionConfig {
    /// Create a new configuration by setting the starting position of the algorithm.
    pub fn new<I>(x0: I) -> Self
    where
        I: AsRef<[Float]>,
    {
        Self {
            x0: DVector::from_row_slice(x0.as_ref()),
            parameter_names: None,
            transform: None,
            subproblem: TrustRegionSubproblem::default(),
            initial_radius: 1.0,
            max_radius: 1_000.0,
            eta: 1e-4,
        }
    }

    /// Set the trust-region subproblem solver.
    pub const fn with_subproblem(mut self, subproblem: TrustRegionSubproblem) -> Self {
        self.subproblem = subproblem;
        self
    }

    /// Set the initial trust-region radius.
    pub fn with_initial_radius(mut self, initial_radius: Float) -> GaneshResult<Self> {
        if !initial_radius.is_finite() || initial_radius <= 0.0 {
            return Err(GaneshError::ConfigError(
                "Initial radius must be finite and greater than 0".to_string(),
            ));
        }
        self.initial_radius = initial_radius;
        Ok(self)
    }

    /// Set the maximum trust-region radius.
    pub fn with_max_radius(mut self, max_radius: Float) -> GaneshResult<Self> {
        if !max_radius.is_finite() || max_radius <= 0.0 {
            return Err(GaneshError::ConfigError(
                "Maximum radius must be finite and greater than 0".to_string(),
            ));
        }
        self.max_radius = max_radius;
        Ok(self)
    }

    /// Set the step-acceptance threshold on the ratio of actual to predicted reduction.
    pub fn with_eta(mut self, eta: Float) -> GaneshResult<Self> {
        if !eta.is_finite() || !(0.0..1.0).contains(&eta) {
            return Err(GaneshError::ConfigError(
                "eta must be finite and in the range [0, 1)".to_string(),
            ));
        }
        self.eta = eta;
        Ok(self)
    }
}

impl SupportsTransform for TrustRegionConfig {
    fn get_transform_mut(&mut self) -> &mut Option<Box<dyn Transform>> {
        &mut self.transform
    }
}

impl SupportsParameterNames for TrustRegionConfig {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

/// Trust-region optimizer for smooth unconstrained minimization.
///
/// This implementation uses the classical trust-region framework with either the Cauchy point or
/// Powell dogleg method for the subproblem, following the presentation in [^1].
///
/// [^1]: [J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed. Springer, 2006, ch. 4.](https://doi.org/10.1007/978-0-387-40065-5)
#[derive(Clone)]
pub struct TrustRegion {
    x: DVector<Float>,
    f: Float,
    g: DVector<Float>,
    h: DMatrix<Float>,
    radius: Float,
    max_radius: Float,
}

impl Default for TrustRegion {
    fn default() -> Self {
        Self {
            x: DVector::zeros(0),
            f: Float::INFINITY,
            g: DVector::zeros(0),
            h: DMatrix::zeros(0, 0),
            radius: 1.0,
            max_radius: 1_000.0,
        }
    }
}

impl TrustRegion {
    fn predicted_reduction(&self, p: &DVector<Float>) -> Float {
        -(self.g.dot(p) + 0.5 * p.dot(&(&self.h * p)))
    }

    fn cauchy_point(&self) -> DVector<Float> {
        let g_norm = self.g.norm();
        if g_norm <= Float::EPSILON {
            return DVector::zeros(self.g.len());
        }
        let g_bg = self.g.dot(&(&self.h * &self.g));
        let tau = if g_bg <= 0.0 {
            1.0
        } else {
            (g_norm.powi(3) / (self.radius * g_bg)).min(1.0)
        };
        -self.g.scale(tau * self.radius / g_norm)
    }

    fn dogleg(&self) -> DVector<Float> {
        let p_u = {
            let g_bg = self.g.dot(&(&self.h * &self.g));
            if g_bg <= Float::EPSILON {
                self.cauchy_point()
            } else {
                -self.g.scale(self.g.dot(&self.g) / g_bg)
            }
        };
        if p_u.norm() >= self.radius {
            return p_u.scale(self.radius / p_u.norm());
        }

        let p_b = LU::new(self.h.clone())
            .solve(&(-&self.g))
            .filter(|p| p.iter().all(|v| v.is_finite()));
        let Some(p_b) = p_b else {
            return self.cauchy_point();
        };
        if p_b.norm() <= self.radius {
            return p_b;
        }

        let diff = &p_b - &p_u;
        let a = diff.dot(&diff);
        let b = 2.0 * p_u.dot(&diff);
        let c = p_u.dot(&p_u) - self.radius.powi(2);
        let disc = b.mul_add(b, -4.0 * a * c).max(0.0);
        let tau = (-b + disc.sqrt()) / (2.0 * a);
        &p_u + diff.scale(tau)
    }

    fn trust_region_step(&self, subproblem: TrustRegionSubproblem) -> DVector<Float> {
        match subproblem {
            TrustRegionSubproblem::CauchyPoint => self.cauchy_point(),
            TrustRegionSubproblem::Dogleg => self.dogleg(),
        }
    }
}

impl<P, U, E> Algorithm<P, GradientStatus, U, E> for TrustRegion
where
    P: Gradient<U, E>,
{
    type Summary = MinimizationSummary;
    type Config = TrustRegionConfig;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut GradientStatus,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let t_problem = TransformedProblem::new(problem, &config.transform);
        self.x = t_problem.to_owned_internal(&config.x0);
        let (f, g, h) = t_problem.evaluate_with_gradient_and_hessian(&self.x, args)?;
        self.f = f;
        self.g = g;
        self.h = h;
        self.radius = config.initial_radius;
        self.max_radius = config.max_radius.max(config.initial_radius);
        status.n_f_evals += 1;
        status.n_g_evals += 1;
        status.n_h_evals += 1;
        status.initialize((config.x0.clone(), self.f));
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut GradientStatus,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let t_problem = TransformedProblem::new(problem, &config.transform);
        let p = self.trust_region_step(config.subproblem);
        let predicted = self.predicted_reduction(&p);
        if predicted <= Float::EPSILON {
            self.radius *= 0.25;
            return Ok(());
        }

        let x_trial = &self.x + &p;
        let f_trial = t_problem.evaluate(&x_trial, args)?;
        status.inc_n_f_evals();
        let actual = self.f - f_trial;
        let rho = actual / predicted;
        let hits_boundary = (p.norm() - self.radius).abs() <= 1e-12 * (1.0 + self.radius);

        if rho < 0.25 {
            self.radius = 0.25 * p.norm();
        } else if rho > 0.75 && hits_boundary {
            self.radius = (2.0 * self.radius).min(self.max_radius);
        }

        if rho > config.eta {
            let (g_trial, h_trial) = t_problem.gradient_with_hessian(&x_trial, args)?;
            status.inc_n_g_evals();
            status.inc_n_h_evals();
            self.x = x_trial;
            self.f = f_trial;
            self.g = g_trial;
            self.h = h_trial;
            status.set_position((t_problem.to_owned_external(&self.x), self.f));
        }
        Ok(())
    }

    fn postprocessing(
        &mut self,
        problem: &P,
        status: &mut GradientStatus,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let t_problem = TransformedProblem::new(problem, &config.transform);
        let (g_int, h_int) = t_problem.gradient_with_hessian(&self.x, args)?;
        status.inc_n_g_evals();
        status.inc_n_h_evals();
        let hessian = t_problem.pushforward_hessian(&self.x, &g_int, &h_int);
        status.set_hess(&hessian);
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &GradientStatus,
        _args: &U,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        Ok(MinimizationSummary {
            x0: config.x0.clone(),
            x: status.x.clone(),
            fx: status.fx,
            bounds: None,
            cost_evals: status.n_f_evals,
            gradient_evals: status.n_g_evals,
            message: status.message.clone(),
            parameter_names: config.parameter_names.clone(),
            std: status
                .err
                .clone()
                .unwrap_or_else(|| DVector::from_element(status.x.len(), 0.0)),
            covariance: status
                .cov
                .clone()
                .unwrap_or_else(|| DMatrix::identity(status.x.len(), status.x.len())),
        })
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientStatus, U, E, Self::Config>
    where
        Self: Sized,
    {
        Callbacks::empty()
            .with_terminator(TrustRegionGTerminator::default())
            .with_terminator(MaxSteps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{Callbacks, MaxSteps, ScaleTransform},
        test_functions::Rosenbrock,
    };
    use approx::assert_relative_eq;
    use std::convert::Infallible;

    struct IllConditionedQuadratic;
    impl crate::traits::CostFunction<(), Infallible> for IllConditionedQuadratic {
        fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(0.5 * (x[0] * x[0] + 100.0 * x[1] * x[1]))
        }
    }
    impl crate::traits::Gradient<(), Infallible> for IllConditionedQuadratic {
        fn gradient(&self, x: &DVector<Float>, _args: &()) -> Result<DVector<Float>, Infallible> {
            Ok(DVector::from_vec(vec![x[0], 100.0 * x[1]]))
        }
        fn hessian(&self, _x: &DVector<Float>, _args: &()) -> Result<DMatrix<Float>, Infallible> {
            Ok(DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 100.0])))
        }
    }

    #[test]
    fn test_trust_region_dogleg_rosenbrock() {
        let problem = Rosenbrock { n: 2 };
        let mut solver = TrustRegion::default();
        let result = solver
            .process(
                &problem,
                &(),
                TrustRegionConfig::new([-1.2, 1.0]),
                Callbacks::empty()
                    .with_terminator(TrustRegionGTerminator::default())
                    .with_terminator(MaxSteps(200)),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 1e-10);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_trust_region_cauchy_point_ill_conditioned_quadratic() {
        let problem = IllConditionedQuadratic;
        let mut solver = TrustRegion::default();
        let result = solver
            .process(
                &problem,
                &(),
                TrustRegionConfig::new([10.0, -2.0])
                    .with_subproblem(TrustRegionSubproblem::CauchyPoint)
                    .with_initial_radius(1.0)
                    .unwrap(),
                Callbacks::empty()
                    .with_terminator(TrustRegionGTerminator::new(1e-3).unwrap())
                    .with_terminator(MaxSteps(2_000)),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 1e-6);
    }

    #[test]
    fn test_trust_region_dogleg_with_transform() {
        let problem = IllConditionedQuadratic;
        let mut solver = TrustRegion::default();
        let result = solver
            .process(
                &problem,
                &(),
                TrustRegionConfig::new([1.5, -1.5])
                    .with_transform(&ScaleTransform::from_parameter_scales([2.0, 0.5]).unwrap()),
                Callbacks::empty()
                    .with_terminator(TrustRegionGTerminator::new(1e-4).unwrap())
                    .with_terminator(MaxSteps(200)),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 1e-8);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-4);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-4);
    }
}
