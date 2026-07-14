//! Trust-region minimization.

use std::{marker::PhantomData, ops::ControlFlow};

use crate::core::{
    Callbacks, EvalCounts, LinearAlgebra, LinearSolve, Matrix, MinimizationSummary,
    NalgebraProvider, PseudoInverse, RealScalar, Vector,
};
pub use crate::traits::{Algorithm, CostFunction, Gradient, Status, Transform, TransformedProblem};
use crate::{
    algorithms::gradient::GradientStatus,
    error::{GaneshError, GaneshResult},
    traits::{SupportsParameterNames, Terminator},
};

impl<T: RealScalar, B: LinearAlgebra<T>> SupportsParameterNames for TrustRegionConfig<T, B> {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

/// Terminates [`TrustRegion`] once the gradient norm is sufficiently small.
#[derive(Clone)]
pub struct TrustRegionGTerminator<T: RealScalar = f64> {
    /// Absolute gradient-norm tolerance.
    pub eps_abs: T,
}

impl<T: RealScalar> Default for TrustRegionGTerminator<T> {
    fn default() -> Self {
        Self {
            eps_abs: T::epsilon().cbrt(),
        }
    }
}

impl<T: RealScalar> TrustRegionGTerminator<T> {
    /// Construct a gradient terminator with a validated absolute tolerance.
    pub fn new(eps_abs: T) -> GaneshResult<Self> {
        if eps_abs <= T::zero() {
            return Err(GaneshError::ConfigError(
                "eps_abs must be greater than 0".to_string(),
            ));
        }
        Ok(Self { eps_abs })
    }
}

impl<T, B, P, U, E>
    Terminator<TrustRegion<T, B>, P, GradientStatus<T, B>, U, E, TrustRegionConfig<T, B>>
    for TrustRegionGTerminator<T>
where
    T: RealScalar,
    B: LinearAlgebra<T> + LinearSolve<T> + PseudoInverse<T>,
    P: Gradient<T, B, U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut TrustRegion<T, B>,
        _problem: &P,
        status: &mut GradientStatus<T, B>,
        _args: &U,
        _config: &TrustRegionConfig<T, B>,
    ) -> ControlFlow<()> {
        if algorithm.g.norm() < self.eps_abs {
            status
                .set_message()
                .succeed_with_message("GRADIENT CONVERGED");
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Compact result returned by fixed-step convenience runs.
#[derive(Clone, Debug)]
pub struct MinimizationResult<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Final parameters produced by the optimizer.
    pub x: Vector<T, B>,
    /// Final objective value.
    pub fx: T,
    /// Evaluation counts performed by the run.
    pub evals: EvalCounts,
}

/// Subproblem used by [`TrustRegion`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TrustRegionSubproblem {
    /// Cauchy-point steepest descent step.
    CauchyPoint,
    /// Powell dogleg step.
    #[default]
    Dogleg,
}

/// Trust-region configuration.
pub struct TrustRegionConfig<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Subproblem solver used for each step.
    subproblem: TrustRegionSubproblem,
    /// Initial trust-region radius.
    initial_radius: T,
    /// Maximum trust-region radius.
    max_radius: T,
    /// Minimum ratio of actual to predicted reduction needed to accept a step.
    eta: T,
    /// Optional user-facing parameter names copied into summaries.
    parameter_names: Option<Vec<String>>,
    /// Optional differentiable coordinate transform.
    transform: Option<Box<dyn Transform<T, B>>>,
}

impl<T, B> Default for TrustRegionConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            subproblem: TrustRegionSubproblem::default(),
            initial_radius: T::one(),
            max_radius: T::literal(1_000.0),
            eta: T::literal(1e-4),
            parameter_names: None,
            transform: None,
        }
    }
}

impl<T, B> TrustRegionConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Create a configuration with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the trust-region subproblem solver.
    pub const fn with_subproblem(mut self, subproblem: TrustRegionSubproblem) -> Self {
        self.subproblem = subproblem;
        self
    }

    /// Set the initial trust-region radius.
    pub fn with_initial_radius(mut self, initial_radius: T) -> GaneshResult<Self> {
        if !initial_radius.is_finite() || initial_radius <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Initial radius must be finite and greater than 0".to_string(),
            ));
        }
        self.initial_radius = initial_radius;
        Ok(self)
    }

    /// Set the maximum trust-region radius.
    pub fn with_max_radius(mut self, max_radius: T) -> GaneshResult<Self> {
        if !max_radius.is_finite() || max_radius <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Maximum radius must be finite and greater than 0".to_string(),
            ));
        }
        self.max_radius = max_radius;
        Ok(self)
    }

    /// Set the step-acceptance threshold.
    pub fn with_eta(mut self, eta: T) -> GaneshResult<Self> {
        if !eta.is_finite() || eta < T::zero() || eta >= T::one() {
            return Err(GaneshError::ConfigError(
                "eta must be finite and in the range [0, 1)".to_string(),
            ));
        }
        self.eta = eta;
        Ok(self)
    }

    /// Configure a differentiable coordinate transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: Transform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

/// Trust-region optimizer.
#[derive(Clone, Debug)]
pub struct TrustRegion<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    x: Vector<T, B>,
    f: T,
    g: Vector<T, B>,
    h: Matrix<T, B>,
    radius: T,
    max_radius: T,
    _provider: PhantomData<B>,
}

impl<T, B> Default for TrustRegion<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            x: Vector::<T, B>::zeros(0),
            f: T::zero(),
            g: Vector::<T, B>::zeros(0),
            h: Matrix::<T, B>::zeros(0, 0),
            radius: T::one(),
            max_radius: T::literal(1_000.0),
            _provider: PhantomData,
        }
    }
}

impl<T, B> TrustRegion<T, B>
where
    T: RealScalar,
    B: LinearSolve<T>,
{
    fn predicted_reduction(&self, p: &Vector<T, B>) -> T {
        -T::literal(0.5).mul_add(p.dot(&self.h.mul_vec(p)), self.g.dot(p))
    }

    fn cauchy_point(&self) -> Vector<T, B> {
        let g_norm = self.g.norm();
        if g_norm <= T::epsilon() {
            return Vector::<T, B>::zeros(self.g.len());
        }

        let g_bg = self.g.dot(&self.h.mul_vec(&self.g));
        let tau = if g_bg <= T::zero() {
            T::one()
        } else {
            let candidate = g_norm.powi(3) / (self.radius * g_bg);
            if candidate < T::one() {
                candidate
            } else {
                T::one()
            }
        };
        self.g.scale(-(tau * self.radius / g_norm))
    }

    fn dogleg(&self) -> Vector<T, B> {
        let p_u = {
            let g_bg = self.g.dot(&self.h.mul_vec(&self.g));
            if g_bg <= T::epsilon() {
                self.cauchy_point()
            } else {
                self.g.scale(-(self.g.dot(&self.g) / g_bg))
            }
        };
        if p_u.norm() >= self.radius {
            return p_u.scale(self.radius / p_u.norm());
        }

        let p_b = self.h.lu_solve(&self.g.neg()).filter(Vector::all_finite);
        let Some(p_b) = p_b else {
            return self.cauchy_point();
        };
        if p_b.norm() <= self.radius {
            return p_b;
        }

        let diff = p_b.sub(&p_u);
        let two = T::literal(2.0);
        let four = T::literal(4.0);
        let a = diff.dot(&diff);
        let b = two * p_u.dot(&diff);
        let c = self.radius.mul_add(-self.radius, p_u.dot(&p_u));
        let disc = b.mul_add(b, -four * a * c);
        let disc = if disc > T::zero() { disc } else { T::zero() };
        let tau = (-b + disc.sqrt()) / (two * a);
        p_u.add(&diff.scale(tau))
    }

    fn step_direction(&self, subproblem: TrustRegionSubproblem) -> Vector<T, B> {
        match subproblem {
            TrustRegionSubproblem::CauchyPoint => self.cauchy_point(),
            TrustRegionSubproblem::Dogleg => self.dogleg(),
        }
    }

    /// Run a fixed number of trust-region steps.
    ///
    /// # Errors
    ///
    /// Returns an error if the objective or derivatives fail to evaluate.
    pub fn run_steps<P, U, E>(
        &mut self,
        problem: &P,
        args: &U,
        init: Vector<T, B>,
        config: TrustRegionConfig<T, B>,
        steps: usize,
    ) -> Result<MinimizationResult<T, B>, E>
    where
        P: Gradient<T, B, U, E>,
    {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        self.x = transformed.to_internal(&init);
        (self.f, self.g, self.h) = transformed.evaluate_with_gradient_and_hessian(&self.x, args)?;
        self.radius = config.initial_radius;
        self.max_radius = if config.max_radius > config.initial_radius {
            config.max_radius
        } else {
            config.initial_radius
        };

        let mut evals = EvalCounts::new(1, 1, 1);
        let quarter = T::literal(0.25);
        let three_quarters = T::literal(0.75);
        let two = T::literal(2.0);
        let boundary_tol_scale = T::literal(1e-12);

        for _ in 0..steps {
            let p = self.step_direction(config.subproblem);
            let predicted = self.predicted_reduction(&p);
            if predicted <= T::epsilon() {
                self.radius = self.radius * quarter;
                continue;
            }

            let x_trial = self.x.add(&p);
            let f_trial = transformed.evaluate(&x_trial, args)?;
            evals.record_f();
            let rho = (self.f - f_trial) / predicted;
            let hits_boundary =
                (p.norm() - self.radius).abs() <= boundary_tol_scale * (T::one() + self.radius);

            if rho < quarter {
                self.radius = quarter * p.norm();
            } else if rho > three_quarters && hits_boundary {
                let candidate = two * self.radius;
                self.radius = if candidate < self.max_radius {
                    candidate
                } else {
                    self.max_radius
                };
            }

            if rho > config.eta {
                let (g_trial, h_trial) = transformed.gradient_with_hessian(&x_trial, args)?;
                evals.record_gh();
                self.x = x_trial;
                self.f = f_trial;
                self.g = g_trial;
                self.h = h_trial;
            }
        }

        Ok(MinimizationResult {
            x: transformed.to_external(&self.x),
            fx: self.f,
            evals,
        })
    }
}

impl<T, B, P, U, E> Algorithm<P, GradientStatus<T, B>, U, E> for TrustRegion<T, B>
where
    T: RealScalar,
    B: LinearSolve<T> + PseudoInverse<T>,
    P: Gradient<T, B, U, E>,
{
    type Summary = MinimizationSummary<T, B>;
    type Config = TrustRegionConfig<T, B>;
    type Init = Vector<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut GradientStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        self.x = transformed.to_internal(init);
        (self.f, self.g, self.h) = transformed.evaluate_with_gradient_and_hessian(&self.x, args)?;
        self.radius = config.initial_radius;
        self.max_radius = if config.max_radius > config.initial_radius {
            config.max_radius
        } else {
            config.initial_radius
        };
        status.evals.record_fgh();
        status.initialize(init.clone(), self.f);
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut GradientStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        let p = self.step_direction(config.subproblem);
        let predicted = self.predicted_reduction(&p);
        let quarter = T::literal(0.25);
        if predicted <= T::epsilon() {
            self.radius = self.radius * quarter;
            return Ok(());
        }

        let x_trial = self.x.add(&p);
        let f_trial = transformed.evaluate(&x_trial, args)?;
        status.evals.record_f();
        let rho = (self.f - f_trial) / predicted;
        let three_quarters = T::literal(0.75);
        let hits_boundary =
            (p.norm() - self.radius).abs() <= T::literal(1e-12) * (T::one() + self.radius);
        if rho < quarter {
            self.radius = quarter * p.norm();
        } else if rho > three_quarters && hits_boundary {
            let candidate = T::literal(2.0) * self.radius;
            self.radius = if candidate < self.max_radius {
                candidate
            } else {
                self.max_radius
            };
        }

        if rho > config.eta {
            let (gradient, hessian) = transformed.gradient_with_hessian(&x_trial, args)?;
            status.evals.record_gh();
            self.x = x_trial;
            self.f = f_trial;
            self.g = gradient;
            self.h = hessian;
            status.set_position(transformed.to_external(&self.x), self.f);
        }
        Ok(())
    }

    fn postprocessing(
        &mut self,
        problem: &P,
        status: &mut GradientStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        let hessian = problem.hessian(&transformed.to_external(&self.x), args)?;
        status.evals.record_h();
        status.set_hess(hessian);
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &GradientStatus<T, B>,
        _args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        let n = status.x.len();
        Ok(MinimizationSummary {
            bounds: config
                .transform
                .as_deref()
                .and_then(|transform| transform.parameter_bounds())
                .map(Vec::from),
            parameter_names: config.parameter_names.clone(),
            message: status.message.clone(),
            x0: init.clone(),
            x: status.x.clone(),
            std: status
                .err
                .clone()
                .unwrap_or_else(|| crate::core::summary::unknown_uncertainties(n)),
            fx: status.fx,
            evals: status.evals,
            covariance: status.cov.clone().unwrap_or_else(|| Matrix::identity(n)),
        })
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty().with_terminator(TrustRegionGTerminator::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::gradient_free::{DifferentialEvolution, DifferentialEvolutionConfig};
    use crate::core::MaxSteps;
    #[cfg(feature = "backend-ndarray")]
    use crate::core::NdArrayProvider;
    use crate::traits::{Bounds, ScaleTransform};
    use approx::assert_relative_eq;
    use std::convert::Infallible;

    struct Quadratic;

    impl<T, B> CostFunction<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn evaluate(&self, x: &Vector<T, B>, _args: &()) -> Result<T, Infallible> {
            Ok(x.dot(x))
        }
    }

    impl<T, B> Gradient<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn gradient(&self, x: &Vector<T, B>, _args: &()) -> Result<Vector<T, B>, Infallible> {
            Ok(x.scale(T::literal(2.0)))
        }

        fn hessian(&self, x: &Vector<T, B>, _args: &()) -> Result<Matrix<T, B>, Infallible> {
            Ok(Matrix::<T, B>::identity(x.len()).scale(T::literal(2.0)))
        }
    }

    fn vector<T, B>(values: &[f64]) -> Vector<T, B>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Vector::<T, B>::from_vec(values.iter().map(|value| T::literal(*value)).collect())
    }

    #[test]
    fn trust_region_f64_reaches_quadratic_minimum() {
        let result = TrustRegion::<f64>::default()
            .run_steps(
                &Quadratic,
                &(),
                vector::<f64, NalgebraProvider>(&[3.0, -2.0]),
                TrustRegionConfig::<f64>::default(),
                12,
            )
            .unwrap();
        assert!(result.fx < 1e-20);
        assert_relative_eq!(Vector::get(&result.x, 0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(Vector::get(&result.x, 1), 0.0, epsilon = 1e-10);
        assert!(result.evals.g() > 0);
        assert!(result.evals.h() > 0);
    }

    #[test]
    fn default_scalar_syntax_is_f64_and_explicit_f32_compiles() {
        let _: TrustRegion = TrustRegion::<f64>::default();
        let _: TrustRegion<f32> = TrustRegion::<f32>::default();
        let _: TrustRegionConfig = TrustRegionConfig::<f64>::default();
        let _: DifferentialEvolution = DifferentialEvolution::<f64>::new(Some(1));
        let _: DifferentialEvolutionConfig = DifferentialEvolutionConfig::<f64>::default();
    }

    #[test]
    fn differential_evolution_f64_is_seeded() {
        let run = || {
            let config = DifferentialEvolutionConfig::<f64>::default()
                .with_population_size(24)
                .unwrap()
                .with_initial_scale(2.0)
                .unwrap();
            DifferentialEvolution::<f64>::new(Some(7))
                .run_steps(
                    &Quadratic,
                    &(),
                    vector::<f64, NalgebraProvider>(&[3.0, -2.0]),
                    config,
                    80,
                )
                .unwrap()
        };
        let result_a = run();
        let result_b = run();
        assert_eq!(result_a.fx, result_b.fx);
        assert_eq!(result_a.x, result_b.x);
        assert!(result_a.fx < 1e-4);
    }

    #[test]
    fn differential_evolution_uses_algorithm_lifecycle_with_bounds() {
        let bounds = Bounds::<f32, NalgebraProvider>::new([(-4.0, 4.0), (-4.0, 4.0)]).unwrap();
        let config = DifferentialEvolutionConfig::<f32>::default()
            .with_population_size(24)
            .unwrap()
            .with_initial_scale(1.0)
            .unwrap()
            .with_transform(bounds);
        let mut algorithm = DifferentialEvolution::<f32>::new(Some(7));
        let result = algorithm
            .process(
                &Quadratic,
                &(),
                vector::<f32, NalgebraProvider>(&[3.0, -2.0]),
                config,
                Callbacks::empty().with_terminator(MaxSteps(80)),
            )
            .unwrap();
        assert!(result.fx < 1e-4);
        assert!(result.x.get(0).abs() < 0.1);
        assert!(result.x.get(1).abs() < 0.1);
        assert!(result.evals.f() > 24);
    }

    #[test]
    fn trust_region_f32_builds_in_the_default_crate() {
        let result = TrustRegion::<f32>::default()
            .run_steps(
                &Quadratic,
                &(),
                vector::<f32, NalgebraProvider>(&[3.0, -2.0]),
                TrustRegionConfig::<f32>::default(),
                12,
            )
            .unwrap();
        assert!(result.fx < 1e-10);
    }

    #[test]
    fn trust_region_uses_the_production_algorithm_lifecycle() {
        let mut algorithm = TrustRegion::<f32>::default();
        let result = algorithm
            .process_default(
                &Quadratic,
                &(),
                vector::<f32, NalgebraProvider>(&[3.0, -2.0]),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 1e-8);
        assert!(result.evals.f() > 0);
        assert!(result.evals.g() > 0);
        assert!(result.evals.h() > 0);
    }

    #[test]
    fn trust_region_runs_through_provider_generic_transform() {
        let transform =
            ScaleTransform::<f32, NalgebraProvider>::from_parameter_scales([2.0, 0.5]).unwrap();
        let config = TrustRegionConfig::<f32>::default().with_transform(transform);
        let mut algorithm = TrustRegion::<f32>::default();
        let result = algorithm
            .process(
                &Quadratic,
                &(),
                vector::<f32, NalgebraProvider>(&[3.0, -2.0]),
                config,
                TrustRegion::<f32>::default_callbacks(),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 1e-8);
    }

    #[cfg(feature = "backend-ndarray")]
    #[test]
    fn trust_region_runs_with_ndarray_provider() {
        let result = TrustRegion::<f64, NdArrayProvider>::default()
            .run_steps(
                &Quadratic,
                &(),
                vector::<f64, NdArrayProvider>(&[3.0, -2.0]),
                TrustRegionConfig::<f64, NdArrayProvider>::default(),
                12,
            )
            .unwrap();
        assert!(result.fx < 1e-20);
        assert_relative_eq!(Vector::get(&result.x, 0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(Vector::get(&result.x, 1), 0.0, epsilon = 1e-10);
    }

    #[cfg(feature = "backend-ndarray")]
    #[test]
    fn differential_evolution_runs_with_ndarray_provider() {
        let config = DifferentialEvolutionConfig::<f64, NdArrayProvider>::default()
            .with_population_size(24)
            .unwrap()
            .with_initial_scale(2.0)
            .unwrap();
        let result = DifferentialEvolution::<f64, NdArrayProvider>::new(Some(7))
            .run_steps(
                &Quadratic,
                &(),
                vector::<f64, NdArrayProvider>(&[3.0, -2.0]),
                config,
                80,
            )
            .unwrap();
        assert!(result.fx < 1e-4);
    }
}
