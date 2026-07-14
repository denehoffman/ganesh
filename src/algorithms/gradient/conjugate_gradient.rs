use crate::algorithms::{gradient::GradientStatus, line_search::StrongWolfeLineSearch};
use crate::core::{
    Callbacks, LinearAlgebra, Matrix, MaxSteps, MinimizationSummary, NalgebraProvider,
    PseudoInverse, RealScalar, Vector,
};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{
    Algorithm, Gradient, LineSearch, LineSearchOutput, Status, SupportsParameterNames, Terminator,
    Transform, TransformedProblem,
};
use std::marker::PhantomData;
use std::ops::ControlFlow;

/// Terminates [`ConjugateGradient`] once the gradient norm is sufficiently small.
#[derive(Clone)]
pub struct ConjugateGradientGTerminator<T: RealScalar = f64> {
    /// Absolute gradient-norm tolerance.
    pub eps_abs: T,
}

impl<T: RealScalar, B: LinearAlgebra<T>, L> SupportsParameterNames
    for ConjugateGradientConfig<T, B, L>
{
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

impl<T: RealScalar> Default for ConjugateGradientGTerminator<T> {
    fn default() -> Self {
        Self {
            eps_abs: T::epsilon().cbrt(),
        }
    }
}

impl<T: RealScalar> ConjugateGradientGTerminator<T> {
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

impl<T, B, L, P, U, E>
    Terminator<
        ConjugateGradient<T, B, L>,
        P,
        GradientStatus<T, B>,
        U,
        E,
        ConjugateGradientConfig<T, B, L>,
    > for ConjugateGradientGTerminator<T>
where
    T: RealScalar,
    B: LinearAlgebra<T> + PseudoInverse<T>,
    P: Gradient<T, B, U, E>,
    L: for<'a> LineSearch<T, B, TransformedProblem<'a, P, T, B>, U, E>
        + Clone
        + Default
        + Send
        + Sync,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut ConjugateGradient<T, B, L>,
        _problem: &P,
        status: &mut GradientStatus<T, B>,
        _args: &U,
        _config: &ConjugateGradientConfig<T, B, L>,
    ) -> ControlFlow<()> {
        if algorithm.gradient.norm() < self.eps_abs {
            status
                .set_message()
                .succeed_with_message("GRADIENT CONVERGED");
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// The update formula used to compute the nonlinear conjugate-gradient coefficient `beta_k`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ConjugateGradientUpdate {
    /// Fletcher–Reeves update.
    FletcherReeves,
    /// Polak–Ribiere update clipped at zero.
    #[default]
    PolakRibierePlus,
    /// Hestenes–Stiefel update clipped at zero.
    HestenesStiefelPlus,
    /// Dai–Yuan update.
    DaiYuan,
    /// Hager–Zhang update.
    HagerZhang,
}

/// Configuration for the scalar- and linear-algebra-generic nonlinear conjugate-gradient optimizer.
pub struct ConjugateGradientConfig<
    T: RealScalar = f64,
    B: LinearAlgebra<T> = NalgebraProvider,
    L = StrongWolfeLineSearch<T, B>,
> {
    /// Conjugate-gradient coefficient update.
    update: ConjugateGradientUpdate,
    /// Line-search implementation.
    line_search: L,
    /// Optional user-facing parameter names copied into summaries.
    parameter_names: Option<Vec<String>>,
    /// Optional coordinate transform.
    transform: Option<Box<dyn Transform<T, B>>>,
}

impl<T, B, L> Default for ConjugateGradientConfig<T, B, L>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    L: Default,
{
    fn default() -> Self {
        Self {
            update: ConjugateGradientUpdate::default(),
            line_search: L::default(),
            parameter_names: None,
            transform: None,
        }
    }
}

impl<T, B, L> ConjugateGradientConfig<T, B, L>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Create a configuration with default hyperparameters.
    pub fn new() -> Self
    where
        L: Default,
    {
        Self::default()
    }

    /// Replace the line-search implementation, changing the configuration's line-search type.
    pub fn with_line_search<N>(self, line_search: N) -> ConjugateGradientConfig<T, B, N> {
        ConjugateGradientConfig {
            update: self.update,
            line_search,
            parameter_names: self.parameter_names,
            transform: self.transform,
        }
    }

    /// Select the coefficient update formula.
    pub const fn with_update(mut self, update: ConjugateGradientUpdate) -> Self {
        self.update = update;
        self
    }

    /// Configure a coordinate transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: Transform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

/// Scalar- and linear-algebra-generic nonlinear conjugate-gradient optimizer.
#[derive(Clone, Debug)]
pub struct ConjugateGradient<
    T: RealScalar = f64,
    B: LinearAlgebra<T> = NalgebraProvider,
    L = StrongWolfeLineSearch<T, B>,
> {
    x: Vector<T, B>,
    fx: T,
    gradient: Vector<T, B>,
    direction: Vector<T, B>,
    line_search: L,
    _provider: PhantomData<B>,
}

impl<T, B, L> Default for ConjugateGradient<T, B, L>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    L: Default,
{
    fn default() -> Self {
        Self {
            x: Vector::zeros(0),
            fx: T::infinity(),
            gradient: Vector::zeros(0),
            direction: Vector::zeros(0),
            line_search: L::default(),
            _provider: PhantomData,
        }
    }
}

impl<T, B, L> ConjugateGradient<T, B, L>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn beta(&self, update: ConjugateGradientUpdate, next: &Vector<T, B>) -> T {
        let y = next.sub(&self.gradient);
        let gradient_norm_sq = self.gradient.dot(&self.gradient);
        let next_norm_sq = next.dot(next);
        let direction_y = self.direction.dot(&y);
        let zero = T::zero();
        let beta = match update {
            ConjugateGradientUpdate::FletcherReeves => {
                if gradient_norm_sq <= T::epsilon() {
                    zero
                } else {
                    next_norm_sq / gradient_norm_sq
                }
            }
            ConjugateGradientUpdate::PolakRibierePlus => {
                if gradient_norm_sq <= T::epsilon() {
                    zero
                } else {
                    let value = next.dot(&y) / gradient_norm_sq;
                    if value > zero {
                        value
                    } else {
                        zero
                    }
                }
            }
            ConjugateGradientUpdate::HestenesStiefelPlus => {
                if direction_y.abs() <= T::epsilon() {
                    zero
                } else {
                    let value = next.dot(&y) / direction_y;
                    if value > zero {
                        value
                    } else {
                        zero
                    }
                }
            }
            ConjugateGradientUpdate::DaiYuan => {
                if direction_y.abs() <= T::epsilon() {
                    zero
                } else {
                    next_norm_sq / direction_y
                }
            }
            ConjugateGradientUpdate::HagerZhang => {
                if direction_y.abs() <= T::epsilon() {
                    zero
                } else {
                    let correction = y.sub(
                        &self
                            .direction
                            .scale(T::literal(2.0) * y.dot(&y) / direction_y),
                    );
                    correction.dot(next) / direction_y
                }
            }
        };
        if beta.is_finite() {
            beta
        } else {
            zero
        }
    }
}

impl<T, B, L, P, U, E> Algorithm<P, GradientStatus<T, B>, U, E> for ConjugateGradient<T, B, L>
where
    T: RealScalar,
    B: LinearAlgebra<T> + PseudoInverse<T>,
    P: Gradient<T, B, U, E>,
    L: for<'a> LineSearch<T, B, TransformedProblem<'a, P, T, B>, U, E>
        + Clone
        + Default
        + Send
        + Sync,
{
    type Summary = MinimizationSummary<T, B>;
    type Config = ConjugateGradientConfig<T, B, L>;
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
        (self.fx, self.gradient) = transformed.evaluate_with_gradient(&self.x, args)?;
        self.direction = self.gradient.neg();
        self.line_search = config.line_search.clone();
        status.evals.record_fg();
        status.initialize(init.clone(), self.fx);
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
        if self.direction.dot(&self.gradient) >= T::zero() {
            self.direction = self.gradient.neg();
        }
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        if let Ok(LineSearchOutput {
            alpha,
            fx,
            gradient,
        }) = self.line_search.search(
            &self.x,
            &self.direction,
            None,
            &transformed,
            args,
            &mut status.evals,
        )? {
            self.x = self.x.add_scaled(&self.direction, alpha);
            self.fx = fx;
            let beta = self.beta(config.update, &gradient);
            self.gradient = gradient;
            self.direction = self.gradient.neg().add_scaled(&self.direction, beta);
            if self.direction.dot(&self.gradient) >= T::zero() {
                self.direction = self.gradient.neg();
            }
            status.set_position(transformed.to_external(&self.x), self.fx);
        } else {
            self.direction = self.gradient.neg();
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
        let external = transformed.to_external(&self.x);
        let hessian = problem.hessian(&external, args)?;
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
        let dimension = status.x.len();
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
                .unwrap_or_else(|| crate::core::summary::unknown_uncertainties(dimension)),
            fx: status.fx,
            evals: status.evals,
            covariance: status
                .cov
                .clone()
                .unwrap_or_else(|| Matrix::identity(dimension)),
        })
    }

    fn reset(&mut self) {
        self.x = Vector::zeros(0);
        self.fx = T::infinity();
        self.gradient = Vector::zeros(0);
        self.direction = Vector::zeros(0);
        self.line_search = L::default();
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty()
            .with_terminator(ConjugateGradientGTerminator::default())
            .with_terminator(MaxSteps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algorithms::line_search::HagerZhangLineSearch, core::MaxSteps, test_functions::Rosenbrock,
        traits::Bounds,
    };
    use approx::assert_relative_eq;

    #[test]
    fn test_conjugate_gradient_polak_ribiere_plus() {
        let problem = Rosenbrock { n: 2 };
        let mut solver = ConjugateGradient::<f64>::default();
        let result = solver
            .process(
                &problem,
                &(),
                Vector::from_vec(vec![-1.2, 1.0]),
                ConjugateGradientConfig::<f64>::default()
                    .with_update(ConjugateGradientUpdate::PolakRibierePlus),
                ConjugateGradient::<f64>::default_callbacks().with_terminator(MaxSteps(1_000)),
            )
            .unwrap();
        assert!(
            result.message.success(),
            "{} (fx = {})",
            result.message,
            result.fx
        );
        assert!(result.fx < 1e-10);
        assert_relative_eq!(result.x.get(0), 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x.get(1), 1.0, epsilon = 2e-5);
        assert!(result.std.to_vec().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_conjugate_gradient_hager_zhang() {
        let problem = Rosenbrock { n: 2 };
        let mut solver =
            ConjugateGradient::<f64, NalgebraProvider, HagerZhangLineSearch>::default();
        let result = solver
            .process(
                &problem,
                &(),
                Vector::from_vec(vec![-1.2, 1.0]),
                ConjugateGradientConfig::<f64, NalgebraProvider, HagerZhangLineSearch>::default()
                    .with_update(ConjugateGradientUpdate::HagerZhang)
                    .with_line_search(HagerZhangLineSearch::<f64>::default()),
                ConjugateGradient::<f64, NalgebraProvider, HagerZhangLineSearch>::default_callbacks().with_terminator(MaxSteps(1_000)),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 1e-10);
    }

    #[test]
    fn test_conjugate_gradient_fletcher_reeves_with_transform() {
        let problem = Rosenbrock { n: 2 };
        let mut solver = ConjugateGradient::<f64>::default();
        let result = solver
            .process(
                &problem,
                &(),
                Vector::from_vec(vec![-1.2, 1.0]),
                ConjugateGradientConfig::<f64>::default()
                    .with_update(ConjugateGradientUpdate::FletcherReeves)
                    .with_transform(Bounds::new([(-2.0, 2.0), (-1.0, 3.0)]).unwrap()),
                ConjugateGradient::<f64>::default_callbacks().with_terminator(MaxSteps(10_000)),
            )
            .unwrap();
        assert!(
            result.message.success(),
            "{} (fx = {})",
            result.message,
            result.fx
        );
        assert!(result.fx < 1e-10);
        assert!(result.x.get(0) >= -2.0 && result.x.get(0) <= 2.0);
        assert!(result.x.get(1) >= -1.0 && result.x.get(1) <= 3.0);
    }

    #[test]
    fn provider_conjugate_gradient_runs_f32_with_bounds() {
        let problem = Rosenbrock { n: 2 };
        let bounds = Bounds::new([(-2.0_f32, 2.0), (-1.0, 3.0)]).unwrap();
        let config = ConjugateGradientConfig::<f32>::default().with_transform(bounds);
        let mut solver = ConjugateGradient::<f32>::default();
        let result = solver
            .process(
                &problem,
                &(),
                Vector::from_vec(vec![-1.2, 1.0]),
                config,
                ConjugateGradient::<f32>::default_callbacks().with_terminator(MaxSteps(5_000)),
            )
            .unwrap();
        assert!(result.fx < 1e-3);
        assert!((result.x.get(0) - 1.0).abs() < 0.05);
        assert!((result.x.get(1) - 1.0).abs() < 0.05);
    }
}
