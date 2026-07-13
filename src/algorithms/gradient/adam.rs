use crate::algorithms::gradient::BackendGradientStatus;
use crate::core::{
    BackendMinimizationSummary, LinearAlgebra, Matrix, NalgebraBackend, RealScalar, Vector,
};
use crate::traits::{BackendTransform, BackendTransformedProblem, CostFunction, Gradient};
use crate::{
    algorithms::gradient::LegacyGradientStatus,
    core::{Callbacks, LegacyMinimizationSummary},
    error::{GaneshError, GaneshResult},
    traits::{
        Algorithm, LegacyCostFunction, LegacyGradient, Status, SupportsParameterNames,
        SupportsTransform, Terminator, Transform, TransformedProblem,
    },
    DMatrix, DVector, Float,
};
use std::marker::PhantomData;
use std::ops::ControlFlow;

/// A [`Terminator`] which terminates the [`Adam`] algorithm if the Exponential Moving Average (EMA)
/// loss does not improve in the number of steps defined by the [`AdamConfig`] `patience`
/// parameter.
#[derive(Copy, Clone)]
pub struct AdamEMATerminator {
    /// The value for the slope of the exponential moving average loss decay (default = `0.9`).
    pub beta_c: Float,
    /// The minimum change in EMA loss which will increase the patience counter (default = `MACH_EPS^(1/2)`).
    pub eps_loss: Float,
    /// The number of allowed iterations with no improvement in the loss (according to an exponential moving average) before the algorithm terminates (default = `1`).
    pub patience: usize,
}
impl Default for AdamEMATerminator {
    fn default() -> Self {
        Self {
            beta_c: 0.9,
            eps_loss: Float::EPSILON.sqrt(),
            patience: 1,
        }
    }
}
impl<P, U, E> Terminator<Adam, P, LegacyGradientStatus, U, E, AdamConfig> for AdamEMATerminator
where
    P: LegacyGradient<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut Adam,
        _problem: &P,
        status: &mut LegacyGradientStatus,
        _args: &U,
        _config: &AdamConfig,
    ) -> ControlFlow<()> {
        let prev_ema_loss = algorithm.ema_loss;
        algorithm.ema_loss = self
            .beta_c
            .mul_add(prev_ema_loss, (1.0 - self.beta_c) * algorithm.f);
        if (algorithm.ema_loss - prev_ema_loss).abs() < self.eps_loss {
            algorithm.ema_counter += 1;
        } else {
            algorithm.ema_counter = 0;
        }
        if algorithm.ema_counter >= self.patience {
            status.set_message().succeed_with_message(format!(
                "EMA LOSS HAS NOT IMPROVED IN {} STEPS",
                algorithm.ema_counter
            ));
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// The internal configuration struct for the [`Adam`] algorithm.
#[derive(Clone)]
pub struct AdamConfig {
    parameter_names: Option<Vec<String>>,
    transform: Option<Box<dyn Transform>>,
    alpha: Float,
    beta_1: Float,
    beta_2: Float,
    epsilon: Float,
}
impl AdamConfig {
    /// Create a new configuration with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }
    /// Set the initial learning rate $`\alpha`$ (default = `0.001`).
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `value` is not strictly positive.
    pub fn with_alpha(mut self, value: Float) -> GaneshResult<Self> {
        if value <= 0.0 {
            return Err(GaneshError::ConfigError(
                "Initial learning rate must be positive and greater than 0".to_string(),
            ));
        }
        self.alpha = value;
        Ok(self)
    }
    /// Set the value for the hyperparameter $`\beta_1`$ (default = `0.9`).
    ///
    /// This represents the exponential decay rate of the first moment estimate, $`m`$.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `value` is not in the interval `[0, 1)`.
    pub fn with_beta_1(mut self, value: Float) -> GaneshResult<Self> {
        if !(0.0..1.0).contains(&value) {
            return Err(GaneshError::ConfigError(
                "beta_1 must be in the range [0, 1)".to_string(),
            ));
        }
        self.beta_1 = value;
        Ok(self)
    }
    /// Set the value for the hyperparameter $`\beta_2`$ (default = `0.999`).
    ///
    /// This represents the exponential decay rate of the second moment estimate, $`v`$.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `value` is not in the interval `[0, 1)`.
    pub fn with_beta_2(mut self, value: Float) -> GaneshResult<Self> {
        if !(0.0..1.0).contains(&value) {
            return Err(GaneshError::ConfigError(
                "beta_2 must be in the range [0, 1)".to_string(),
            ));
        }
        self.beta_2 = value;
        Ok(self)
    }
    /// Set the value for the divide-by-zero tolerance in the update step (default = `1e-8`).
    ///
    /// This ensures the update does not divide by zero if the bias-corrected second raw moment
    /// estimate is zero for any parameter.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `value` is not strictly positive.
    pub fn with_epsilon(mut self, value: Float) -> GaneshResult<Self> {
        if value <= 0.0 {
            return Err(GaneshError::ConfigError(
                "Divide-by-zero tolerance must be positive and greater than 0".to_string(),
            ));
        }
        self.epsilon = value;
        Ok(self)
    }
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            parameter_names: None,
            transform: None,
            alpha: 0.001,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8, // NOTE: I think this can be independent of bit precision
        }
    }
}
impl SupportsTransform for AdamConfig {
    fn get_transform_mut(&mut self) -> &mut Option<Box<dyn Transform>> {
        &mut self.transform
    }
}
impl SupportsParameterNames for AdamConfig {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

/// The Adam algorithm.
///
/// This minimization [`Algorithm`] is intended to be used with stochastic objective functions, and
/// the algorithm itself is described in [^1].
///
/// [^1]: [D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” 2014, arXiv. doi: 10.48550/ARXIV.1412.6980.](https://doi.org/10.48550/ARXIV.1412.6980)
#[derive(Clone, Default)]
pub struct Adam {
    x: DVector<Float>,
    f: Float,
    g: DVector<Float>,
    m: DVector<Float>,
    v: DVector<Float>,
    ema_loss: Float,
    ema_counter: usize,
}
impl<P, U, E> Algorithm<P, LegacyGradientStatus, U, E> for Adam
where
    P: LegacyGradient<U, E>,
{
    type Summary = LegacyMinimizationSummary;
    type Config = AdamConfig;
    type Init = DVector<Float>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut LegacyGradientStatus,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let t_problem = TransformedProblem::new(problem, &config.transform);
        self.x = t_problem.to_owned_internal(init);
        self.g = DVector::zeros(self.x.len());
        self.f = t_problem.evaluate(&self.x, args)?;
        status.initialize((init.clone(), self.f));
        status.evals.record_f();
        self.m = DVector::zeros(self.x.len());
        self.v = DVector::zeros(self.x.len());
        Ok(())
    }

    fn step(
        &mut self,
        i_step: usize,
        problem: &P,
        status: &mut LegacyGradientStatus,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let t_problem = TransformedProblem::new(problem, &config.transform);
        self.g = t_problem.gradient(&self.x, args)?;
        status.evals.record_g();
        self.m = self.m.scale(config.beta_1) + self.g.scale(1.0 - config.beta_1);
        self.v =
            self.v.scale(config.beta_2) + self.g.map(|gi| gi.powi(2)).scale(1.0 - config.beta_2);
        let alpha_t = config.alpha * (1.0 - config.beta_2.powi(i_step as i32 + 1)).sqrt()
            / (1.0 - config.beta_1.powi(i_step as i32 + 1));
        self.x -= self
            .m
            .scale(alpha_t)
            .component_div(&self.v.map(|vi| vi.sqrt() + config.epsilon));
        self.f = t_problem.evaluate(&self.x, args)?;
        status.evals.record_f();
        status.set_position((t_problem.to_owned_external(&self.x), self.f));
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &LegacyGradientStatus,
        _args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        Ok(LegacyMinimizationSummary {
            x0: init.clone(),
            x: status.x.clone(),
            fx: status.fx,
            bounds: None,
            evals: status.evals,
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
        self.ema_loss = 0.0;
        self.ema_counter = 0;
    }

    fn default_callbacks() -> Callbacks<Self, P, LegacyGradientStatus, U, E, Self::Config>
    where
        Self: Sized,
    {
        Callbacks::empty().with_terminator(AdamEMATerminator::default())
    }
}

/// Configuration for scalar- and backend-generic Adam.
pub struct BackendAdamConfig<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Learning rate.
    pub alpha: T,
    /// First-moment decay.
    pub beta_1: T,
    /// Second-moment decay.
    pub beta_2: T,
    /// Denominator stabilizer.
    pub epsilon: T,
    /// EMA decay used for convergence detection.
    pub ema_decay: T,
    /// Minimum EMA change treated as improvement.
    pub ema_tolerance: T,
    /// Consecutive stable EMA steps required for convergence.
    pub patience: usize,
    /// Optional coordinate transform.
    pub transform: Option<Box<dyn BackendTransform<T, B>>>,
}

impl<T, B> Default for BackendAdamConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            alpha: T::literal(0.001),
            beta_1: T::literal(0.9),
            beta_2: T::literal(0.999),
            epsilon: T::literal(1e-8),
            ema_decay: T::literal(0.9),
            ema_tolerance: T::epsilon().sqrt(),
            patience: 1,
            transform: None,
        }
    }
}

impl<T, B> BackendAdamConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Configure a coordinate transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: BackendTransform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

/// Scalar- and backend-generic Adam optimizer.
#[derive(Clone, Debug)]
pub struct BackendAdam<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    x: Vector<T, B>,
    fx: T,
    first_moment: Vector<T, B>,
    second_moment: Vector<T, B>,
    ema_loss: T,
    ema_counter: usize,
    _backend: PhantomData<B>,
}

impl<T, B> Default for BackendAdam<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            x: Vector::zeros(0),
            fx: T::zero(),
            first_moment: Vector::zeros(0),
            second_moment: Vector::zeros(0),
            ema_loss: T::zero(),
            ema_counter: 0,
            _backend: PhantomData,
        }
    }
}

impl<T, B, P, U, E> Algorithm<P, BackendGradientStatus<T, B>, U, E> for BackendAdam<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: Gradient<T, B, U, E>,
{
    type Summary = BackendMinimizationSummary<T, B>;
    type Config = BackendAdamConfig<T, B>;
    type Init = Vector<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut BackendGradientStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        self.x = transformed.to_internal(init);
        self.fx = transformed.evaluate(&self.x, args)?;
        self.first_moment = Vector::zeros(self.x.len());
        self.second_moment = Vector::zeros(self.x.len());
        self.ema_loss = self.fx;
        self.ema_counter = 0;
        status.evals.record_f();
        status.initialize(init.clone(), self.fx);
        Ok(())
    }

    fn step(
        &mut self,
        current_step: usize,
        problem: &P,
        status: &mut BackendGradientStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        let gradient = transformed.gradient(&self.x, args)?;
        status.evals.record_g();
        let one = T::one();
        self.first_moment = self
            .first_moment
            .scale(config.beta_1)
            .add(&gradient.scale(one - config.beta_1));
        self.second_moment = Vector::from_vec(
            (0..gradient.len())
                .map(|index| {
                    config.beta_2 * self.second_moment.get(index)
                        + (one - config.beta_2) * gradient.get(index).powi(2)
                })
                .collect(),
        );
        let exponent = current_step.saturating_add(1).min(i32::MAX as usize) as i32;
        let alpha = config.alpha * (one - config.beta_2.powi(exponent)).sqrt()
            / (one - config.beta_1.powi(exponent));
        self.x = Vector::from_vec(
            (0..self.x.len())
                .map(|index| {
                    self.x.get(index)
                        - alpha * self.first_moment.get(index)
                            / (self.second_moment.get(index).sqrt() + config.epsilon)
                })
                .collect(),
        );
        self.fx = transformed.evaluate(&self.x, args)?;
        status.evals.record_f();
        status.set_position(transformed.to_external(&self.x), self.fx);

        let previous = self.ema_loss;
        self.ema_loss = config.ema_decay * previous + (one - config.ema_decay) * self.fx;
        if (self.ema_loss - previous).abs() < config.ema_tolerance {
            self.ema_counter += 1;
        } else {
            self.ema_counter = 0;
        }
        if self.ema_counter >= config.patience {
            status
                .set_message()
                .succeed_with_message("EMA LOSS CONVERGED");
        }
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
            std: Vector::zeros(dimension),
            fx: status.fx,
            evals: status.evals,
            covariance: Matrix::identity(dimension),
        })
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn default_callbacks() -> Callbacks<Self, P, BackendGradientStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty().with_terminator(crate::core::MaxSteps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{Bounds, MaxSteps},
        test_functions::Rosenbrock,
    };
    use approx::assert_relative_eq;
    use std::convert::Infallible;

    struct GenericQuadratic;

    impl<T, B> crate::traits::CostFunction<T, B> for GenericQuadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn evaluate(&self, x: &Vector<T, B>, _: &()) -> Result<T, Infallible> {
            Ok(x.dot(x))
        }
    }

    impl<T, B> Gradient<T, B> for GenericQuadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn gradient(&self, x: &Vector<T, B>, _: &()) -> Result<Vector<T, B>, Infallible> {
            Ok(x.scale(T::literal(2.0)))
        }
    }

    #[test]
    fn test_adam() {
        let mut solver = Adam::default();
        let problem = Rosenbrock { n: 2 };
        let starting_values = vec![
            [-2.0, 2.0],
            [2.0, 2.0],
            [2.0, -2.0],
            [-2.0, -2.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ];
        for starting_value in starting_values {
            let result = solver
                .process(
                    &problem,
                    &(),
                    DVector::from_row_slice(&starting_value),
                    AdamConfig::default(),
                    Adam::default_callbacks().with_terminator(MaxSteps(1_000_000)),
                )
                .unwrap();
            assert!(result.message.success());
            assert_relative_eq!(result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        }
    }

    #[test]
    fn test_bounded_adam() {
        let mut solver = Adam::default();
        let problem = Rosenbrock { n: 2 };
        let starting_values = vec![
            [-2.0, 2.0],
            [2.0, 2.0],
            [2.0, -2.0],
            [-2.0, -2.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ];
        for starting_value in starting_values {
            let result = solver
                .process(
                    &problem,
                    &(),
                    DVector::from_row_slice(&starting_value),
                    AdamConfig::default().with_transform(&Bounds::from([(-4.0, 4.0), (-4.0, 4.0)])),
                    Adam::default_callbacks().with_terminator(MaxSteps(1_000_000)),
                )
                .unwrap();
            assert!(result.message.success());
            assert_relative_eq!(result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        }
    }

    #[test]
    fn backend_adam_optimizes_f32_quadratic() {
        let mut solver = BackendAdam::<f32>::default();
        let config = BackendAdamConfig {
            alpha: 0.05,
            ema_tolerance: 1e-7,
            patience: 20,
            ..BackendAdamConfig::default()
        };
        let result = solver
            .process(
                &GenericQuadratic,
                &(),
                Vector::from_vec(vec![3.0, -2.0]),
                config,
                Callbacks::empty().with_terminator(MaxSteps(20_000)),
            )
            .unwrap();
        assert!(result.fx < 1e-5);
        assert!(result.evals.f() > 0);
        assert!(result.evals.g() > 0);
    }
}
