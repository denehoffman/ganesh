use crate::algorithms::gradient::GradientStatus;
use crate::core::{
    Callbacks, LinearAlgebra, Matrix, MinimizationSummary, NalgebraProvider, RealScalar, Vector,
};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{
    Algorithm, CostFunction, Gradient, Status, SupportsParameterNames, Terminator, Transform,
    TransformedProblem,
};
use std::marker::PhantomData;
use std::ops::ControlFlow;

/// Terminates [`Adam`] when its exponential moving-average loss stops improving.
#[derive(Copy, Clone)]
pub struct AdamEMATerminator<T: RealScalar = f64> {
    /// Exponential moving-average decay (default = `0.9`).
    pub beta_c: T,
    /// Minimum EMA change counted as an improvement (default = `T::EPSILON.sqrt()`).
    pub eps_loss: T,
    /// Allowed consecutive iterations without improvement (default = `1`).
    pub patience: usize,
}

impl<T: RealScalar, B: LinearAlgebra<T>> SupportsParameterNames for AdamConfig<T, B> {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

impl<T: RealScalar> Default for AdamEMATerminator<T> {
    fn default() -> Self {
        Self {
            beta_c: T::literal(0.9),
            eps_loss: T::epsilon().sqrt(),
            patience: 1,
        }
    }
}

impl<T, B, P, U, E> Terminator<Adam<T, B>, P, GradientStatus<T, B>, U, E, AdamConfig<T, B>>
    for AdamEMATerminator<T>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: Gradient<T, B, U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut Adam<T, B>,
        _problem: &P,
        status: &mut GradientStatus<T, B>,
        _args: &U,
        _config: &AdamConfig<T, B>,
    ) -> ControlFlow<()> {
        let previous = algorithm.ema_loss;
        algorithm.ema_loss = self.beta_c * previous + (T::one() - self.beta_c) * algorithm.fx;
        if (algorithm.ema_loss - previous).abs() < self.eps_loss {
            algorithm.ema_counter += 1;
        } else {
            algorithm.ema_counter = 0;
        }
        if algorithm.ema_counter >= self.patience {
            status.set_message().succeed_with_message(format!(
                "EMA LOSS HAS NOT IMPROVED IN {} STEPS",
                algorithm.ema_counter
            ));
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Configuration for scalar- and linear-algebra-generic Adam.
pub struct AdamConfig<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Learning rate.
    alpha: T,
    /// First-moment decay.
    beta_1: T,
    /// Second-moment decay.
    beta_2: T,
    /// Denominator stabilizer.
    epsilon: T,
    /// Optional user-facing parameter names copied into summaries.
    parameter_names: Option<Vec<String>>,
    /// Optional coordinate transform.
    transform: Option<Box<dyn Transform<T, B>>>,
}

impl<T, B> Default for AdamConfig<T, B>
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
            parameter_names: None,
            transform: None,
        }
    }
}

impl<T, B> AdamConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Create a configuration with the default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the initial learning rate.
    pub fn with_alpha(mut self, value: T) -> GaneshResult<Self> {
        if value <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Initial learning rate must be positive and greater than 0".to_string(),
            ));
        }
        self.alpha = value;
        Ok(self)
    }

    /// Set the first-moment exponential decay rate.
    pub fn with_beta_1(mut self, value: T) -> GaneshResult<Self> {
        if value < T::zero() || value >= T::one() {
            return Err(GaneshError::ConfigError(
                "beta_1 must be in the range [0, 1)".to_string(),
            ));
        }
        self.beta_1 = value;
        Ok(self)
    }

    /// Set the second-moment exponential decay rate.
    pub fn with_beta_2(mut self, value: T) -> GaneshResult<Self> {
        if value < T::zero() || value >= T::one() {
            return Err(GaneshError::ConfigError(
                "beta_2 must be in the range [0, 1)".to_string(),
            ));
        }
        self.beta_2 = value;
        Ok(self)
    }

    /// Set the divide-by-zero tolerance used in the update.
    pub fn with_epsilon(mut self, value: T) -> GaneshResult<Self> {
        if value <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Divide-by-zero tolerance must be positive and greater than 0".to_string(),
            ));
        }
        self.epsilon = value;
        Ok(self)
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

/// Scalar- and linear-algebra-generic Adam optimizer.
#[derive(Clone, Debug)]
pub struct Adam<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    x: Vector<T, B>,
    fx: T,
    first_moment: Vector<T, B>,
    second_moment: Vector<T, B>,
    ema_loss: T,
    ema_counter: usize,
    _provider: PhantomData<B>,
}

impl<T, B> Default for Adam<T, B>
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
            _provider: PhantomData,
        }
    }
}

impl<T, B, P, U, E> Algorithm<P, GradientStatus<T, B>, U, E> for Adam<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: Gradient<T, B, U, E>,
{
    type Summary = MinimizationSummary<T, B>;
    type Config = AdamConfig<T, B>;
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
        self.fx = transformed.evaluate(&self.x, args)?;
        self.first_moment = Vector::zeros(self.x.len());
        self.second_moment = Vector::zeros(self.x.len());
        // The EMA terminator starts from zero and updates after the first optimization step.
        self.ema_loss = T::zero();
        self.ema_counter = 0;
        status.evals.record_f();
        status.initialize(init.clone(), self.fx);
        Ok(())
    }

    fn step(
        &mut self,
        current_step: usize,
        problem: &P,
        status: &mut GradientStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
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
            std: crate::core::summary::unknown_uncertainties(dimension),
            fx: status.fx,
            evals: status.evals,
            covariance: Matrix::identity(dimension),
        })
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty().with_terminator(AdamEMATerminator::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::MaxSteps, test_functions::Rosenbrock, traits::Bounds};
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
        let mut solver = Adam::<f64>::default();
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
                    Vector::from_vec(starting_value.to_vec()),
                    AdamConfig::<f64>::default(),
                    Adam::default_callbacks().with_terminator(MaxSteps(1_000_000)),
                )
                .unwrap();
            assert!(result.message.success());
            assert!(
                result.fx.abs() <= 2.0 * f64::EPSILON.cbrt(),
                "start = {starting_value:?}, fx = {}, message = {}",
                result.fx,
                result.message
            );
        }
    }

    #[test]
    fn test_bounded_adam() {
        let mut solver = Adam::<f64>::default();
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
                    Vector::from_vec(starting_value.to_vec()),
                    AdamConfig::<f64>::default()
                        .with_transform(Bounds::new([(-4.0, 4.0), (-4.0, 4.0)]).unwrap()),
                    Adam::default_callbacks().with_terminator(MaxSteps(1_000_000)),
                )
                .unwrap();
            assert!(result.message.success());
            assert_relative_eq!(result.fx, 0.0, epsilon = f64::EPSILON.cbrt());
        }
    }

    #[test]
    fn provider_adam_optimizes_f32_quadratic() {
        let mut solver = Adam::<f32>::default();
        let config = AdamConfig::default().with_alpha(0.05).unwrap();
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
