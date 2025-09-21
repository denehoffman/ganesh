use crate::{
    algorithms::gradient::GradientStatus,
    core::{Callbacks, MinimizationSummary},
    traits::{
        Algorithm, CostFunction, Gradient, SupportsTransform, Terminator, Transform,
        TransformedProblem,
    },
    DMatrix, DVector, Float,
};
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
impl<P, U, E> Terminator<Adam, P, GradientStatus, U, E, AdamConfig> for AdamEMATerminator
where
    P: Gradient<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut Adam,
        _problem: &P,
        status: &mut GradientStatus,
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
            status.set_converged();
            status.with_message(&format!(
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
    x0: DVector<Float>,
    transform: Option<Box<dyn Transform>>,
    alpha: Float,
    beta_1: Float,
    beta_2: Float,
    epsilon: Float,
}
impl AdamConfig {
    /// Create a new configuration by setting the starting position of the algorithm.
    pub fn new<I>(x0: I) -> Self
    where
        I: AsRef<[Float]>,
    {
        Self {
            x0: DVector::from_row_slice(x0.as_ref()),
            transform: None,
            alpha: 0.001,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8, // NOTE: I think this can be independent of bit precision
        }
    }
    /// Set the initial learning rate $`\alpha`$ (default = `0.001`).
    pub const fn with_alpha(mut self, value: Float) -> Self {
        self.alpha = value;
        self
    }
    /// Set the value for the hyperparameter $`\beta_1`$ (default = `0.9`).
    ///
    /// This represents the exponential decay rate of the first moment estimate, $`m`$.
    pub const fn with_beta_1(mut self, value: Float) -> Self {
        self.beta_1 = value;
        self
    }
    /// Set the value for the hyperparameter $`\beta_2`$ (default = `0.999`).
    ///
    /// This represents the exponential decay rate of the second moment estimate, $`v`$.
    pub const fn with_beta_2(mut self, value: Float) -> Self {
        self.beta_2 = value;
        self
    }
    /// Set the value for the divide-by-zero tolerance in the update step (default = `1e-8`).
    ///
    /// This ensures the update does not divide by zero if the bias-corrected second raw moment
    /// estimate is zero for any parameter.
    pub const fn with_epsilon(mut self, value: Float) -> Self {
        self.epsilon = value;
        self
    }
}
impl SupportsTransform for AdamConfig {
    fn get_transform_mut(&mut self) -> &mut Option<Box<dyn Transform>> {
        &mut self.transform
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
impl<P, U, E> Algorithm<P, GradientStatus, U, E> for Adam
where
    P: Gradient<U, E>,
{
    type Summary = MinimizationSummary;
    type Config = AdamConfig;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut GradientStatus,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let t_problem = TransformedProblem::new(problem, &config.transform);
        self.x = t_problem.to_owned_internal(&config.x0);
        self.g = DVector::zeros(self.x.len());
        self.f = t_problem.evaluate(&config.x0, args)?;
        status.with_position((config.x0.clone(), self.f));
        status.inc_n_f_evals();
        self.m = DVector::zeros(self.x.len());
        self.v = DVector::zeros(self.x.len());
        Ok(())
    }

    fn step(
        &mut self,
        i_step: usize,
        problem: &P,
        status: &mut GradientStatus,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let t_problem = TransformedProblem::new(problem, &config.transform);
        self.g = t_problem.gradient(&self.x, args)?;
        status.inc_n_g_evals();
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
        status.inc_n_f_evals();
        status.with_position((t_problem.to_owned_external(&self.x), self.f));
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
            converged: status.converged,
            cost_evals: status.n_f_evals,
            gradient_evals: status.n_g_evals,
            message: status.message.clone(),
            parameter_names: None,
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

    fn default_callbacks() -> Callbacks<Self, P, GradientStatus, U, E, Self::Config>
    where
        Self: Sized,
    {
        Callbacks::empty().with_terminator(AdamEMATerminator::default())
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
                    AdamConfig::new(starting_value),
                    Adam::default_callbacks().with_terminator(MaxSteps(1_000_000)),
                )
                .unwrap();
            assert!(result.converged);
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
                    AdamConfig::new(starting_value)
                        .with_transform(&Bounds::from([(-4.0, 4.0), (-4.0, 4.0)])),
                    Adam::default_callbacks().with_terminator(MaxSteps(1_000_000)),
                )
                .unwrap();
            assert!(result.converged);
            assert_relative_eq!(result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        }
    }
}
