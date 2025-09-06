use crate::{
    algorithms::gradient::GradientStatus,
    core::{utils::maybe_warn, Bounds, Callbacks, MinimizationSummary},
    traits::{Algorithm, Boundable, Bounded, Gradient, Terminator},
    DMatrix, DVector, Float,
};
use std::ops::ControlFlow;

/// A [`Terminator`] which terminates the [`Adam`] algorithm if the Exponential Moving Average (EMA)
/// loss does not improve in the number of steps defined by the [`AdamConfig`] `patience`
/// parameter.
pub struct AdamEMATerminator;
impl<P, U, E> Terminator<Adam, P, GradientStatus, U, E> for AdamEMATerminator
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
    ) -> ControlFlow<()> {
        let prev_ema_loss = algorithm.ema_loss;
        algorithm.ema_loss = algorithm
            .config
            .beta_c
            .mul_add(prev_ema_loss, (1.0 - algorithm.config.beta_c) * algorithm.f);
        if (algorithm.ema_loss - prev_ema_loss).abs() < algorithm.config.eps_loss {
            algorithm.ema_counter += 1;
        } else {
            algorithm.ema_counter = 0;
        }
        if algorithm.ema_counter >= algorithm.config.patience {
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
    bounds: Option<Bounds>,
    alpha: Float,
    beta_1: Float,
    beta_2: Float,
    epsilon: Float,
    beta_c: Float,
    eps_loss: Float,
    patience: usize,
}
impl AdamConfig {
    /// Set the starting position of the algorithm.
    pub fn with_x0<I: IntoIterator<Item = Float>>(mut self, x0: I) -> Self {
        let x0 = x0.into_iter().collect::<Vec<Float>>();
        self.x0 = DVector::from_column_slice(&x0);
        self
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
    /// Set the value for the slope of the exponential moving average loss decay (default = `0.9`).
    ///
    /// This value can be tuned to specify convergence behavior.
    pub const fn with_beta_c(mut self, value: Float) -> Self {
        self.beta_c = value;
        self
    }
    /// Set the number of allowed iterations with no improvement in the loss (according to an
    /// exponential moving average) before the algorithm terminates (default = `1`).
    pub const fn with_patience(mut self, value: usize) -> Self {
        self.patience = value;
        self
    }
}
impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            x0: DVector::zeros(0),
            bounds: None,
            alpha: 0.001,
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8, // NOTE: I think this can be independent of bit precision
            beta_c: 0.9,
            eps_loss: Float::EPSILON.sqrt(),
            patience: 1,
        }
    }
}
impl Bounded for AdamConfig {
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds> {
        &mut self.bounds
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
    config: AdamConfig,
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
        config: Self::Config,
        problem: &mut P,
        status: &mut GradientStatus,
        args: &U,
    ) -> Result<(), E> {
        self.config = config;
        if self.config.bounds.is_some() {
            maybe_warn("The Adam optimizer has experimental support for bounded parameters, but it may be unstable and fail to converge!");
        }
        let bounds = self.config.bounds.as_ref();
        self.x = self.config.x0.unconstrain_from(bounds);
        self.g = DVector::zeros(self.x.len());
        self.f = problem.evaluate(&self.x.constrain_to(bounds), args)?;
        status.with_position((self.x.constrain_to(bounds), self.f));
        status.inc_n_f_evals();
        self.m = DVector::zeros(self.x.len());
        self.v = DVector::zeros(self.x.len());
        Ok(())
    }

    fn step(
        &mut self,
        i_step: usize,
        problem: &mut P,
        status: &mut GradientStatus,
        args: &U,
    ) -> Result<(), E> {
        let bounds = self.config.bounds.as_ref();
        self.g = problem.gradient(&self.x.constrain_to(bounds), args)?;
        status.inc_n_g_evals();
        self.m = self.m.scale(self.config.beta_1) + self.g.scale(1.0 - self.config.beta_1);
        self.v = self.v.scale(self.config.beta_2)
            + self.g.map(|gi| gi.powi(2)).scale(1.0 - self.config.beta_2);
        let alpha_t = self.config.alpha * (1.0 - self.config.beta_2.powi(i_step as i32 + 1)).sqrt()
            / (1.0 - self.config.beta_1.powi(i_step as i32 + 1));
        self.x -= self
            .m
            .scale(alpha_t)
            .component_div(&self.v.map(|vi| vi.sqrt() + self.config.epsilon));
        self.f = problem.evaluate(&self.x.constrain_to(bounds), args)?;
        status.inc_n_f_evals();
        status.with_position((self.x.constrain_to(bounds), self.f));
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &GradientStatus,
        _args: &U,
    ) -> Result<Self::Summary, E> {
        Ok(MinimizationSummary {
            x0: self.config.x0.clone(),
            x: status.x.clone(),
            fx: status.fx,
            bounds: self.config.bounds.clone(),
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

    fn default_callbacks() -> Callbacks<Self, P, GradientStatus, U, E>
    where
        Self: Sized,
    {
        Callbacks::empty().with_terminator(AdamEMATerminator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::MaxSteps, test_functions::Rosenbrock};
    use approx::assert_relative_eq;

    #[test]
    fn test_adam() {
        let mut solver = Adam::default();
        let mut problem = Rosenbrock { n: 2 };
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
                    &mut problem,
                    &mut (),
                    AdamConfig::default().with_x0(starting_value),
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
        let mut problem = Rosenbrock { n: 2 };
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
                    &mut problem,
                    &mut (),
                    AdamConfig::default()
                        .with_x0(starting_value)
                        .with_bounds([(-4.0, 4.0), (-4.0, 4.0)]),
                    Adam::default_callbacks().with_terminator(MaxSteps(1_000_000)),
                )
                .unwrap();
            assert!(result.converged);
            assert_relative_eq!(result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        }
    }
}
