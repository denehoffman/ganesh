use nalgebra::DVector;

use crate::{
    algorithms::gradient::GradientStatus,
    core::{bound::Bounded, Bound, Bounds, MinimizationSummary},
    maybe_warn,
    traits::{Algorithm, Configurable, CostFunction, Gradient},
    Float,
};

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
    pub fn with_x0<I: IntoIterator<Item = Float>>(&mut self, x0: I) -> &mut Self {
        let x0 = x0.into_iter().collect::<Vec<Float>>();
        self.x0 = DVector::from_column_slice(&x0);
        self
    }
    /// Set the initial learning rate $`\alpha`$ (default = `0.001`).
    pub const fn with_alpha(&mut self, value: Float) -> &mut Self {
        self.alpha = value;
        self
    }
    /// Set the value for the hyperparameter $`\beta_1`$ (default = `0.9`).
    ///
    /// This represents the exponential decay rate of the first moment estimate, $`m`$.
    pub const fn with_beta_1(&mut self, value: Float) -> &mut Self {
        self.beta_1 = value;
        self
    }
    /// Set the value for the hyperparameter $`\beta_2`$ (default = `0.999`).
    ///
    /// This represents the exponential decay rate of the second moment estimate, $`v`$.
    pub const fn with_beta_2(&mut self, value: Float) -> &mut Self {
        self.beta_2 = value;
        self
    }
    /// Set the value for the divide-by-zero tolerance in the update step (default = `1e-8`).
    ///
    /// This ensures the update does not divide by zero if the bias-corrected second raw moment
    /// estimate is zero for any parameter.
    pub const fn with_epsilon(&mut self, value: Float) -> &mut Self {
        self.epsilon = value;
        self
    }
    /// Set the value for the slope of the exponential moving average loss decay (default = `0.9`).
    ///
    /// This value can be tuned to specify convergence behavior.
    pub const fn with_beta_c(&mut self, value: Float) -> &mut Self {
        self.beta_c = value;
        self
    }
    /// Set the number of allowed iterations with no improvement in the loss (according to an
    /// exponential moving average) before the algorithm terminates (default = `1`).
    pub const fn with_patience(&mut self, value: usize) -> &mut Self {
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
/// [^1] [D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” 2014, arXiv. doi: 10.48550/ARXIV.1412.6980.](https://doi.org/10.48550/ARXIV.1412.6980)
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
impl Configurable for Adam {
    type Config = AdamConfig;

    fn get_config_mut(&mut self) -> &mut Self::Config {
        &mut self.config
    }
}
impl<U, E> Algorithm<GradientStatus, U, E> for Adam {
    type Summary = MinimizationSummary;

    fn initialize(
        &mut self,
        func: &dyn CostFunction<U, E>,
        status: &mut GradientStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        if self.config.bounds.is_some() {
            maybe_warn("The Adam optimizer has experimental support for bounded parameters, but it may be unstable and fail to converge!");
        }
        let bounds = self.config.bounds.as_ref();
        self.x = Bound::to_bounded(self.config.x0.as_slice(), bounds);
        self.g = DVector::zeros(self.x.len());
        self.f = func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?;
        status.with_position((Bound::to_unbounded(self.x.as_slice(), bounds), self.f));
        status.inc_n_f_evals();
        self.m = DVector::zeros(self.x.len());
        self.v = DVector::zeros(self.x.len());
        Ok(())
    }

    fn step(
        &mut self,
        i_step: usize,
        func: &dyn CostFunction<U, E>,
        status: &mut GradientStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        let bounds = self.config.bounds.as_ref();
        self.g = func.gradient_bounded(self.x.as_slice(), bounds, user_data)?;
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
        self.f = func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?;
        status.inc_n_f_evals();
        status.with_position((Bound::to_bounded(self.x.as_slice(), bounds), self.f));
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        _func: &dyn CostFunction<U, E>,
        status: &mut GradientStatus,
        _user_data: &mut U,
    ) -> Result<bool, E> {
        let prev_ema_loss = self.ema_loss;
        self.ema_loss = self
            .config
            .beta_c
            .mul_add(prev_ema_loss, (1.0 - self.config.beta_c) * self.f);
        if (self.ema_loss - prev_ema_loss).abs() < self.config.eps_loss {
            self.ema_counter += 1;
        } else {
            self.ema_counter = 0;
        }
        if self.ema_counter >= self.config.patience {
            status.set_converged();
            status.with_message(&format!(
                "EMA LOSS HAS NOT IMPROVED IN {} STEPS",
                self.ema_counter
            ));
            return Ok(true);
        }
        Ok(false)
    }

    fn summarize(
        &self,
        _func: &dyn CostFunction<U, E>,
        parameter_names: Option<&Vec<String>>,
        status: &GradientStatus,
        _user_data: &U,
    ) -> Result<Self::Summary, E> {
        let result = MinimizationSummary {
            x0: self.config.x0.iter().cloned().collect(),
            x: status.x.iter().cloned().collect(),
            fx: status.fx,
            bounds: self.config.bounds.clone(),
            converged: status.converged,
            cost_evals: status.n_f_evals,
            gradient_evals: status.n_g_evals,
            message: status.message.clone(),
            parameter_names: parameter_names.as_ref().map(|names| names.to_vec()),
            std: status
                .err
                .as_ref()
                .map(|e| e.iter().cloned().collect())
                .unwrap_or_else(|| vec![0.0; status.x.len()]),
        };
        Ok(result)
    }

    fn reset(&mut self) {
        self.ema_loss = 0.0;
        self.ema_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use approx::assert_relative_eq;

    use crate::{
        core::{Bounded, CtrlCAbortSignal, Engine},
        test_functions::Rosenbrock,
        traits::Configurable,
        Float,
    };

    use super::Adam;

    #[test]
    fn test_adam() -> Result<(), Infallible> {
        let mut m = Engine::new(Adam::default()).setup_engine(|e| {
            e.with_abort_signal(CtrlCAbortSignal::new())
                .with_max_steps(1_000_000)
        });
        let problem = Rosenbrock { n: 2 };
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([-2.0, 2.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([2.0, 2.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([2.0, -2.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([-2.0, -2.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([0.0, 0.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([1.0, 1.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        Ok(())
    }

    #[test]
    fn test_bounded_conjugate_gradient() -> Result<(), Infallible> {
        let mut m = Engine::new(Adam::default()).setup_engine(|e| {
            e.setup_algorithm(|a| a.setup_config(|c| c.with_bounds(vec![(-4.0, 4.0), (-4.0, 4.0)])))
                .with_abort_signal(CtrlCAbortSignal::new())
                .with_max_steps(1_000_000)
        });
        let problem = Rosenbrock { n: 2 };
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([-2.0, 2.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([2.0, 2.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([2.0, -2.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([-2.0, -2.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([0.0, 0.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        m.setup_algorithm(|a| a.setup_config(|c| c.with_x0([1.0, 1.0])))
            .process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        Ok(())
    }
}
