use crate::{
    algorithms::{gradient::GradientStatus, line_search::StrongWolfeLineSearch},
    core::{Callbacks, MaxSteps, MinimizationSummary},
    error::{GaneshError, GaneshResult},
    traits::{
        linesearch::LineSearchOutput, Algorithm, Gradient, LineSearch, Status,
        SupportsParameterNames, SupportsTransform, Terminator, Transform, TransformedProblem,
    },
    DMatrix, DVector, Float,
};
use std::ops::ControlFlow;

/// A [`Terminator`] which stops [`ConjugateGradient`] once the gradient norm is sufficiently small.
#[derive(Clone)]
pub struct ConjugateGradientGTerminator {
    /// The absolute gradient-norm tolerance.
    pub eps_abs: Float,
}

impl Default for ConjugateGradientGTerminator {
    fn default() -> Self {
        Self {
            eps_abs: Float::cbrt(Float::EPSILON),
        }
    }
}

impl ConjugateGradientGTerminator {
    /// Generate a new [`ConjugateGradientGTerminator`] with a given absolute tolerance.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `eps_abs` is not strictly positive.
    pub fn new(eps_abs: Float) -> GaneshResult<Self> {
        if eps_abs <= 0.0 {
            return Err(GaneshError::ConfigError(
                "eps_abs must be greater than 0".to_string(),
            ));
        }
        Ok(Self { eps_abs })
    }
}

impl<P, U, E> Terminator<ConjugateGradient, P, GradientStatus, U, E, ConjugateGradientConfig>
    for ConjugateGradientGTerminator
where
    P: Gradient<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut ConjugateGradient,
        _problem: &P,
        status: &mut GradientStatus,
        _args: &U,
        _config: &ConjugateGradientConfig,
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

/// The update formula used to compute the nonlinear conjugate-gradient coefficient `beta_k`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ConjugateGradientUpdate {
    /// Fletcher-Reeves update.
    ///
    /// [^1]: [Fletcher, R., & Reeves, C. M. (1964). Function minimization by conjugate gradients. The Computer Journal, 7(2), 149-154.](https://doi.org/10.1093/comjnl/7.2.149)
    FletcherReeves,
    /// Polak-Ribiere update clipped at zero.
    ///
    /// [^1]: [Polak, E., & Ribiere, G. (1969). Note sur la convergence de méthodes de directions conjuguées. Revue française d’informatique et de recherche opérationnelle. Série rouge, 3(16), 35-43.](https://www.numdam.org/item/RO_1969__3_16_35_0/)
    #[default]
    PolakRibierePlus,
    /// Hestenes-Stiefel update clipped at zero.
    ///
    /// [^1]: [Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients for solving linear systems. Journal of Research of the National Bureau of Standards, 49(6), 409-436.](https://doi.org/10.6028/jres.049.044)
    HestenesStiefelPlus,
    /// Dai-Yuan update.
    ///
    /// [^1]: [Dai, Y.-H., & Yuan, Y. (1999). A nonlinear conjugate gradient method with a strong global convergence property. SIAM Journal on Optimization, 10(1), 177-182.](https://doi.org/10.1137/S1052623497318992)
    DaiYuan,
    /// Hager-Zhang update.
    ///
    /// [^1]: [Hager, W. W., & Zhang, H. (2005). A new conjugate gradient method with guaranteed descent and an efficient line search. SIAM Journal on Optimization, 16(1), 170-192.](https://doi.org/10.1137/030601880)
    HagerZhang,
}

/// Configuration for the [`ConjugateGradient`] algorithm.
#[derive(Clone, Default)]
pub struct ConjugateGradientConfig {
    parameter_names: Option<Vec<String>>,
    transform: Option<Box<dyn Transform>>,
    line_search: StrongWolfeLineSearch,
    update: ConjugateGradientUpdate,
}

impl ConjugateGradientConfig {
    /// Create a new configuration with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the line search algorithm to use.
    pub const fn with_line_search(mut self, line_search: StrongWolfeLineSearch) -> Self {
        self.line_search = line_search;
        self
    }

    /// Set the update formula used for the conjugate-gradient coefficient.
    pub const fn with_update(mut self, update: ConjugateGradientUpdate) -> Self {
        self.update = update;
        self
    }
}

impl SupportsTransform for ConjugateGradientConfig {
    fn get_transform_mut(&mut self) -> &mut Option<Box<dyn Transform>> {
        &mut self.transform
    }
}

impl SupportsParameterNames for ConjugateGradientConfig {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

/// Nonlinear Conjugate Gradient optimizer for unconstrained smooth minimization.
///
/// This implementation uses a strong-Wolfe line search and supports several standard formulas for
/// the conjugate-gradient coefficient through [`ConjugateGradientUpdate`].
#[derive(Clone)]
pub struct ConjugateGradient {
    x: DVector<Float>,
    f: Float,
    g: DVector<Float>,
    g_previous: DVector<Float>,
    d: DVector<Float>,
    line_search: StrongWolfeLineSearch,
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        Self {
            x: DVector::zeros(0),
            f: Float::INFINITY,
            g: DVector::zeros(0),
            g_previous: DVector::zeros(0),
            d: DVector::zeros(0),
            line_search: StrongWolfeLineSearch::default(),
        }
    }
}

impl ConjugateGradient {
    fn beta(&self, update: ConjugateGradientUpdate, g_next: &DVector<Float>) -> Float {
        let y = g_next - &self.g;
        let g_norm_sq = self.g.dot(&self.g);
        let g_next_norm_sq = g_next.dot(g_next);
        let d_y = self.d.dot(&y);

        let beta = match update {
            ConjugateGradientUpdate::FletcherReeves => {
                if g_norm_sq <= Float::EPSILON {
                    0.0
                } else {
                    g_next_norm_sq / g_norm_sq
                }
            }
            ConjugateGradientUpdate::PolakRibierePlus => {
                if g_norm_sq <= Float::EPSILON {
                    0.0
                } else {
                    (g_next.dot(&y) / g_norm_sq).max(0.0)
                }
            }
            ConjugateGradientUpdate::HestenesStiefelPlus => {
                if d_y.abs() <= Float::EPSILON {
                    0.0
                } else {
                    (g_next.dot(&y) / d_y).max(0.0)
                }
            }
            ConjugateGradientUpdate::DaiYuan => {
                if d_y.abs() <= Float::EPSILON {
                    0.0
                } else {
                    g_next_norm_sq / d_y
                }
            }
            ConjugateGradientUpdate::HagerZhang => {
                if d_y.abs() <= Float::EPSILON {
                    0.0
                } else {
                    let y_norm_sq = y.dot(&y);
                    let correction = &y - self.d.scale(2.0 * y_norm_sq / d_y);
                    correction.dot(g_next) / d_y
                }
            }
        };

        if beta.is_finite() {
            beta
        } else {
            0.0
        }
    }
}

impl<P, U, E> Algorithm<P, GradientStatus, U, E> for ConjugateGradient
where
    P: Gradient<U, E>,
{
    type Summary = MinimizationSummary;
    type Config = ConjugateGradientConfig;
    type Init = DVector<Float>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut GradientStatus,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let t_problem = TransformedProblem::new(problem, &config.transform);
        self.x = t_problem.to_owned_internal(init);
        let (f, g) = t_problem.evaluate_with_gradient(&self.x, args)?;
        self.f = f;
        self.g = g;
        self.g_previous = self.g.clone();
        self.d = -&self.g;
        self.line_search = config.line_search.clone();
        status.n_f_evals += 1;
        status.n_g_evals += 1;
        status.initialize((init.clone(), self.f));
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
        if self.d.dot(&self.g) >= 0.0 {
            self.d = -&self.g;
        }

        if let Ok(LineSearchOutput { alpha, fx, g }) = self
            .line_search
            .search(&self.x, &self.d, None, &t_problem, None, args, status)?
        {
            self.x += self.d.scale(alpha);
            self.f = fx;
            self.g_previous = self.g.clone();
            let beta = self.beta(config.update, &g);
            self.g = g;
            self.d = -&self.g + self.d.scale(beta);
            if self.d.dot(&self.g) >= 0.0 {
                self.d = -&self.g;
            }
            status.set_position((t_problem.to_owned_external(&self.x), self.f));
        } else {
            self.d = -&self.g;
        }
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &GradientStatus,
        _args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        Ok(MinimizationSummary {
            x0: init.clone(),
            x: status.x.clone(),
            fx: status.fx,
            bounds: None,
            n_f_evals: status.n_f_evals,
            n_g_evals: status.n_g_evals,
            n_h_evals: status.n_h_evals,
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
            .with_terminator(ConjugateGradientGTerminator::default())
            .with_terminator(MaxSteps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algorithms::line_search::StrongWolfeLineSearch,
        core::{Bounds, MaxSteps},
        test_functions::Rosenbrock,
    };
    use approx::assert_relative_eq;
    use nalgebra::dvector;

    #[test]
    fn test_conjugate_gradient_polak_ribiere_plus() {
        let problem = Rosenbrock { n: 2 };
        let mut solver = ConjugateGradient::default();
        let result = solver
            .process(
                &problem,
                &(),
                dvector![-1.2, 1.0],
                ConjugateGradientConfig::default()
                    .with_update(ConjugateGradientUpdate::PolakRibierePlus),
                ConjugateGradient::default_callbacks().with_terminator(MaxSteps(1_000)),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 1e-10);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(result.x[1], 1.0, epsilon = 2e-5);
    }

    #[test]
    fn test_conjugate_gradient_hager_zhang() {
        let problem = Rosenbrock { n: 2 };
        let mut solver = ConjugateGradient::default();
        let result = solver
            .process(
                &problem,
                &(),
                dvector![-1.2, 1.0],
                ConjugateGradientConfig::default()
                    .with_update(ConjugateGradientUpdate::HagerZhang)
                    .with_line_search(StrongWolfeLineSearch::default()),
                ConjugateGradient::default_callbacks().with_terminator(MaxSteps(1_000)),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 1e-10);
    }

    #[test]
    fn test_conjugate_gradient_fletcher_reeves_with_transform() {
        let problem = Rosenbrock { n: 2 };
        let mut solver = ConjugateGradient::default();
        let result = solver
            .process(
                &problem,
                &(),
                dvector![-1.2, 1.0],
                ConjugateGradientConfig::default()
                    .with_update(ConjugateGradientUpdate::FletcherReeves)
                    .with_transform(&Bounds::from([(-2.0, 2.0), (-1.0, 3.0)])),
                ConjugateGradient::default_callbacks().with_terminator(MaxSteps(1_000)),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 1e-10);
        assert!(result.x[0] >= -2.0 && result.x[0] <= 2.0);
        assert!(result.x[1] >= -1.0 && result.x[1] <= 3.0);
    }
}
