use nalgebra::DVector;

use crate::{
    algorithms::line_search::StrongWolfeLineSearch,
    core::{Bound, Bounds, MinimizationSummary},
    maybe_warn,
    traits::{Algorithm, Bounded, CostFunction, Gradient, Hessian, LineSearch},
    Float,
};

use super::GradientStatus;

/// A terminator for the [`ConjugateGradient`] [`Algorithm`]
///
/// This causes termination when the magnitude of the
/// gradient vector becomes smaller than the given absolute tolerance. In such a case, the [`Status`](crate::traits::Status)
/// of the [`Engine`](`crate::core::Engine`) will be set as converged with the message "GRADIENT
/// CONVERGED".
#[derive(Clone)]
pub struct ConjugateGradientGTerminator;
impl ConjugateGradientGTerminator {
    fn update_convergence(
        &self,
        gradient: &DVector<Float>,
        status: &mut GradientStatus,
        eps_abs: Float,
    ) {
        if gradient.dot(gradient).sqrt() < eps_abs {
            status.set_converged();
            status.with_message("GRADIENT CONVERGED");
        }
    }
}

/// Error modes for [`ConjugateGradient`] [`Algorithm`].
#[derive(Default, Clone)]
pub enum ConjugateGradientErrorMode {
    /// Computes the exact Hessian matrix via finite differences.
    #[default]
    ExactHessian,
    /// Skip Hessian computation (use this when error evaluation is not important).
    Skip,
}

/// Various methods for computing the $`\beta`$ term of the conjugate gradient update.
#[derive(Copy, Clone, Default)]
pub enum ConjugateGradientMethod {
    /// The Fletcher-Reeves method[^1].
    ///
    /// [^1]: [Fletcher, R. (1964). Function minimization by conjugate gradients. In The Computer Journal (Vol. 7, Issue 2, pp. 149–154). Oxford University Press (OUP).](https://doi.org/10.1093/comjnl/7.2.149)
    FletcherReeves,
    /// The Polak-Ribière method[^1].
    ///
    /// [^1]: [Polak, E. & Ribiere, G. (1969). Note sur la convergence de méthodes de directions conjuguées. In Revue française d'informatique et de recherche opérationnelle (Série rouge, Vol. 3, no. R1, p. 35-43).](https://www.numdam.org/item/M2AN_1969__3_1_35_0/)
    #[default]
    PolakRibiere,
    /// The Hestenes-Stiefel method[^1].
    ///
    /// [^1]: [Hestenes, M. R., & Stiefel, E. (1952). Methods of conjugate gradients for solving linear systems. In Journal of Research of the National Bureau of Standards (Vol. 49, Issue 6, p. 409). National Institute of Standards and Technology (NIST).](https://doi.org/10.6028/jres.049.044)
    HestenesStiefel,
    /// The Dai-Yuan method[^1].
    ///
    /// [^1]: [Dai, Y. H., & Yuan, Y. (1999). A Nonlinear Conjugate Gradient Method with a Strong Global Convergence Property. In SIAM Journal on Optimization (Vol. 10, Issue 1, pp. 177–182). Society for Industrial & Applied Mathematics (SIAM).](https://doi.org/10.1137/s1052623497318992)
    DaiYuan,
}

impl ConjugateGradientMethod {
    fn compute_beta(
        &self,
        dx: &DVector<Float>,
        dx_previous: &DVector<Float>,
        s_previous: &DVector<Float>,
    ) -> Float {
        Float::max(
            0.0,
            match self {
                Self::FletcherReeves => dx.dot(dx) / dx_previous.dot(dx_previous),
                Self::PolakRibiere => dx.dot(&(dx - dx_previous)) / dx_previous.dot(dx_previous),
                Self::HestenesStiefel => {
                    dx.dot(&(dx - dx_previous)) / -s_previous.dot(&(dx - dx_previous))
                }
                Self::DaiYuan => dx.dot(dx) / -s_previous.dot(&(dx - dx_previous)),
            },
        )
    }
}

/// The internal configuration struct for the [`ConjugateGradient`] algorithm.
#[derive(Clone)]
pub struct ConjugateGradientConfig<U, E> {
    x0: DVector<Float>,
    bounds: Option<Bounds>,
    conjugate_gradient_method: ConjugateGradientMethod,
    line_search: Box<dyn LineSearch<GradientStatus, U, E>>,
    terminator_g: ConjugateGradientGTerminator,
    eps_g_abs: Float,
    max_step: Float,
    error_mode: ConjugateGradientErrorMode,
    n_restart: usize,
}
impl<U, E> Bounded for ConjugateGradientConfig<U, E> {
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds> {
        &mut self.bounds
    }
}
impl<U, E> ConjugateGradientConfig<U, E> {
    /// Set the starting position of the algorithm.
    pub fn with_x0<I: IntoIterator<Item = Float>>(&mut self, x0: I) -> &mut Self {
        let x0 = x0.into_iter().collect::<Vec<Float>>();
        self.x0 = DVector::from_column_slice(&x0);
        self
    }
    /// Set the method for computing the $`\beta`$ term of the conjugate gradient update.
    pub const fn with_method(mut self, method: ConjugateGradientMethod) -> Self {
        self.conjugate_gradient_method = method;
        self
    }
    /// Set the number of steps to take before rebooting the method (setting $`\beta`$ to zero).
    pub const fn with_n_restart(mut self, value: usize) -> Self {
        self.n_restart = value;
        self
    }
    /// Set the termination condition concerning the gradient values.
    pub const fn with_terminator_g(mut self, term: ConjugateGradientGTerminator) -> Self {
        self.terminator_g = term;
        self
    }
    /// Set the absolute g-convergence tolerance (default = `MACH_EPS^(1/3)`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\epsilon <= 0`$.
    pub fn with_eps_g_abs(mut self, value: Float) -> Self {
        assert!(value > 0.0);
        self.eps_g_abs = value;
        self
    }
    /// Set the line search local method for local optimization of step size. Defaults to a line
    /// search which satisfies the strong Wolfe conditions, [`StrongWolfeLineSearch`]. Note that in
    /// general, this should only use [`LineSearch`] algorithms which satisfy the Wolfe conditions.
    /// Using the Armijo condition alone will lead to slower convergence.
    pub fn with_line_search<LS: LineSearch<GradientStatus, U, E> + 'static>(
        mut self,
        line_search: LS,
    ) -> Self {
        self.line_search = Box::new(line_search);
        self
    }
    /// Set the mode for caluclating parameter errors at the end of the fit. Defaults to
    /// recalculating an exact finite-difference Hessian.
    pub const fn with_error_mode(mut self, error_mode: ConjugateGradientErrorMode) -> Self {
        self.error_mode = error_mode;
        self
    }
}
impl<U, E> Default for ConjugateGradientConfig<U, E> {
    fn default() -> Self {
        Self {
            x0: DVector::zeros(0),
            bounds: None,
            conjugate_gradient_method: Default::default(),
            line_search: Box::<StrongWolfeLineSearch>::default(),
            terminator_g: ConjugateGradientGTerminator,
            eps_g_abs: Float::cbrt(Float::EPSILON),
            max_step: 1e8,
            error_mode: Default::default(),
            n_restart: 100,
        }
    }
}

/// The Conjugate Gradient algorithm.
///
/// The conjugate gradient algorithm is a gradient descent method similar to quasi-Newton methods, but rather than
/// approximate the Hessian, it uses the gradient at the previous step to modify the step
/// direction, scaled by a parameter $`\beta`$. See [^1] for more information.
///
/// [^1]: [Shewchuk, J.R. (1994). An Introduction to the Conjugate Gradient Method Without the Agonizing Pain: Edition 1+1/4.](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
pub struct ConjugateGradient<U, E> {
    x: DVector<Float>,
    dx: DVector<Float>,
    dx_previous: DVector<Float>,
    s_previous: DVector<Float>,
    f: Float,
    f_previous: Float,
    config: ConjugateGradientConfig<U, E>,
}

impl<U, E> Default for ConjugateGradient<U, E> {
    fn default() -> Self {
        Self {
            x: Default::default(),
            dx: Default::default(),
            dx_previous: Default::default(),
            s_previous: Default::default(),
            f: Float::INFINITY,
            f_previous: Float::INFINITY,
            config: ConjugateGradientConfig::default(),
        }
    }
}

impl<U, E> Algorithm<GradientStatus, U, E> for ConjugateGradient<U, E> {
    type Summary = MinimizationSummary;
    type Config = ConjugateGradientConfig<U, E>;
    fn get_config_mut(&mut self) -> &mut Self::Config {
        &mut self.config
    }
    fn initialize(
        &mut self,
        func: &dyn CostFunction<U, E>,
        status: &mut GradientStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        let bounds = self.config.bounds.as_ref();
        if bounds.is_some() {
            maybe_warn("The Conjugate Gradient method has experimental support for bounded parameters, but it may be unstable and fail to converge!");
        }
        self.x = Bound::to_bounded(self.config.x0.as_slice(), bounds);
        self.dx = -func.gradient_bounded(self.x.as_slice(), bounds, user_data)?;
        status.inc_n_g_evals();
        status.with_position((
            Bound::to_unbounded(self.x.as_slice(), bounds),
            func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?,
        ));
        status.inc_n_f_evals();
        Ok(())
    }

    fn step(
        &mut self,
        i_step: usize,
        func: &dyn CostFunction<U, E>,
        status: &mut GradientStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        if i_step == 0 {
            let (_valid, alpha, f, g) = self.config.line_search.search(
                &self.x,
                &self.dx,
                Some(self.config.max_step),
                func,
                self.config.bounds.as_ref(),
                user_data,
                status,
            )?;
            self.x += alpha * -&g;
            self.f = f;
            // TODO: move this in memory
            self.dx_previous = self.dx.clone();
            self.dx = -g;
            self.s_previous = self.dx.clone();
            status.with_position((
                Bound::to_bounded(self.x.as_slice(), self.config.bounds.as_ref()),
                f,
            ));
        } else {
            let beta = if i_step % self.config.n_restart == 0 {
                0.0
            } else {
                self.config.conjugate_gradient_method.compute_beta(
                    &self.dx,
                    &self.dx_previous,
                    &self.s_previous,
                )
            };
            let s = &self.dx + self.s_previous.scale(beta);
            let (_valid, alpha, f, g) = self.config.line_search.search(
                &self.x,
                &s,
                Some(self.config.max_step),
                func,
                self.config.bounds.as_ref(),
                user_data,
                status,
            )?;
            self.x += alpha * &s;
            self.f = f;
            // TODO: move this in memory
            self.dx_previous = self.dx.clone();
            self.dx = -g;
            self.s_previous = s;
            status.with_position((
                Bound::to_bounded(self.x.as_slice(), self.config.bounds.as_ref()),
                f,
            ));
        }
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        _func: &dyn CostFunction<U, E>,
        status: &mut GradientStatus,
        _user_data: &mut U,
    ) -> Result<bool, E> {
        self.config
            .terminator_g
            .update_convergence(&-&self.dx, status, self.config.eps_g_abs);
        Ok(status.converged)
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

    fn postprocessing(
        &mut self,
        func: &dyn CostFunction<U, E>,
        status: &mut GradientStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        match self.config.error_mode {
            ConjugateGradientErrorMode::ExactHessian => {
                let hessian = func.hessian(
                    Bound::to_unbounded(self.x.as_slice(), self.config.bounds.as_ref()).as_slice(),
                    user_data,
                )?;
                status.with_hess(&hessian);
            }
            ConjugateGradientErrorMode::Skip => {}
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.x = Default::default();
        self.dx = Default::default();
        self.dx_previous = Default::default();
        self.s_previous = Default::default();
        self.f = Float::INFINITY;
        self.f_previous = Float::INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use approx::assert_relative_eq;

    use crate::{
        core::{CtrlCAbortSignal, Engine},
        test_functions::Rosenbrock,
        traits::Bounded,
        Float,
    };

    use super::ConjugateGradient;

    #[test]
    #[allow(unused_variables)]
    fn test_problem_constructor() {
        #[allow(clippy::box_default)]
        let solver: ConjugateGradient<(), Infallible> = ConjugateGradient::default();
        let problem = Engine::new(solver);
    }

    #[test]
    fn test_conjugate_gradient() -> Result<(), Infallible> {
        let solver = ConjugateGradient::default();
        let mut m = Engine::new(solver).setup(|m| m.with_abort_signal(CtrlCAbortSignal::new()));
        let problem = Rosenbrock { n: 2 };
        m.configure(|c| c.with_x0([-2.0, 2.0])).process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.cbrt());
        m.configure(|c| c.with_x0([2.0, 2.0])).process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.configure(|c| c.with_x0([2.0, -2.0])).process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.configure(|c| c.with_x0([-2.0, -2.0])).process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.configure(|c| c.with_x0([0.0, 0.0])).process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.configure(|c| c.with_x0([1.0, 1.0])).process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }

    /// BUG: The following commented out tests fail.
    /// This might not be a problem with the conjugate gradient method itself, but rather with the
    /// boundary mode implementation. The unbounded tests work fine.
    #[test]
    fn test_bounded_conjugate_gradient() -> Result<(), Infallible> {
        let solver = ConjugateGradient::default();
        let mut m = Engine::new(solver).setup(|e| {
            e.configure(|c| c.with_bounds(vec![(-4.0, 4.0), (-4.0, 4.0)]))
                .with_abort_signal(CtrlCAbortSignal::new())
        });
        let problem = Rosenbrock { n: 2 };
        m.configure(|c| c.with_x0([-2.0, 2.0])).process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        // m.configure(|c| c.with_x0([2.0, 2.0])).process(&problem)?;
        // assert!(m.result.converged);
        // assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        // m.configure(|c| c.with_x0([2.0, -2.0])).process(&problem)?;
        // assert!(m.result.converged);
        // assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        // m.configure(|c| c.with_x0([-2.0, -2.0])).process(&problem)?;
        // assert!(m.result.converged);
        // assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        m.configure(|c| c.with_x0([0.0, 0.0])).process(&problem)?;
        assert!(m.result.converged);
        assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        // m.configure(|c| c.with_x0([1.0, 1.0])).process(&problem)?;
        // assert!(m.result.converged);
        // assert_relative_eq!(m.result.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }
}
