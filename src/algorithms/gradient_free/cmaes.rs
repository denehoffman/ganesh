//! Scalar- and linear-algebra-generic full-covariance CMA-ES.

use crate::algorithms::gradient_free::GradientFreeStatus;
use crate::core::utils::sample_standard_normal;
use crate::core::{
    Callbacks, LinearAlgebra, Matrix, MaxSteps, MinimizationSummary, NalgebraProvider,
    RandomScalar, SymmetricEigen, Vector,
};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{
    Algorithm, CostFunction, Status, SupportsParameterNames, Terminator, Transform,
    TransformedProblem,
};
use fastrand::Rng;
use std::collections::VecDeque;
use std::{marker::PhantomData, ops::ControlFlow};

#[derive(Clone, Debug)]
struct Candidate<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    x: Vector<T, B>,
    y: Vector<T, B>,
    fx: T,
}

impl<T: RandomScalar, B: LinearAlgebra<T>> SupportsParameterNames for CMAESConfig<T, B> {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

/// Configuration for linear-algebra-generic CMA-ES.
pub struct CMAESConfig<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Population size; zero selects the standard logarithmic default.
    population_size: usize,
    /// Initial global step size.
    initial_sigma: T,
    /// Optional names for the optimized parameters.
    parameter_names: Option<Vec<String>>,
    /// Optional coordinate transform.
    transform: Option<Box<dyn Transform<T, B>>>,
}

impl<T, B> Default for CMAESConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            population_size: 0,
            initial_sigma: T::literal(0.5),
            parameter_names: None,
            transform: None,
        }
    }
}

impl<T, B> CMAESConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Create a configuration with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the offspring population size.
    pub fn with_population_size(mut self, population_size: usize) -> GaneshResult<Self> {
        if population_size < 2 {
            return Err(GaneshError::ConfigError(
                "CMA-ES population size must be at least 2".to_string(),
            ));
        }
        self.population_size = population_size;
        Ok(self)
    }

    /// Set the initial global step size.
    pub fn with_initial_sigma(mut self, sigma: T) -> GaneshResult<Self> {
        if !sigma.is_finite() || sigma <= T::zero() {
            return Err(GaneshError::ConfigError(
                "CMA-ES sigma must be finite and greater than 0".to_string(),
            ));
        }
        self.initial_sigma = sigma;
        Ok(self)
    }

    /// Configure a coordinate transform or smooth bounds.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: Transform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

/// Terminates [`CMAES`] when its effective global step becomes sufficiently small.
#[derive(Clone, Copy)]
pub struct CMAESSigmaTerminator<T: RandomScalar = f64> {
    /// Absolute effective-step tolerance.
    pub eps_abs: T,
}
impl<T: RandomScalar> Default for CMAESSigmaTerminator<T> {
    fn default() -> Self {
        Self {
            eps_abs: T::literal(1e-10),
        }
    }
}

/// Terminates when a principal-axis perturbation no longer changes the mean.
#[derive(Clone, Copy, Default)]
pub struct CMAESNoEffectAxisTerminator;
/// Terminates when a coordinate perturbation no longer changes the mean.
#[derive(Clone, Copy, Default)]
pub struct CMAESNoEffectCoordTerminator;

/// Terminates when the covariance condition number is too large.
#[derive(Clone, Copy)]
pub struct CMAESConditionCovTerminator<T: RandomScalar = f64> {
    /// Maximum covariance condition number.
    pub max_condition: T,
}
impl<T: RandomScalar> Default for CMAESConditionCovTerminator<T> {
    fn default() -> Self {
        Self {
            max_condition: T::literal(1e14),
        }
    }
}

/// Terminates when the recent best values are exactly equal.
#[derive(Clone, Copy, Default)]
pub struct CMAESEqualFunValuesTerminator;
/// Terminates when recent best and median histories stagnate.
#[derive(Clone, Copy, Default)]
pub struct CMAESStagnationTerminator;

/// Terminates when the coordinate scale grows too far beyond its initial value.
#[derive(Clone, Copy)]
pub struct CMAESTolXUpTerminator<T: RandomScalar = f64> {
    /// Maximum growth relative to the initial sigma.
    pub max_growth: T,
}
impl<T: RandomScalar> Default for CMAESTolXUpTerminator<T> {
    fn default() -> Self {
        Self {
            max_growth: T::literal(1e4),
        }
    }
}

/// Terminates when the recent function-value range is sufficiently small.
#[derive(Clone, Copy)]
pub struct CMAESTolFunTerminator<T: RandomScalar = f64> {
    /// Absolute recent-fitness tolerance.
    pub eps_abs: T,
}
impl<T: RandomScalar> Default for CMAESTolFunTerminator<T> {
    fn default() -> Self {
        Self {
            eps_abs: T::literal(1e-12),
        }
    }
}

/// Terminates when coordinate deviations and the covariance path are sufficiently small.
#[derive(Clone, Copy)]
pub struct CMAESTolXTerminator<T: RandomScalar = f64> {
    /// Absolute coordinate tolerance; zero selects `1e-12 * initial_sigma`.
    pub eps_abs: T,
}
impl<T: RandomScalar> Default for CMAESTolXTerminator<T> {
    fn default() -> Self {
        Self { eps_abs: T::zero() }
    }
}

/// Scalar- and linear-algebra-generic covariance-matrix adaptation evolution strategy.
#[derive(Clone, Debug)]
pub struct CMAES<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    rng: Rng,
    mean: Vector<T, B>,
    covariance: Matrix<T, B>,
    path_c: Vector<T, B>,
    path_sigma: Vector<T, B>,
    sigma: T,
    weights: Vec<T>,
    mu_eff: T,
    generation: usize,
    population_size: usize,
    initial_sigma: T,
    best_history: VecDeque<T>,
    median_history: VecDeque<T>,
    recent_generation_values: Vec<T>,
    best_x: Vector<T, B>,
    best_fx: T,
    _provider: PhantomData<B>,
}

impl<T, B> CMAES<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Construct with an optional deterministic seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::new, Rng::with_seed),
            mean: Vector::zeros(0),
            covariance: Matrix::zeros(0, 0),
            path_c: Vector::zeros(0),
            path_sigma: Vector::zeros(0),
            sigma: T::one(),
            weights: Vec::new(),
            mu_eff: T::one(),
            generation: 0,
            population_size: 0,
            initial_sigma: T::one(),
            best_history: VecDeque::new(),
            median_history: VecDeque::new(),
            recent_generation_values: Vec::new(),
            best_x: Vector::zeros(0),
            best_fx: T::infinity(),
            _provider: PhantomData,
        }
    }

    fn outer(left: &Vector<T, B>, right: &Vector<T, B>) -> Matrix<T, B> {
        let mut matrix = Matrix::zeros(left.len(), right.len());
        for row in 0..left.len() {
            for column in 0..right.len() {
                matrix.set(row, column, left.get(row) * right.get(column));
            }
        }
        matrix
    }

    fn parameters(&self, dimension: usize) -> (T, T, T, T, T, T) {
        let n = T::literal(dimension as f64);
        let one = T::one();
        let cc = (T::literal(4.0) + self.mu_eff / n)
            / (n + T::literal(4.0) + T::literal(2.0) * self.mu_eff / n);
        let cs = (self.mu_eff + T::literal(2.0)) / (n + self.mu_eff + T::literal(5.0));
        let c1 = T::literal(2.0) / ((n + T::literal(1.3)).powi(2) + self.mu_eff);
        let cmu_candidate = T::literal(2.0) * (self.mu_eff - T::literal(2.0) + one / self.mu_eff)
            / ((n + T::literal(2.0)).powi(2) + self.mu_eff);
        let cmu = if cmu_candidate < one - c1 {
            cmu_candidate
        } else {
            one - c1
        };
        let damping_extra = ((self.mu_eff - one) / (n + one)).sqrt() - one;
        let damping = one
            + T::literal(2.0)
                * if damping_extra > T::zero() {
                    damping_extra
                } else {
                    T::zero()
                }
            + cs;
        let chi = n.sqrt() * (one - one / (T::literal(4.0) * n) + one / (T::literal(21.0) * n * n));
        (cc, cs, c1, cmu, damping, chi)
    }

    fn equal_fun_values_window(&self) -> usize {
        10 + (30.0 * self.mean.len() as f64 / self.population_size.max(1) as f64).ceil() as usize
    }

    fn stagnation_window(&self) -> usize {
        let minimum = 120
            + (30.0 * self.mean.len() as f64 / self.population_size.max(1) as f64).ceil() as usize;
        let proportional = (self.generation as f64 * 0.2).ceil() as usize;
        proportional.max(minimum).min(20_000)
    }

    fn record_generation_history(&mut self, population: &[Candidate<T, B>]) {
        self.best_history.push_back(population[0].fx);
        self.median_history
            .push_back(population[population.len() / 2].fx);
        self.recent_generation_values = population.iter().map(|candidate| candidate.fx).collect();
        let history_limit = self.equal_fun_values_window().max(self.stagnation_window());
        while self.best_history.len() > history_limit {
            self.best_history.pop_front();
        }
        while self.median_history.len() > self.stagnation_window() {
            self.median_history.pop_front();
        }
    }

    fn median(values: &[T]) -> T {
        let mut sorted = values.to_vec();
        sorted.sort_by(|left, right| left.total_cmp(right));
        let middle = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[middle - 1] + sorted[middle]) / T::literal(2.0)
        } else {
            sorted[middle]
        }
    }

    fn equal_fun_values(&self) -> bool {
        let equal_window = self.equal_fun_values_window();
        if self.best_history.len() < equal_window {
            return false;
        }
        let (minimum, maximum) = self
            .best_history
            .iter()
            .rev()
            .take(equal_window)
            .copied()
            .fold(
                (T::infinity(), -T::infinity()),
                |(minimum, maximum), value| {
                    (
                        if value < minimum { value } else { minimum },
                        if value > maximum { value } else { maximum },
                    )
                },
            );
        maximum.total_cmp(&minimum).is_eq()
    }

    fn stagnated(&self) -> bool {
        let window = self.stagnation_window();
        if self.best_history.len() < window || self.median_history.len() < window {
            return false;
        }
        let section = (window as f64 * 0.3).ceil() as usize;
        let best: Vec<T> = self
            .best_history
            .iter()
            .rev()
            .take(window)
            .copied()
            .collect();
        let median: Vec<T> = self
            .median_history
            .iter()
            .rev()
            .take(window)
            .copied()
            .collect();
        Self::median(&best[..section]) >= Self::median(&best[(window - section)..])
            && Self::median(&median[..section]) >= Self::median(&median[(window - section)..])
    }

    fn tol_fun(&self, tolerance: T) -> bool {
        let window = self.equal_fun_values_window();
        if self.best_history.len() < window || self.recent_generation_values.is_empty() {
            return false;
        }
        let (minimum, maximum) = self
            .recent_generation_values
            .iter()
            .copied()
            .chain(self.best_history.iter().rev().take(window).copied())
            .fold(
                (T::infinity(), -T::infinity()),
                |(minimum, maximum), value| {
                    (
                        if value < minimum { value } else { minimum },
                        if value > maximum { value } else { maximum },
                    )
                },
            );
        maximum - minimum < tolerance
    }

    fn tol_x(&self, tolerance: T) -> bool {
        let coordinates_small = (0..self.mean.len()).all(|index| {
            let variance = self.covariance.get(index, index);
            self.sigma
                * (if variance > T::zero() {
                    variance
                } else {
                    T::zero()
                })
                .sqrt()
                < tolerance
        });
        let path_small =
            (0..self.mean.len()).all(|index| self.sigma * self.path_c.get(index).abs() < tolerance);
        coordinates_small && path_small
    }

    fn eigensystem(&self) -> Option<(Vector<T, B>, Matrix<T, B>)>
    where
        B: SymmetricEigen<T>,
    {
        self.covariance.symmetric_eigen()
    }
}

macro_rules! cmaes_history_terminator {
    ($terminator:ty, |$algorithm:ident, $this:ident| $condition:expr, $message:literal) => {
        impl<T, B, P, U, E>
            Terminator<CMAES<T, B>, P, GradientFreeStatus<T, B>, U, E, CMAESConfig<T, B>>
            for $terminator
        where
            T: RandomScalar,
            B: LinearAlgebra<T> + SymmetricEigen<T>,
            P: CostFunction<T, B, U, E>,
        {
            fn check_for_termination(
                &mut self,
                _current_step: usize,
                algorithm: &mut CMAES<T, B>,
                _problem: &P,
                status: &mut GradientFreeStatus<T, B>,
                _args: &U,
                _config: &CMAESConfig<T, B>,
            ) -> ControlFlow<()> {
                let $algorithm = algorithm;
                let $this = self;
                if $condition {
                    status.set_message().succeed_with_message($message);
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            }
        }
    };
}

cmaes_history_terminator!(
    CMAESEqualFunValuesTerminator,
    |algorithm, _this| algorithm.equal_fun_values(),
    "EQUAL FUNCTION VALUES"
);
cmaes_history_terminator!(
    CMAESStagnationTerminator,
    |algorithm, _this| algorithm.stagnated(),
    "STAGNATION"
);
cmaes_history_terminator!(
    CMAESTolFunTerminator<T>,
    |algorithm, this| algorithm.tol_fun(this.eps_abs),
    "TOL FUN"
);
cmaes_history_terminator!(
    CMAESTolXTerminator<T>,
    |algorithm, this| {
        let tolerance = if this.eps_abs > T::zero() {
            this.eps_abs
        } else {
            T::literal(1e-12) * algorithm.initial_sigma
        };
        algorithm.tol_x(tolerance)
    },
    "TOL X"
);

impl<T, B, P, U, E> Terminator<CMAES<T, B>, P, GradientFreeStatus<T, B>, U, E, CMAESConfig<T, B>>
    for CMAESSigmaTerminator<T>
where
    T: RandomScalar,
    B: LinearAlgebra<T> + SymmetricEigen<T>,
    P: CostFunction<T, B, U, E>,
{
    fn check_for_termination(
        &mut self,
        _: usize,
        algorithm: &mut CMAES<T, B>,
        _: &P,
        status: &mut GradientFreeStatus<T, B>,
        _: &U,
        _: &CMAESConfig<T, B>,
    ) -> ControlFlow<()> {
        let Some((values, _)) = algorithm.eigensystem() else {
            return ControlFlow::Continue(());
        };
        let maximum = (0..values.len()).fold(T::zero(), |max, index| {
            if values.get(index) > max {
                values.get(index)
            } else {
                max
            }
        });
        if algorithm.sigma * maximum.sqrt() <= self.eps_abs {
            status
                .set_message()
                .succeed_with_message("SIGMA WITHIN TOLERANCE");
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

impl<T, B, P, U, E> Terminator<CMAES<T, B>, P, GradientFreeStatus<T, B>, U, E, CMAESConfig<T, B>>
    for CMAESConditionCovTerminator<T>
where
    T: RandomScalar,
    B: LinearAlgebra<T> + SymmetricEigen<T>,
    P: CostFunction<T, B, U, E>,
{
    fn check_for_termination(
        &mut self,
        _: usize,
        algorithm: &mut CMAES<T, B>,
        _: &P,
        status: &mut GradientFreeStatus<T, B>,
        _: &U,
        _: &CMAESConfig<T, B>,
    ) -> ControlFlow<()> {
        let Some((values, _)) = algorithm.eigensystem() else {
            return ControlFlow::Continue(());
        };
        let (minimum, maximum) =
            (0..values.len()).fold((T::infinity(), T::zero()), |(min, max), index| {
                let value = values.get(index);
                (
                    if value < min { value } else { min },
                    if value > max { value } else { max },
                )
            });
        if minimum <= T::epsilon() || maximum / minimum > self.max_condition {
            status
                .set_message()
                .succeed_with_message("CONDITION COVARIANCE");
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

impl<T, B, P, U, E> Terminator<CMAES<T, B>, P, GradientFreeStatus<T, B>, U, E, CMAESConfig<T, B>>
    for CMAESTolXUpTerminator<T>
where
    T: RandomScalar,
    B: LinearAlgebra<T> + SymmetricEigen<T>,
    P: CostFunction<T, B, U, E>,
{
    fn check_for_termination(
        &mut self,
        _: usize,
        algorithm: &mut CMAES<T, B>,
        _: &P,
        status: &mut GradientFreeStatus<T, B>,
        _: &U,
        _: &CMAESConfig<T, B>,
    ) -> ControlFlow<()> {
        let Some((values, _)) = algorithm.eigensystem() else {
            return ControlFlow::Continue(());
        };
        let maximum = (0..values.len()).fold(T::zero(), |max, index| {
            if values.get(index) > max {
                values.get(index)
            } else {
                max
            }
        });
        if algorithm.sigma * maximum.sqrt() > self.max_growth * algorithm.initial_sigma {
            status.set_message().succeed_with_message("TOL X UP");
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

impl<T, B, P, U, E> Terminator<CMAES<T, B>, P, GradientFreeStatus<T, B>, U, E, CMAESConfig<T, B>>
    for CMAESNoEffectAxisTerminator
where
    T: RandomScalar,
    B: LinearAlgebra<T> + SymmetricEigen<T>,
    P: CostFunction<T, B, U, E>,
{
    fn check_for_termination(
        &mut self,
        _: usize,
        algorithm: &mut CMAES<T, B>,
        _: &P,
        status: &mut GradientFreeStatus<T, B>,
        _: &U,
        _: &CMAESConfig<T, B>,
    ) -> ControlFlow<()> {
        let Some((values, vectors)) = algorithm.eigensystem() else {
            return ControlFlow::Continue(());
        };
        for axis in 0..algorithm.mean.len() {
            let variance = if values.get(axis) > T::zero() {
                values.get(axis)
            } else {
                T::zero()
            };
            let unchanged = (0..algorithm.mean.len()).all(|coordinate| {
                algorithm.mean.get(coordinate)
                    + T::literal(0.1)
                        * algorithm.sigma
                        * variance.sqrt()
                        * vectors.get(coordinate, axis)
                    == algorithm.mean.get(coordinate)
            });
            if unchanged {
                status.set_message().succeed_with_message("NO EFFECT AXIS");
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

impl<T, B, P, U, E> Terminator<CMAES<T, B>, P, GradientFreeStatus<T, B>, U, E, CMAESConfig<T, B>>
    for CMAESNoEffectCoordTerminator
where
    T: RandomScalar,
    B: LinearAlgebra<T> + SymmetricEigen<T>,
    P: CostFunction<T, B, U, E>,
{
    fn check_for_termination(
        &mut self,
        _: usize,
        algorithm: &mut CMAES<T, B>,
        _: &P,
        status: &mut GradientFreeStatus<T, B>,
        _: &U,
        _: &CMAESConfig<T, B>,
    ) -> ControlFlow<()> {
        for coordinate in 0..algorithm.mean.len() {
            let variance = algorithm.covariance.get(coordinate, coordinate);
            let variance = if variance > T::zero() {
                variance
            } else {
                T::zero()
            };
            if algorithm.mean.get(coordinate) + T::literal(0.2) * algorithm.sigma * variance.sqrt()
                == algorithm.mean.get(coordinate)
            {
                status.set_message().succeed_with_message("NO EFFECT COORD");
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

impl<T, B> Default for CMAES<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::new(Some(0))
    }
}

impl<T, B, P, U, E> Algorithm<P, GradientFreeStatus<T, B>, U, E> for CMAES<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T> + SymmetricEigen<T>,
    P: CostFunction<T, B, U, E>,
{
    type Summary = MinimizationSummary<T, B>;
    type Config = CMAESConfig<T, B>;
    type Init = Vector<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut GradientFreeStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        self.mean = transformed.to_internal(init);
        let dimension = self.mean.len();
        self.covariance = Matrix::identity(dimension);
        self.path_c = Vector::zeros(dimension);
        self.path_sigma = Vector::zeros(dimension);
        self.sigma = config.initial_sigma;
        self.initial_sigma = config.initial_sigma;
        self.generation = 0;
        let lambda = if config.population_size == 0 {
            4 + (3.0 * (dimension as f64).ln()).floor() as usize
        } else {
            config.population_size.max(2)
        };
        self.population_size = lambda;
        self.best_history.clear();
        self.median_history.clear();
        self.recent_generation_values.clear();
        let raw_weights = (0..lambda)
            .map(|index| T::literal((((lambda + 1) as f64) / 2.0).ln() - ((index + 1) as f64).ln()))
            .collect::<Vec<_>>();
        let positive_sum = raw_weights
            .iter()
            .copied()
            .filter(|weight| *weight > T::zero())
            .fold(T::zero(), |sum, weight| sum + weight);
        let negative_sum = raw_weights
            .iter()
            .copied()
            .filter(|weight| *weight < T::zero())
            .fold(T::zero(), |sum, weight| sum + weight.abs());
        let positive_weights = raw_weights
            .iter()
            .copied()
            .filter(|weight| *weight > T::zero())
            .map(|weight| weight / positive_sum)
            .collect::<Vec<_>>();
        self.mu_eff = T::one()
            / positive_weights
                .iter()
                .copied()
                .map(|weight| weight * weight)
                .fold(T::zero(), |sum, value| sum + value);
        let negative_weights = raw_weights
            .iter()
            .copied()
            .filter(|weight| *weight < T::zero())
            .map(|weight| weight / negative_sum)
            .collect::<Vec<_>>();
        let mu_eff_minus = if negative_weights.is_empty() {
            T::zero()
        } else {
            T::one()
                / negative_weights
                    .iter()
                    .copied()
                    .map(|weight| weight * weight)
                    .fold(T::zero(), |sum, value| sum + value)
        };
        let (_, _, c1, cmu, _, _) = self.parameters(dimension);
        let n = T::literal(dimension as f64);
        let alpha_mu_minus = T::one() + c1 / cmu;
        let alpha_mu_eff_minus =
            T::one() + T::literal(2.0) * mu_eff_minus / (self.mu_eff + T::literal(2.0));
        let alpha_posdef = (T::one() - c1 - cmu) / (n * cmu);
        let negative_scale = [alpha_mu_minus, alpha_mu_eff_minus, alpha_posdef]
            .into_iter()
            .fold(
                T::infinity(),
                |minimum, value| if value < minimum { value } else { minimum },
            );
        self.weights = raw_weights
            .into_iter()
            .map(|weight| {
                if weight >= T::zero() {
                    weight / positive_sum
                } else {
                    negative_scale * weight / negative_sum
                }
            })
            .collect();
        self.best_x = self.mean.clone();
        self.best_fx = transformed.evaluate(&self.mean, args)?;
        status.evals.record_f();
        status.initialize(init.clone(), self.best_fx);
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut GradientFreeStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        let dimension = self.mean.len();
        let Some((eigenvalues, eigenvectors)) = self.covariance.symmetric_eigen() else {
            status
                .set_message()
                .fail_with_message("COVARIANCE EIGENDECOMPOSITION FAILED");
            return Ok(());
        };
        let clamped: Vector<T, B> = Vector::from_vec(
            (0..dimension)
                .map(|index| {
                    let value = eigenvalues.get(index);
                    if value > T::epsilon() {
                        value
                    } else {
                        T::epsilon()
                    }
                })
                .collect(),
        );
        let lambda = if config.population_size == 0 {
            4 + (3.0 * (dimension as f64).ln()).floor() as usize
        } else {
            config.population_size.max(2)
        };
        let mut population = Vec::with_capacity(lambda);
        for _ in 0..lambda {
            let normal: Vector<T, B> = Vector::from_vec(
                (0..dimension)
                    .map(|_| sample_standard_normal(&mut self.rng))
                    .collect(),
            );
            let scaled: Vector<T, B> = Vector::from_vec(
                (0..dimension)
                    .map(|index| normal.get(index) * clamped.get(index).sqrt())
                    .collect(),
            );
            let y = eigenvectors.mul_vec(&scaled);
            let x = self.mean.add_scaled(&y, self.sigma);
            let fx = transformed.evaluate(&x, args)?;
            status.evals.record_f();
            population.push(Candidate { x, y, fx });
        }
        population.sort_by(|left, right| left.fx.total_cmp(&right.fx));
        self.record_generation_history(&population);
        if population[0].fx < self.best_fx {
            self.best_fx = population[0].fx;
            self.best_x = population[0].x.clone();
        }
        let old_mean = self.mean.clone();
        let mut y_weighted = Vector::zeros(dimension);
        for (candidate, weight) in population.iter().zip(&self.weights) {
            if *weight <= T::zero() {
                break;
            }
            y_weighted = y_weighted.add_scaled(&candidate.y, *weight);
        }
        self.mean = old_mean.add_scaled(&y_weighted, self.sigma);
        let coordinates = eigenvectors.transpose().mul_vec(&y_weighted);
        let inverse_scaled = Vector::from_vec(
            (0..dimension)
                .map(|index| coordinates.get(index) / clamped.get(index).sqrt())
                .collect(),
        );
        let inverse_sqrt_y = eigenvectors.mul_vec(&inverse_scaled);
        let (cc, cs, c1, cmu, damping, chi) = self.parameters(dimension);
        let path_sigma_scale = (cs * (T::literal(2.0) - cs) * self.mu_eff).sqrt();
        self.path_sigma = self
            .path_sigma
            .scale(T::one() - cs)
            .add_scaled(&inverse_sqrt_y, path_sigma_scale);
        let decay_exponent =
            i32::try_from(self.generation.saturating_add(1).saturating_mul(2)).unwrap_or(i32::MAX);
        let decay = (T::one() - cs).powi(decay_exponent);
        let normalized_path = self.path_sigma.norm() / (T::one() - decay).sqrt() / chi;
        let h_sigma = normalized_path
            < T::literal(1.4) + T::literal(2.0) / (T::literal(dimension as f64) + T::one());
        self.path_c = self.path_c.scale(T::one() - cc);
        if h_sigma {
            self.path_c = self.path_c.add_scaled(
                &y_weighted,
                (cc * (T::literal(2.0) - cc) * self.mu_eff).sqrt(),
            );
        }
        let old_covariance = self.covariance.clone();
        let h_correction = if h_sigma {
            T::zero()
        } else {
            cc * (T::literal(2.0) - cc)
        };
        let covariance_decay = old_covariance.scale(T::one() - c1 - cmu + c1 * h_correction);
        let rank_one = Self::outer(&self.path_c, &self.path_c).scale(c1);
        self.covariance = &covariance_decay + &rank_one;
        for (candidate, weight) in population.iter().zip(&self.weights) {
            let mut covariance_weight = *weight;
            if covariance_weight < T::zero() {
                let coordinates = eigenvectors.transpose().mul_vec(&candidate.y);
                let norm_squared = (0..dimension).fold(T::zero(), |sum, index| {
                    sum + coordinates.get(index).powi(2) / clamped.get(index)
                });
                if norm_squared > T::epsilon() {
                    covariance_weight =
                        covariance_weight * T::literal(dimension as f64) / norm_squared;
                }
            }
            let contribution =
                Self::outer(&candidate.y, &candidate.y).scale(cmu * covariance_weight);
            self.covariance = &self.covariance + &contribution;
        }
        self.sigma =
            self.sigma * ((cs / damping) * (self.path_sigma.norm() / chi - T::one())).exp();
        self.generation += 1;
        if let Some((values, vectors)) = self.covariance.symmetric_eigen() {
            let mut diagonal = Matrix::zeros(dimension, dimension);
            for index in 0..dimension {
                let value = values.get(index);
                diagonal.set(
                    index,
                    index,
                    if value > T::epsilon() {
                        value
                    } else {
                        T::epsilon()
                    },
                );
            }
            self.covariance = vectors.mul_mat(&diagonal).mul_mat(&vectors.transpose());
        }
        status.set_position(transformed.to_external(&self.best_x), self.best_fx);
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &GradientFreeStatus<T, B>,
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
            covariance: self.covariance.scale(self.sigma * self.sigma),
        })
    }

    fn reset(&mut self) {
        self.mean = Vector::zeros(0);
        self.covariance = Matrix::zeros(0, 0);
        self.path_c = Vector::zeros(0);
        self.path_sigma = Vector::zeros(0);
        self.sigma = T::one();
        self.initial_sigma = T::one();
        self.population_size = 0;
        self.generation = 0;
        self.weights.clear();
        self.best_history.clear();
        self.median_history.clear();
        self.recent_generation_values.clear();
        self.best_x = Vector::zeros(0);
        self.best_fx = T::infinity();
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientFreeStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty()
            .with_terminator(CMAESSigmaTerminator::default())
            .with_terminator(CMAESNoEffectAxisTerminator)
            .with_terminator(CMAESNoEffectCoordTerminator)
            .with_terminator(CMAESConditionCovTerminator::default())
            .with_terminator(CMAESEqualFunValuesTerminator)
            .with_terminator(CMAESStagnationTerminator)
            .with_terminator(CMAESTolXUpTerminator::default())
            .with_terminator(CMAESTolFunTerminator::default())
            .with_terminator(CMAESTolXTerminator::default())
            .with_terminator(MaxSteps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::Rosenbrock;
    use crate::traits::Bounds;
    use std::convert::Infallible;

    struct Flat;

    impl<T, B> CostFunction<T, B> for Flat
    where
        T: RandomScalar,
        B: LinearAlgebra<T>,
    {
        fn evaluate(&self, _x: &Vector<T, B>, _: &()) -> Result<T, Infallible> {
            Ok(T::one())
        }
    }

    #[test]
    fn cmaes_runs_f32_with_full_covariance_and_bounds() {
        let bounds = Bounds::new([(-3.0_f32, 3.0), (-3.0, 3.0)]).unwrap();
        let config = CMAESConfig {
            population_size: 20,
            initial_sigma: 0.7,
            ..CMAESConfig::<f32>::default()
        }
        .with_transform(bounds);
        let mut algorithm = CMAES::<f32>::new(Some(31));
        let result = algorithm
            .process(
                &Rosenbrock { n: 2 },
                &(),
                Vector::from_vec(vec![-1.2, 1.0]),
                config,
                Callbacks::empty().with_terminator(MaxSteps(2_000)),
            )
            .unwrap();
        assert!(result.fx < 1e-3);
    }

    #[test]
    fn cmaes_stops_on_equal_value_policy_and_preserves_metadata() {
        let names = vec!["x".to_string(), "y".to_string()];
        let config = CMAESConfig {
            parameter_names: Some(names.clone()),
            ..CMAESConfig::<f64>::default()
        };
        let result = CMAES::<f64>::new(Some(0))
            .process(
                &Flat,
                &(),
                Vector::from_vec(vec![0.0, 0.0]),
                config,
                CMAES::default_callbacks().with_terminator(MaxSteps(500)),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(
            result.message.text_or_empty().contains("FUNCTION VALUES")
                || result.message.text_or_empty().contains("TOL FUN")
        );
        assert_eq!(result.parameter_names, Some(names));
    }
}
