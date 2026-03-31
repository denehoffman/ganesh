use crate::{
    DMatrix, DVector, Float,
    algorithms::gradient_free::GradientFreeStatus,
    core::{Bounds, Callbacks, MaxSteps, MinimizationSummary, Point, utils::SampleFloat},
    error::{GaneshError, GaneshResult},
    traits::algorithm::{BoundsHandlingMode, resolve_bounds_and_transform},
    traits::{
        Algorithm, CostFunction, Status, SupportsBounds, SupportsParameterNames,
        SupportsTransform, Terminator, Transform,
    },
};
use fastrand::Rng;
use nalgebra::SymmetricEigen;
use std::{collections::VecDeque, ops::ControlFlow};

/// A [`Terminator`] which stops [`CMAES`] once the global step size becomes sufficiently small.
#[derive(Clone, Copy)]
pub struct CMAESSigmaTerminator {
    /// Absolute tolerance on the effective step size.
    pub eps_abs: Float,
}

/// Stop if adding a small principal-axis perturbation no longer changes the mean.
#[derive(Clone, Copy, Default)]
pub struct CMAESNoEffectAxisTerminator;

impl<P, U, E> Terminator<CMAES, P, GradientFreeStatus, U, E, CMAESConfig>
    for CMAESNoEffectAxisTerminator
where
    P: CostFunction<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut CMAES,
        _problem: &P,
        status: &mut GradientFreeStatus,
        _args: &U,
        _config: &CMAESConfig,
    ) -> ControlFlow<()> {
        for i in 0..algorithm.mean.len() {
            let shifted = &algorithm.mean
                + algorithm
                    .b_mat
                    .column(i)
                    .into_owned()
                    .scale(0.1 * algorithm.sigma * algorithm.d_vec[i]);
            if shifted == algorithm.mean {
                status
                    .set_message()
                    .succeed_with_message("NO EFFECT AXIS");
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

/// Stop if adding a small coordinate-wise perturbation no longer changes the mean.
#[derive(Clone, Copy, Default)]
pub struct CMAESNoEffectCoordTerminator;

impl<P, U, E> Terminator<CMAES, P, GradientFreeStatus, U, E, CMAESConfig>
    for CMAESNoEffectCoordTerminator
where
    P: CostFunction<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut CMAES,
        _problem: &P,
        status: &mut GradientFreeStatus,
        _args: &U,
        _config: &CMAESConfig,
    ) -> ControlFlow<()> {
        for i in 0..algorithm.mean.len() {
            let mut shifted = algorithm.mean.clone();
            shifted[i] += 0.2 * algorithm.sigma * algorithm.cov[(i, i)].max(0.0).sqrt();
            if shifted == algorithm.mean {
                status
                    .set_message()
                    .succeed_with_message("NO EFFECT COORD");
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}

/// Stop if the covariance matrix condition number becomes too large.
#[derive(Clone, Copy)]
pub struct CMAESConditionCovTerminator {
    /// Maximum allowed covariance condition number.
    pub max_condition: Float,
}

impl Default for CMAESConditionCovTerminator {
    fn default() -> Self {
        Self { max_condition: 1e14 }
    }
}

impl<P, U, E> Terminator<CMAES, P, GradientFreeStatus, U, E, CMAESConfig>
    for CMAESConditionCovTerminator
where
    P: CostFunction<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut CMAES,
        _problem: &P,
        status: &mut GradientFreeStatus,
        _args: &U,
        _config: &CMAESConfig,
    ) -> ControlFlow<()> {
        if algorithm.condition_covariance() > self.max_condition {
            status
                .set_message()
                .succeed_with_message("CONDITION COVARIANCE");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// Stop if the best fitness values over the recent history have zero range.
#[derive(Clone, Copy, Default)]
pub struct CMAESEqualFunValuesTerminator;

impl<P, U, E> Terminator<CMAES, P, GradientFreeStatus, U, E, CMAESConfig>
    for CMAESEqualFunValuesTerminator
where
    P: CostFunction<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut CMAES,
        _problem: &P,
        status: &mut GradientFreeStatus,
        _args: &U,
        _config: &CMAESConfig,
    ) -> ControlFlow<()> {
        if algorithm.equal_fun_values() {
            status
                .set_message()
                .succeed_with_message("EQUAL FUNCTION VALUES");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// Stop if recent best and median fitness medians indicate stagnation.
#[derive(Clone, Copy, Default)]
pub struct CMAESStagnationTerminator;

impl<P, U, E> Terminator<CMAES, P, GradientFreeStatus, U, E, CMAESConfig>
    for CMAESStagnationTerminator
where
    P: CostFunction<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut CMAES,
        _problem: &P,
        status: &mut GradientFreeStatus,
        _args: &U,
        _config: &CMAESConfig,
    ) -> ControlFlow<()> {
        if algorithm.stagnated() {
            status.set_message().succeed_with_message("STAGNATION");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// Stop if the overall coordinate scale has increased too much.
#[derive(Clone, Copy)]
pub struct CMAESTolXUpTerminator {
    /// Multiplicative ceiling relative to the initial sigma.
    pub max_growth: Float,
}

impl Default for CMAESTolXUpTerminator {
    fn default() -> Self {
        Self { max_growth: 1e4 }
    }
}

impl<P, U, E> Terminator<CMAES, P, GradientFreeStatus, U, E, CMAESConfig>
    for CMAESTolXUpTerminator
where
    P: CostFunction<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut CMAES,
        _problem: &P,
        status: &mut GradientFreeStatus,
        _args: &U,
        _config: &CMAESConfig,
    ) -> ControlFlow<()> {
        if algorithm.sigma * algorithm.d_vec.amax() > self.max_growth * algorithm.initial_sigma {
            status.set_message().succeed_with_message("TOL X UP");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// Stop if recent best and current-generation function values fall below a tolerance range.
#[derive(Clone, Copy)]
pub struct CMAESTolFunTerminator {
    /// Absolute tolerance on recent fitness spread.
    pub eps_abs: Float,
}

impl Default for CMAESTolFunTerminator {
    fn default() -> Self {
        Self { eps_abs: 1e-12 }
    }
}

impl<P, U, E> Terminator<CMAES, P, GradientFreeStatus, U, E, CMAESConfig>
    for CMAESTolFunTerminator
where
    P: CostFunction<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut CMAES,
        _problem: &P,
        status: &mut GradientFreeStatus,
        _args: &U,
        _config: &CMAESConfig,
    ) -> ControlFlow<()> {
        if algorithm.tol_fun(self.eps_abs) {
            status.set_message().succeed_with_message("TOL FUN");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// Stop if the coordinate-wise standard deviations and evolution path become too small.
#[derive(Clone, Copy)]
pub struct CMAESTolXTerminator {
    /// Absolute coordinate tolerance.
    pub eps_abs: Float,
}

impl Default for CMAESTolXTerminator {
    fn default() -> Self {
        Self { eps_abs: 0.0 }
    }
}

impl<P, U, E> Terminator<CMAES, P, GradientFreeStatus, U, E, CMAESConfig> for CMAESTolXTerminator
where
    P: CostFunction<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut CMAES,
        _problem: &P,
        status: &mut GradientFreeStatus,
        _args: &U,
        _config: &CMAESConfig,
    ) -> ControlFlow<()> {
        let tol = if self.eps_abs > 0.0 {
            self.eps_abs
        } else {
            1e-12 * algorithm.initial_sigma
        };
        if algorithm.tol_x(tol) {
            status.set_message().succeed_with_message("TOL X");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

impl Default for CMAESSigmaTerminator {
    fn default() -> Self {
        Self { eps_abs: 1e-10 }
    }
}

impl<P, U, E> Terminator<CMAES, P, GradientFreeStatus, U, E, CMAESConfig> for CMAESSigmaTerminator
where
    P: CostFunction<U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut CMAES,
        _problem: &P,
        status: &mut GradientFreeStatus,
        _args: &U,
        _config: &CMAESConfig,
    ) -> ControlFlow<()> {
        if algorithm.sigma * algorithm.d_vec.amax() <= self.eps_abs {
            status
                .set_message()
                .succeed_with_message("SIGMA WITHIN TOLERANCE");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// Configuration for the [`CMAES`] algorithm.
#[derive(Clone)]
pub struct CMAESConfig {
    x0: DVector<Float>,
    sigma: Float,
    population_size: Option<usize>,
    bounds: Option<Bounds>,
    parameter_names: Option<Vec<String>>,
    transform: Option<Box<dyn Transform>>,
}

impl CMAESConfig {
    /// Create a new [`CMAESConfig`] from the initial mean and global step size.
    pub fn new<I>(x0: I, sigma: Float) -> GaneshResult<Self>
    where
        I: AsRef<[Float]>,
    {
        let x0 = DVector::from_row_slice(x0.as_ref());
        if x0.is_empty() {
            return Err(GaneshError::ConfigError(
                "CMA-ES requires at least one parameter".to_string(),
            ));
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(GaneshError::ConfigError(
                "CMA-ES sigma must be finite and greater than 0".to_string(),
            ));
        }
        Ok(Self {
            x0,
            sigma,
            population_size: None,
            bounds: None,
            parameter_names: None,
            transform: None,
        })
    }

    /// Set the offspring population size `lambda`.
    pub fn with_population_size(mut self, population_size: usize) -> GaneshResult<Self> {
        if population_size < 2 {
            return Err(GaneshError::ConfigError(
                "CMA-ES population size must be at least 2".to_string(),
            ));
        }
        self.population_size = Some(population_size);
        Ok(self)
    }
}

impl SupportsBounds for CMAESConfig {
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds> {
        &mut self.bounds
    }
}

impl SupportsTransform for CMAESConfig {
    fn get_transform_mut(&mut self) -> &mut Option<Box<dyn Transform>> {
        &mut self.transform
    }
}

impl SupportsParameterNames for CMAESConfig {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

#[derive(Clone)]
struct CMAESCandidate {
    y: DVector<Float>,
    point: Point<DVector<Float>>,
}

/// Covariance Matrix Adaptation Evolution Strategy.
///
/// This implementation follows the practical recommendations from [^1], including active
/// covariance updates with negative recombination weights and several standard termination
/// criteria.
///
/// [^1]: [Nikolaus Hansen, “The CMA Evolution Strategy: A Tutorial”, arXiv:1604.00772 (2016).](https://arxiv.org/pdf/1604.00772)
#[derive(Clone)]
pub struct CMAES {
    rng: Rng,
    mean: DVector<Float>,
    sigma: Float,
    initial_sigma: Float,
    cov: DMatrix<Float>,
    b_mat: DMatrix<Float>,
    d_vec: DVector<Float>,
    inv_sqrt_c: DMatrix<Float>,
    p_c: DVector<Float>,
    p_sigma: DVector<Float>,
    weights: DVector<Float>,
    mu_eff_minus: Float,
    mu: usize,
    lambda: usize,
    mu_eff: Float,
    c_c: Float,
    c_sigma: Float,
    c1: Float,
    c_mu: Float,
    damping: Float,
    chi_n: Float,
    generation: usize,
    best: Point<DVector<Float>>,
    best_history: VecDeque<Float>,
    median_history: VecDeque<Float>,
    recent_generation_values: Vec<Float>,
    resolved_transform: Option<Box<dyn Transform>>,
}

impl Default for CMAES {
    fn default() -> Self {
        Self::new(Some(0))
    }
}

impl CMAES {
    /// Create a new [`CMAES`] optimizer with an optional seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::new, Rng::with_seed),
            mean: DVector::zeros(0),
            sigma: 1.0,
            initial_sigma: 1.0,
            cov: DMatrix::zeros(0, 0),
            b_mat: DMatrix::zeros(0, 0),
            d_vec: DVector::zeros(0),
            inv_sqrt_c: DMatrix::zeros(0, 0),
            p_c: DVector::zeros(0),
            p_sigma: DVector::zeros(0),
            weights: DVector::zeros(0),
            mu_eff_minus: 0.0,
            mu: 0,
            lambda: 0,
            mu_eff: 0.0,
            c_c: 0.0,
            c_sigma: 0.0,
            c1: 0.0,
            c_mu: 0.0,
            damping: 0.0,
            chi_n: 0.0,
            generation: 0,
            best: Point::default(),
            best_history: VecDeque::new(),
            median_history: VecDeque::new(),
            recent_generation_values: Vec::new(),
            resolved_transform: None,
        }
    }

    fn initialize_strategy(&mut self, dimension: usize, config: &CMAESConfig) {
        self.sigma = config.sigma;
        self.initial_sigma = config.sigma;
        self.lambda = config
            .population_size
            .unwrap_or_else(|| 4 + (3.0 * (dimension as Float).ln()).floor() as usize);
        self.mu = self.lambda / 2;

        let n = dimension as Float;
        let w_prime = DVector::from_iterator(
            self.lambda,
            (0..self.lambda)
                .map(|i| (((self.lambda + 1) as Float) / 2.0).ln() - ((i + 1) as Float).ln()),
        );
        let positive_sum: Float = w_prime.iter().copied().filter(|w| *w >= 0.0).sum();
        let negative_abs_sum: Float = w_prime
            .iter()
            .copied()
            .filter(|w| *w < 0.0)
            .map(Float::abs)
            .sum();
        let positive_weights: Vec<Float> = w_prime
            .iter()
            .copied()
            .filter(|w| *w > 0.0)
            .map(|w| w / positive_sum)
            .collect();
        self.mu_eff = 1.0 / positive_weights.iter().map(|w| w.powi(2)).sum::<Float>();

        self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n);
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0);
        self.c1 = 2.0 / ((n + 1.3).powi(2) + self.mu_eff);
        self.c_mu = Float::min(
            1.0 - self.c1,
            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((n + 2.0).powi(2) + self.mu_eff),
        );
        let negative_weights: Vec<Float> = w_prime
            .iter()
            .copied()
            .filter(|w| *w < 0.0)
            .map(|w| w / negative_abs_sum)
            .collect();
        self.mu_eff_minus = if negative_weights.is_empty() {
            0.0
        } else {
            1.0 / negative_weights.iter().map(|w| w.powi(2)).sum::<Float>()
        };
        let alpha_mu_minus = 1.0 + self.c1 / self.c_mu;
        let alpha_mu_eff_minus = 1.0 + 2.0 * self.mu_eff_minus / (self.mu_eff + 2.0);
        let alpha_posdef = (1.0 - self.c1 - self.c_mu) / (n * self.c_mu);
        let negative_scale = alpha_mu_minus.min(alpha_mu_eff_minus).min(alpha_posdef);
        self.weights = DVector::from_iterator(
            self.lambda,
            w_prime.iter().map(|w| {
                if *w >= 0.0 {
                    *w / positive_sum
                } else {
                    negative_scale * *w / negative_abs_sum
                }
            }),
        );

        self.damping =
            1.0 + 2.0 * Float::max(0.0, ((self.mu_eff - 1.0) / (n + 1.0)).sqrt() - 1.0)
                + self.c_sigma;
        self.chi_n = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n.powi(2)));

        self.cov = DMatrix::identity(dimension, dimension);
        self.b_mat = DMatrix::identity(dimension, dimension);
        self.d_vec = DVector::from_element(dimension, 1.0);
        self.inv_sqrt_c = DMatrix::identity(dimension, dimension);
        self.p_c = DVector::zeros(dimension);
        self.p_sigma = DVector::zeros(dimension);
        self.generation = 0;
        self.best_history.clear();
        self.median_history.clear();
        self.recent_generation_values.clear();
    }

    fn update_eigendecomposition(&mut self) {
        let sym_cov = 0.5 * (&self.cov + self.cov.transpose());
        let eig = SymmetricEigen::new(sym_cov);
        self.b_mat = eig.eigenvectors;
        self.d_vec = eig.eigenvalues.map(|value| Float::max(value, Float::EPSILON).sqrt());
        let inv_diag = DMatrix::from_diagonal(&self.d_vec.map(|value| 1.0 / value));
        self.inv_sqrt_c = &self.b_mat * inv_diag * self.b_mat.transpose();
        self.cov = &self.b_mat
            * DMatrix::from_diagonal(&self.d_vec.map(|value| value * value))
            * self.b_mat.transpose();
    }

    fn sample_population<P, U, E>(
        &mut self,
        problem: &P,
        args: &U,
    ) -> Result<Vec<CMAESCandidate>, E>
    where
        P: CostFunction<U, E>,
    {
        let mut population = Vec::with_capacity(self.lambda);
        for _ in 0..self.lambda {
            let y = self
                .rng
                .mv_normal(&DVector::zeros(self.mean.len()), &self.cov);
            let x = &self.mean + y.scale(self.sigma);
            let mut point = Point::from(x);
            point.evaluate_transformed(problem, &self.resolved_transform, args)?;
            population.push(CMAESCandidate { y, point });
        }
        population.sort_by(|a, b| a.point.total_cmp(&b.point));
        Ok(population)
    }

    fn update_distribution(&mut self, population: &[CMAESCandidate]) {
        let mean_old = self.mean.clone();
        let mut y_w = DVector::zeros(self.mean.len());
        for (i, candidate) in population.iter().take(self.mu).enumerate() {
            let weight = self.weights[i];
            y_w += candidate.y.scale(weight);
        }
        let z_mean = &self.inv_sqrt_c * &y_w;
        self.mean += y_w.scale(self.sigma);

        self.p_sigma = self.p_sigma.scale(1.0 - self.c_sigma)
            + (&self.b_mat * z_mean)
                .scale((self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt());

        let n = self.mean.len() as Float;
        let norm_factor =
            (1.0 - (1.0 - self.c_sigma).powf(2.0 * (self.generation + 1) as Float)).sqrt();
        let h_sigma = if self.p_sigma.norm() / norm_factor / self.chi_n < 1.4 + 2.0 / (n + 1.0) {
            1.0
        } else {
            0.0
        };

        let mean_shift = (&self.mean - &mean_old).unscale(self.sigma);
        self.p_c = self.p_c.scale(1.0 - self.c_c)
            + mean_shift
                .scale(h_sigma * (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt());

        let rank_one = &self.p_c * self.p_c.transpose();
        let mut rank_mu = DMatrix::zeros(self.mean.len(), self.mean.len());
        let n = self.mean.len() as Float;
        for (i, candidate) in population.iter().enumerate() {
            let mut weight = self.weights[i];
            if weight < 0.0 {
                let norm = (&self.inv_sqrt_c * &candidate.y).norm_squared();
                if norm > Float::EPSILON {
                    weight *= n / norm;
                }
            }
            rank_mu += (&candidate.y * candidate.y.transpose()).scale(weight);
        }
        self.cov = self
            .cov
            .scale(1.0 - self.c1 - self.c_mu + (1.0 - h_sigma) * self.c1 * self.c_c * (2.0 - self.c_c))
            + rank_one.scale(self.c1)
            + rank_mu.scale(self.c_mu);

        self.sigma *= Float::exp((self.c_sigma / self.damping) * (self.p_sigma.norm() / self.chi_n - 1.0));
        self.generation += 1;
        self.record_generation_history(population);
        self.update_eigendecomposition();
    }

    fn equal_fun_values_window(&self) -> usize {
        10 + (30.0 * self.mean.len() as Float / self.lambda as Float).ceil() as usize
    }

    fn stagnation_window(&self) -> usize {
        let min_window = 120 + (30.0 * self.mean.len() as Float / self.lambda as Float).ceil() as usize;
        let twenty_percent = ((self.generation as Float) * 0.2).ceil() as usize;
        twenty_percent.max(min_window).min(20_000)
    }

    fn record_generation_history(&mut self, population: &[CMAESCandidate]) {
        let best = population[0].point.fx_checked();
        let median = population[population.len() / 2].point.fx_checked();
        self.recent_generation_values = population.iter().map(|c| c.point.fx_checked()).collect();
        self.best_history.push_back(best);
        self.median_history.push_back(median);
        let max_equal_window = self.equal_fun_values_window();
        while self.best_history.len() > max_equal_window.max(self.stagnation_window()) {
            self.best_history.pop_front();
        }
        while self.median_history.len() > self.stagnation_window() {
            self.median_history.pop_front();
        }
    }

    fn equal_fun_values(&self) -> bool {
        let window = self.equal_fun_values_window();
        if self.best_history.len() < window {
            return false;
        }
        let recent = self.best_history.iter().rev().take(window);
        let mut min = Float::INFINITY;
        let mut max = Float::NEG_INFINITY;
        for &value in recent {
            min = min.min(value);
            max = max.max(value);
        }
        max.total_cmp(&min).is_eq()
    }

    fn stagnated(&self) -> bool {
        let window = self.stagnation_window();
        if self.best_history.len() < window || self.median_history.len() < window {
            return false;
        }
        let section = ((window as Float) * 0.3).ceil() as usize;
        let best: Vec<Float> = self.best_history.iter().rev().take(window).copied().collect();
        let median: Vec<Float> = self.median_history.iter().rev().take(window).copied().collect();
        let best_recent = Self::median(&best[..section]);
        let best_early = Self::median(&best[(window - section)..]);
        let median_recent = Self::median(&median[..section]);
        let median_early = Self::median(&median[(window - section)..]);
        best_recent >= best_early && median_recent >= median_early
    }

    fn tol_fun(&self, tol: Float) -> bool {
        let window = self.equal_fun_values_window();
        if self.best_history.len() < window || self.recent_generation_values.is_empty() {
            return false;
        }
        let mut min = Float::INFINITY;
        let mut max = Float::NEG_INFINITY;
        for &value in self.best_history.iter().rev().take(window) {
            min = min.min(value);
            max = max.max(value);
        }
        for &value in &self.recent_generation_values {
            min = min.min(value);
            max = max.max(value);
        }
        max - min < tol
    }

    fn tol_x(&self, tol: Float) -> bool {
        let coord_std_small = self
            .cov
            .diagonal()
            .iter()
            .all(|value| self.sigma * value.max(0.0).sqrt() < tol);
        let path_small = self
            .p_c
            .iter()
            .all(|value| self.sigma * value.abs() < tol);
        coord_std_small && path_small
    }

    fn condition_covariance(&self) -> Float {
        let min = self.d_vec.min();
        let max = self.d_vec.max();
        if min <= Float::EPSILON {
            Float::INFINITY
        } else {
            (max / min).powi(2)
        }
    }

    fn median(values: &[Float]) -> Float {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }
}

impl<P, U, E> Algorithm<P, GradientFreeStatus, U, E> for CMAES
where
    P: CostFunction<U, E>,
{
    type Summary = MinimizationSummary;
    type Config = CMAESConfig;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut GradientFreeStatus,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let (_bounds, transform) = resolve_bounds_and_transform(
            &config.bounds,
            &config.transform,
            BoundsHandlingMode::TransformBounds,
        );
        self.resolved_transform = transform;
        self.mean = self.resolved_transform.to_owned_internal(&config.x0);
        self.initialize_strategy(self.mean.len(), config);
        let mut x0 = Point::from(self.mean.clone());
        x0.evaluate_transformed(problem, &self.resolved_transform, args)?;
        self.best = x0.clone();
        status.inc_n_f_evals();
        status.initialize(self.best.to_external(&self.resolved_transform).destructure());
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut GradientFreeStatus,
        args: &U,
        _config: &Self::Config,
    ) -> Result<(), E> {
        let population = self.sample_population(problem, args)?;
        status.n_f_evals += population.len();
        if population[0].point < self.best {
            self.best = population[0].point.clone();
        }
        self.update_distribution(&population);
        status.set_position(self.best.to_external(&self.resolved_transform).destructure());
        status
            .set_message()
            .step_with_message(&format!("sigma = {}", self.sigma));
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &GradientFreeStatus,
        _args: &U,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        Ok(MinimizationSummary {
            x0: config.x0.clone(),
            x: status.x.clone(),
            fx: status.fx,
            bounds: config.bounds.clone(),
            cost_evals: status.n_f_evals,
            gradient_evals: 0,
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
        *self = Self::new(Some(0));
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientFreeStatus, U, E, Self::Config>
    where
        Self: Sized,
    {
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
    use crate::core::MaxSteps;
    use approx::assert_relative_eq;
    use nalgebra::dvector;
    use std::convert::Infallible;

    struct Quadratic;
    impl CostFunction<(), Infallible> for Quadratic {
        fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(x.dot(x))
        }
    }

    #[test]
    fn test_cmaes_quadratic() {
        let problem = Quadratic;
        let mut solver = CMAES::new(Some(0));
        let result = solver
            .process(
                &problem,
                &(),
                CMAESConfig::new([3.0, -2.0], 0.8).unwrap(),
                CMAES::default_callbacks().with_terminator(MaxSteps(120)),
            )
            .unwrap();
        assert!(result.fx < 1e-8);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_cmaes_seed_is_deterministic() {
        let problem = Quadratic;
        let config = CMAESConfig::new([3.0, -2.0], 0.8).unwrap();
        let result_a = CMAES::new(Some(7))
            .process(
                &problem,
                &(),
                config.clone(),
                CMAES::default_callbacks().with_terminator(MaxSteps(40)),
            )
            .unwrap();
        let result_b = CMAES::new(Some(7))
            .process(
                &problem,
                &(),
                config,
                CMAES::default_callbacks().with_terminator(MaxSteps(40)),
            )
            .unwrap();
        assert_eq!(result_a.fx, result_b.fx);
        assert_eq!(result_a.x, result_b.x);
    }

    #[test]
    fn test_cmaes_bounds_via_transform() {
        let problem = Quadratic;
        let mut solver = CMAES::new(Some(0));
        let result = solver
            .process(
                &problem,
                &(),
                CMAESConfig::new([1.5, -1.5], 0.4)
                    .unwrap()
                    .with_bounds([(-2.0, 2.0), (-2.0, 2.0)]),
                CMAES::default_callbacks().with_terminator(MaxSteps(120)),
            )
            .unwrap();
        assert!(result.x.iter().all(|x| (-2.0..=2.0).contains(x)));
        assert!(result.fx < 1e-8);
    }

    #[test]
    fn test_cmaes_uses_hansen_active_weights() {
        let config = CMAESConfig::new([0.0, 0.0, 0.0, 0.0], 1.0)
            .unwrap()
            .with_population_size(8)
            .unwrap();
        let mut solver = CMAES::default();
        solver.initialize_strategy(4, &config);

        let lambda = solver.lambda;
        let w_prime: Vec<Float> = (0..lambda)
            .map(|i| (((lambda + 1) as Float) / 2.0).ln() - ((i + 1) as Float).ln())
            .collect();
        let positive_sum: Float = w_prime.iter().copied().filter(|w| *w >= 0.0).sum();
        let negative_abs_sum: Float = w_prime
            .iter()
            .copied()
            .filter(|w| *w < 0.0)
            .map(Float::abs)
            .sum();
        let mu_eff_minus = 1.0
            / w_prime
                .iter()
                .copied()
                .filter(|w| *w < 0.0)
                .map(|w| (w / negative_abs_sum).powi(2))
                .sum::<Float>();
        let alpha_mu_minus = 1.0 + solver.c1 / solver.c_mu;
        let alpha_mu_eff_minus = 1.0 + 2.0 * mu_eff_minus / (solver.mu_eff + 2.0);
        let alpha_posdef =
            (1.0 - solver.c1 - solver.c_mu) / (solver.mean.len() as Float * solver.c_mu);
        let negative_scale = alpha_mu_minus.min(alpha_mu_eff_minus).min(alpha_posdef);
        let expected = DVector::from_iterator(
            lambda,
            w_prime.iter().map(|w| {
                if *w >= 0.0 {
                    *w / positive_sum
                } else {
                    negative_scale * *w / negative_abs_sum
                }
            }),
        );

        for i in 0..lambda {
            assert_relative_eq!(solver.weights[i], expected[i], epsilon = 1e-12);
        }
        assert!(solver.weights.iter().skip(solver.mu).all(|w| *w < 0.0));
    }

    #[test]
    fn test_cmaes_condition_cov_terminator_triggers() {
        let mut solver = CMAES::default();
        solver.mean = DVector::zeros(2);
        solver.d_vec = dvector![1e8, 1e-1];
        let mut status = GradientFreeStatus::default();
        let mut terminator = CMAESConditionCovTerminator::default();

        let result = terminator.check_for_termination(
            0,
            &mut solver,
            &Quadratic,
            &mut status,
            &(),
            &CMAESConfig::new([0.0, 0.0], 1.0).unwrap(),
        );

        assert!(result.is_break());
        assert!(status.message.text.contains("CONDITION COVARIANCE"));
    }

    #[test]
    fn test_cmaes_equal_fun_values_terminator_triggers_on_flat_history() {
        let mut solver = CMAES::default();
        solver.mean = DVector::zeros(2);
        solver.lambda = 4;
        let window = solver.equal_fun_values_window();
        solver.best_history = std::iter::repeat_n(1.0, window).collect();
        let mut status = GradientFreeStatus::default();
        let mut terminator = CMAESEqualFunValuesTerminator;

        let result = terminator.check_for_termination(
            0,
            &mut solver,
            &Quadratic,
            &mut status,
            &(),
            &CMAESConfig::new([0.0, 0.0], 1.0).unwrap(),
        );

        assert!(result.is_break());
        assert!(status.message.text.contains("EQUAL FUNCTION VALUES"));
    }
}
