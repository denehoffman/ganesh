use crate::{
    algorithms::gradient_free::GradientFreeStatus,
    core::{Bounds, Callbacks, MaxSteps, MinimizationSummary, Point},
    error::{GaneshError, GaneshResult},
    traits::algorithm::{resolve_bounds_and_transform, BoundsHandlingMode},
    traits::{
        Algorithm, CostFunction, SupportsBounds, SupportsParameterNames, SupportsTransform,
        Transform,
    },
    DMatrix, DVector, Float,
};
use fastrand::Rng;

/// Configuration for the [`DifferentialEvolution`] algorithm.
#[derive(Clone)]
pub struct DifferentialEvolutionConfig {
    population_size: Option<usize>,
    differential_weight: Float,
    crossover_probability: Float,
    bounds: Option<Bounds>,
    parameter_names: Option<Vec<String>>,
    transform: Option<Box<dyn Transform>>,
}

impl DifferentialEvolutionConfig {
    /// Create a new [`DifferentialEvolutionConfig`] with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the population size.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `population_size < 4`.
    pub fn with_population_size(mut self, population_size: usize) -> GaneshResult<Self> {
        if population_size < 4 {
            return Err(GaneshError::ConfigError(
                "Differential Evolution population size must be at least 4".to_string(),
            ));
        }
        self.population_size = Some(population_size);
        Ok(self)
    }

    /// Set the differential weight `F` used in mutation.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `differential_weight` is not finite and strictly
    /// positive.
    pub fn with_differential_weight(mut self, differential_weight: Float) -> GaneshResult<Self> {
        if !differential_weight.is_finite() || differential_weight <= 0.0 {
            return Err(GaneshError::ConfigError(
                "Differential weight must be finite and greater than 0".to_string(),
            ));
        }
        self.differential_weight = differential_weight;
        Ok(self)
    }

    /// Set the binomial crossover probability `CR`.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `crossover_probability` is not finite or not in
    /// `[0, 1]`.
    pub fn with_crossover_probability(
        mut self,
        crossover_probability: Float,
    ) -> GaneshResult<Self> {
        if !crossover_probability.is_finite() || !(0.0..=1.0).contains(&crossover_probability) {
            return Err(GaneshError::ConfigError(
                "Crossover probability must be finite and between 0 and 1".to_string(),
            ));
        }
        self.crossover_probability = crossover_probability;
        Ok(self)
    }
}

impl Default for DifferentialEvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: None,
            differential_weight: 0.8,
            crossover_probability: 0.9,
            bounds: None,
            parameter_names: None,
            transform: None,
        }
    }
}

/// Initialization payload for a [`DifferentialEvolution`] run.
#[derive(Clone)]
pub struct DifferentialEvolutionInit {
    x0: DVector<Float>,
    initial_scale: Float,
}

impl DifferentialEvolutionInit {
    /// Create a new initialization payload from the initial point.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `x0` is empty.
    pub fn new<I>(x0: I) -> GaneshResult<Self>
    where
        I: AsRef<[Float]>,
    {
        let x0 = DVector::from_row_slice(x0.as_ref());
        if x0.is_empty() {
            return Err(GaneshError::ConfigError(
                "Differential Evolution requires at least one parameter".to_string(),
            ));
        }
        Ok(Self {
            x0,
            initial_scale: 1.0,
        })
    }

    /// Set the half-width of the uniform perturbation used to initialize the population around `x0`.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if `initial_scale` is not finite and strictly positive.
    pub fn with_initial_scale(mut self, initial_scale: Float) -> GaneshResult<Self> {
        if !initial_scale.is_finite() || initial_scale <= 0.0 {
            return Err(GaneshError::ConfigError(
                "Initial scale must be finite and greater than 0".to_string(),
            ));
        }
        self.initial_scale = initial_scale;
        Ok(self)
    }
}

impl SupportsBounds for DifferentialEvolutionConfig {
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds> {
        &mut self.bounds
    }
}

impl SupportsTransform for DifferentialEvolutionConfig {
    fn get_transform_mut(&mut self) -> &mut Option<Box<dyn Transform>> {
        &mut self.transform
    }
}

impl SupportsParameterNames for DifferentialEvolutionConfig {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

/// Basic Differential Evolution optimizer using the classic `DE/rand/1/bin` strategy.
///
/// This implementation uses a fixed mutation scheme, binomial crossover, and transform-based
/// bounds handling.
///
/// [^1]: [Storn, R., & Price, K. (1997). Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. Journal of Global Optimization, 11(4), 341-359.](https://doi.org/10.1023/A:1008202821328)
#[derive(Clone)]
pub struct DifferentialEvolution {
    rng: Rng,
    population: Vec<Point<DVector<Float>>>,
    best: Point<DVector<Float>>,
    resolved_transform: Option<Box<dyn Transform>>,
}

impl Default for DifferentialEvolution {
    fn default() -> Self {
        Self::new(Some(0))
    }
}

impl DifferentialEvolution {
    /// Create a new [`DifferentialEvolution`] optimizer with an optional seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::new, Rng::with_seed),
            population: Vec::new(),
            best: Point::default(),
            resolved_transform: None,
        }
    }

    fn population_size(&self, config: &DifferentialEvolutionConfig) -> usize {
        config
            .population_size
            .unwrap_or_else(|| (10 * self.best.x.len()).max(4))
    }

    fn sample_offset(&mut self, dim: usize, scale: Float) -> DVector<Float> {
        DVector::from_iterator(
            dim,
            (0..dim).map(|_| self.rng.f64().mul_add(2.0 * scale, -scale)),
        )
    }

    fn choose_distinct_indices(&mut self, population_size: usize, target: usize) -> [usize; 3] {
        let mut a = self.rng.usize(0..population_size);
        while a == target {
            a = self.rng.usize(0..population_size);
        }
        let mut b = self.rng.usize(0..population_size);
        while b == target || b == a {
            b = self.rng.usize(0..population_size);
        }
        let mut c = self.rng.usize(0..population_size);
        while c == target || c == a || c == b {
            c = self.rng.usize(0..population_size);
        }
        [a, b, c]
    }
}

impl<P, U, E> Algorithm<P, GradientFreeStatus, U, E> for DifferentialEvolution
where
    P: CostFunction<U, E>,
{
    type Summary = MinimizationSummary;
    type Config = DifferentialEvolutionConfig;
    type Init = DifferentialEvolutionInit;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut GradientFreeStatus,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let (_bounds, transform) = resolve_bounds_and_transform(
            &config.bounds,
            &config.transform,
            BoundsHandlingMode::TransformBounds,
        );
        self.resolved_transform = transform;
        self.population.clear();

        let x0_internal = self.resolved_transform.to_owned_internal(&init.x0);
        let pop_size = self.population_size(config);
        self.population.reserve(pop_size);

        let mut x0 = Point::from(x0_internal);
        x0.evaluate_transformed(problem, &self.resolved_transform, args)?;
        self.population.push(x0.clone());
        self.best = x0;

        for _ in 1..pop_size {
            let candidate_external =
                &init.x0 + self.sample_offset(init.x0.len(), init.initial_scale);
            let candidate_x = self
                .resolved_transform
                .to_owned_internal(&candidate_external);
            let mut candidate = Point::from(candidate_x);
            candidate.evaluate_transformed(problem, &self.resolved_transform, args)?;
            if candidate < self.best {
                self.best = candidate.clone();
            }
            self.population.push(candidate);
        }

        status.n_f_evals += self.population.len();
        status.initialize(
            self.best
                .to_external(&self.resolved_transform)
                .destructure(),
        );
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut GradientFreeStatus,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let dim = self.best.x.len();
        let population_snapshot = self.population.clone();

        for i in 0..self.population.len() {
            let [a, b, c] = self.choose_distinct_indices(population_snapshot.len(), i);
            let mutant = &population_snapshot[a].x
                + (&population_snapshot[b].x - &population_snapshot[c].x)
                    .scale(config.differential_weight);
            let forced_index = self.rng.usize(0..dim);
            let trial_x = DVector::from_iterator(
                dim,
                (0..dim).map(|j| {
                    if j == forced_index || self.rng.f64() < config.crossover_probability {
                        mutant[j]
                    } else {
                        self.population[i].x[j]
                    }
                }),
            );
            let mut trial = Point::from(trial_x);
            trial.evaluate_transformed(problem, &self.resolved_transform, args)?;
            status.inc_n_f_evals();

            if trial <= self.population[i] {
                self.population[i] = trial;
                if self.population[i] < self.best {
                    self.best = self.population[i].clone();
                }
            }
        }

        status.set_position(
            self.best
                .to_external(&self.resolved_transform)
                .destructure(),
        );
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &GradientFreeStatus,
        _args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        Ok(MinimizationSummary {
            x0: init.x0.clone(),
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
        Callbacks::empty().with_terminator(MaxSteps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::convert::Infallible;

    struct Quadratic;
    impl CostFunction<(), Infallible> for Quadratic {
        fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(x.dot(x))
        }
    }

    #[test]
    fn test_de_quadratic() {
        let problem = Quadratic;
        let mut solver = DifferentialEvolution::new(Some(0));
        let init = DifferentialEvolutionInit::new([3.0, -2.0])
            .unwrap()
            .with_initial_scale(2.0)
            .unwrap();
        let config = DifferentialEvolutionConfig::default()
            .with_population_size(24)
            .unwrap();
        let result = solver
            .process(
                &problem,
                &(),
                init,
                config,
                DifferentialEvolution::default_callbacks().with_terminator(MaxSteps(150)),
            )
            .unwrap();
        assert!(result.fx < 1e-6);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-2);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-2);
    }

    #[test]
    fn test_de_seed_is_deterministic() {
        let problem = Quadratic;
        let init = DifferentialEvolutionInit::new([3.0, -2.0]).unwrap();
        let config = DifferentialEvolutionConfig::default()
            .with_population_size(20)
            .unwrap();
        let result_a = DifferentialEvolution::new(Some(7))
            .process(
                &problem,
                &(),
                init.clone(),
                config,
                DifferentialEvolution::default_callbacks().with_terminator(MaxSteps(60)),
            )
            .unwrap();
        let result_b = DifferentialEvolution::new(Some(7))
            .process(
                &problem,
                &(),
                init,
                DifferentialEvolutionConfig::default()
                    .with_population_size(20)
                    .unwrap(),
                DifferentialEvolution::default_callbacks().with_terminator(MaxSteps(60)),
            )
            .unwrap();
        assert_eq!(result_a.fx, result_b.fx);
        assert_eq!(result_a.x, result_b.x);
    }

    #[test]
    fn test_de_bounds_via_transform() {
        let problem = Quadratic;
        let mut solver = DifferentialEvolution::new(Some(0));
        let init = DifferentialEvolutionInit::new([1.5, -1.5])
            .unwrap()
            .with_initial_scale(1.5)
            .unwrap();
        let config = DifferentialEvolutionConfig::default()
            .with_population_size(20)
            .unwrap()
            .with_bounds([(-2.0, 2.0), (-2.0, 2.0)]);
        let result = solver
            .process(
                &problem,
                &(),
                init,
                config,
                DifferentialEvolution::default_callbacks().with_terminator(MaxSteps(120)),
            )
            .unwrap();
        assert!(result.x.iter().all(|x| (-2.0..=2.0).contains(x)));
        assert!(result.fx < 1e-4);
    }
}
