//! Differential-evolution minimization.

use fastrand::Rng;
use std::marker::PhantomData;

use crate::core::{
    Callbacks, EvalCounts, LinearAlgebra, Matrix, MinimizationSummary, NalgebraProvider,
    RandomScalar, RealScalar, Vector,
};
use crate::error::{GaneshError, GaneshResult};
pub use crate::traits::{
    Algorithm, CostFunction, Gradient, Status, SupportsParameterNames, Transform,
    TransformedProblem,
};
use crate::{algorithms::gradient_free::GradientFreeStatus, core::MaxSteps};

/// Compact result returned by fixed-step convenience runs.
#[derive(Clone, Debug)]
pub struct MinimizationResult<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Final parameters produced by the optimizer.
    pub x: Vector<T, B>,
    /// Final objective value.
    pub fx: T,
    /// Evaluation counts performed by the run.
    pub evals: EvalCounts,
}

impl<T: RealScalar, B: LinearAlgebra<T>> SupportsParameterNames
    for DifferentialEvolutionConfig<T, B>
{
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}
/// Differential-evolution configuration.
pub struct DifferentialEvolutionConfig<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider>
{
    /// Population size used during the run.
    population_size: usize,
    /// Differential weight `F`.
    differential_weight: T,
    /// Binomial crossover probability `CR`.
    crossover_probability: T,
    /// Half-width of the initial uniform perturbation around the starting point.
    initial_scale: T,
    /// Optional names for the optimized parameters.
    parameter_names: Option<Vec<String>>,
    /// Optional coordinate transform, including bounds.
    transform: Option<Box<dyn Transform<T, B>>>,
}

impl<T, B> Default for DifferentialEvolutionConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            population_size: 0,
            differential_weight: T::literal(0.8),
            crossover_probability: T::literal(0.9),
            initial_scale: T::one(),
            parameter_names: None,
            transform: None,
        }
    }
}

impl<T, B> DifferentialEvolutionConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Create a configuration with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the population size.
    pub fn with_population_size(mut self, population_size: usize) -> GaneshResult<Self> {
        if population_size < 4 {
            return Err(GaneshError::ConfigError(
                "Differential Evolution population size must be at least 4".to_string(),
            ));
        }
        self.population_size = population_size;
        Ok(self)
    }

    /// Set the differential mutation weight.
    pub fn with_differential_weight(mut self, value: T) -> GaneshResult<Self> {
        if !value.is_finite() || value <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Differential weight must be finite and greater than 0".to_string(),
            ));
        }
        self.differential_weight = value;
        Ok(self)
    }

    /// Set the binomial crossover probability.
    pub fn with_crossover_probability(mut self, value: T) -> GaneshResult<Self> {
        if !value.is_finite() || value < T::zero() || value > T::one() {
            return Err(GaneshError::ConfigError(
                "Crossover probability must be finite and between 0 and 1".to_string(),
            ));
        }
        self.crossover_probability = value;
        Ok(self)
    }

    /// Set the initial population half-width.
    pub fn with_initial_scale(mut self, value: T) -> GaneshResult<Self> {
        if !value.is_finite() || value <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Initial scale must be finite and greater than 0".to_string(),
            ));
        }
        self.initial_scale = value;
        Ok(self)
    }

    /// Configure a coordinate transform or smooth bounds transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: Transform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

#[derive(Clone, Debug)]
struct Candidate<T: RealScalar, B: LinearAlgebra<T>> {
    x: Vector<T, B>,
    fx: T,
}

/// `DE/rand/1/bin` differential-evolution optimizer.
#[derive(Clone, Debug)]
pub struct DifferentialEvolution<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    rng: Rng,
    population: Vec<Candidate<T, B>>,
    best: Candidate<T, B>,
    _provider: PhantomData<B>,
}

impl<T, B> Default for DifferentialEvolution<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::new(Some(0))
    }
}

impl<T, B> DifferentialEvolution<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Construct a differential-evolution optimizer with an optional seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::new, Rng::with_seed),
            population: Vec::new(),
            best: Candidate {
                x: Vector::<T, B>::zeros(0),
                fx: T::zero(),
            },
            _provider: PhantomData,
        }
    }

    fn sample_offset(&mut self, dim: usize, scale: T) -> Vector<T, B> {
        let two = T::literal(2.0);
        Vector::<T, B>::from_vec(
            (0..dim)
                .map(|_| T::random_unit(&mut self.rng).mul_add(two * scale, -scale))
                .collect(),
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

    /// Run a fixed number of differential evolution generations.
    ///
    /// # Errors
    ///
    /// Returns an error if the objective fails to evaluate.
    pub fn run_steps<P, U, E>(
        &mut self,
        problem: &P,
        args: &U,
        init: Vector<T, B>,
        config: DifferentialEvolutionConfig<T, B>,
        steps: usize,
    ) -> Result<MinimizationResult<T, B>, E>
    where
        P: CostFunction<T, B, U, E>,
    {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        let init_external = init;
        let init = transformed.to_internal(&init_external);
        let pop_size = if config.population_size == 0 {
            (10 * init.len()).max(4)
        } else {
            config.population_size.max(4)
        };
        self.population.clear();
        self.population.reserve(pop_size);

        let fx = transformed.evaluate(&init, args)?;
        self.best = Candidate {
            x: init.clone(),
            fx,
        };
        self.population.push(self.best.clone());

        for _ in 1..pop_size {
            let x = init.add(&self.sample_offset(init.len(), config.initial_scale));
            let candidate = Candidate {
                fx: transformed.evaluate(&x, args)?,
                x,
            };
            if candidate.fx < self.best.fx {
                self.best = candidate.clone();
            }
            self.population.push(candidate);
        }

        let mut evals = EvalCounts::new(self.population.len(), 0, 0);
        for _ in 0..steps {
            let snapshot = self.population.clone();
            let dim = self.best.x.len();
            for i in 0..self.population.len() {
                let [a, b, c] = self.choose_distinct_indices(snapshot.len(), i);
                let forced_index = self.rng.usize(0..dim);
                let trial_x = Vector::<T, B>::from_vec(
                    (0..dim)
                        .map(|j| {
                            if j == forced_index
                                || T::random_unit(&mut self.rng) < config.crossover_probability
                            {
                                snapshot[a].x.get(j)
                                    + (snapshot[b].x.get(j) - snapshot[c].x.get(j))
                                        * config.differential_weight
                            } else {
                                self.population[i].x.get(j)
                            }
                        })
                        .collect(),
                );
                let trial = Candidate {
                    fx: transformed.evaluate(&trial_x, args)?,
                    x: trial_x,
                };
                evals.record_f();
                if trial.fx <= self.population[i].fx {
                    self.population[i] = trial;
                    if self.population[i].fx < self.best.fx {
                        self.best = self.population[i].clone();
                    }
                }
            }
        }

        Ok(MinimizationResult {
            x: transformed.to_external(&self.best.x),
            fx: self.best.fx,
            evals,
        })
    }
}

impl<T, B, P, U, E> Algorithm<P, GradientFreeStatus<T, B>, U, E> for DifferentialEvolution<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    type Summary = MinimizationSummary<T, B>;
    type Config = DifferentialEvolutionConfig<T, B>;
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
        let init = transformed.to_internal(init);
        let population_size = if config.population_size == 0 {
            (10 * init.len()).max(4)
        } else {
            config.population_size.max(4)
        };
        self.population.clear();
        self.population.reserve(population_size);

        self.best = Candidate {
            fx: transformed.evaluate(&init, args)?,
            x: init.clone(),
        };
        self.population.push(self.best.clone());
        for _ in 1..population_size {
            let x = init.add(&self.sample_offset(init.len(), config.initial_scale));
            let candidate = Candidate {
                fx: transformed.evaluate(&x, args)?,
                x,
            };
            if candidate.fx < self.best.fx {
                self.best = candidate.clone();
            }
            self.population.push(candidate);
        }
        status.evals.record_many_f(self.population.len());
        status.initialize(transformed.to_external(&self.best.x), self.best.fx);
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
        let snapshot = self.population.clone();
        let dimension = self.best.x.len();
        for index in 0..self.population.len() {
            let [a, b, c] = self.choose_distinct_indices(snapshot.len(), index);
            let forced_index = self.rng.usize(0..dimension);
            let trial_x = Vector::from_vec(
                (0..dimension)
                    .map(|coordinate| {
                        if coordinate == forced_index
                            || T::random_unit(&mut self.rng) < config.crossover_probability
                        {
                            snapshot[a].x.get(coordinate)
                                + (snapshot[b].x.get(coordinate) - snapshot[c].x.get(coordinate))
                                    * config.differential_weight
                        } else {
                            self.population[index].x.get(coordinate)
                        }
                    })
                    .collect(),
            );
            let trial = Candidate {
                fx: transformed.evaluate(&trial_x, args)?,
                x: trial_x,
            };
            status.evals.record_f();
            if trial.fx <= self.population[index].fx {
                self.population[index] = trial;
                if self.population[index].fx < self.best.fx {
                    self.best = self.population[index].clone();
                }
            }
        }
        status.set_position(transformed.to_external(&self.best.x), self.best.fx);
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
            covariance: Matrix::identity(dimension),
        })
    }

    fn reset(&mut self) {
        self.population.clear();
        self.best = Candidate {
            x: Vector::zeros(0),
            fx: T::zero(),
        };
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientFreeStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty().with_terminator(MaxSteps::default())
    }
}
