//! Scalar- and linear-algebra-generic particle swarm optimization.

use crate::algorithms::gradient_free::GradientFreeStatus;
use crate::core::{
    Callbacks, LinearAlgebra, Matrix, MinimizationSummary, NalgebraProvider, RandomScalar, Vector,
};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{
    Algorithm, CostFunction, SupportsParameterNames, Transform, TransformedProblem,
};
use fastrand::Rng;
use std::marker::PhantomData;

/// Snapshot of one linear-algebra-generic particle and its personal best.
#[derive(Clone, Debug)]
pub struct SwarmParticle<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Current internal-coordinate position.
    pub x: Vector<T, B>,
    /// Current velocity.
    pub velocity: Vector<T, B>,
    /// Best internal-coordinate position visited by this particle.
    pub best_x: Vector<T, B>,
    /// Objective value at `best_x`.
    pub best_fx: T,
}

/// Neighborhood used for the social component of particle motion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SwarmTopology {
    /// Every particle follows the best point found by the full swarm.
    #[default]
    Global,
    /// Each particle follows the best personal best within a circular neighborhood.
    Ring {
        /// Number of neighbors on each side of the particle.
        radius: usize,
    },
}

/// Whether particles observe a fixed or incrementally updated swarm state during a step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SwarmUpdateMethod {
    /// All particles use the personal/global bests from the start of the iteration.
    #[default]
    Synchronous,
    /// Personal and global bests are updated after each particle move.
    Asynchronous,
}

/// Initial velocity distribution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SwarmVelocityInitializer<T: RandomScalar = f64> {
    /// Start every particle at rest.
    #[default]
    Zero,
    /// Draw each component uniformly from `[-scale, scale]`.
    Uniform {
        /// Half-width of the velocity distribution.
        scale: T,
    },
}

/// Initial particle-position distribution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum SwarmPositionInitializer<T: RandomScalar = f64> {
    /// Place one particle at the supplied initial point and draw the rest from a centered cloud.
    #[default]
    Centered,
    /// Draw every particle uniformly from the supplied external-coordinate bounds.
    Uniform {
        /// Inclusive lower and exclusive upper sampling endpoints for each parameter.
        bounds: Vec<(T, T)>,
    },
}

impl<T> SwarmPositionInitializer<T>
where
    T: RandomScalar,
{
    /// Construct a validated uniform external-coordinate initializer.
    ///
    /// # Errors
    /// Returns a configuration error for empty, non-finite, or unordered bounds.
    pub fn uniform<I>(bounds: I) -> GaneshResult<Self>
    where
        I: IntoIterator<Item = (T, T)>,
    {
        let bounds = bounds.into_iter().collect::<Vec<_>>();
        if bounds.is_empty()
            || bounds
                .iter()
                .any(|(lower, upper)| !lower.is_finite() || !upper.is_finite() || lower >= upper)
        {
            return Err(GaneshError::ConfigError(
                "uniform swarm initialization requires finite, ordered bounds".to_string(),
            ));
        }
        Ok(Self::Uniform { bounds })
    }
}

/// Configuration for linear-algebra-generic particle swarm optimization.
pub struct PSOConfig<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Number of particles; zero selects `max(20, 10 * dimension)`.
    particles: usize,
    /// Velocity inertia coefficient.
    inertia: T,
    /// Cognitive attraction coefficient.
    cognitive: T,
    /// Social attraction coefficient.
    social: T,
    /// Half-width of the initial cloud in internal coordinates.
    initial_scale: T,
    /// Initial particle-position distribution.
    position_initializer: SwarmPositionInitializer<T>,
    /// Social neighborhood topology.
    topology: SwarmTopology,
    /// Synchronous or asynchronous particle updates.
    update_method: SwarmUpdateMethod,
    /// Initial particle velocity distribution.
    velocity_initializer: SwarmVelocityInitializer<T>,
    /// Optional names for the optimized parameters.
    parameter_names: Option<Vec<String>>,
    /// Optional transform or smooth bounds.
    transform: Option<Box<dyn Transform<T, B>>>,
}

impl<T: RandomScalar, B: LinearAlgebra<T>> SupportsParameterNames for PSOConfig<T, B> {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

impl<T, B> Default for PSOConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            particles: 0,
            inertia: T::literal(0.8),
            cognitive: T::literal(0.1),
            social: T::literal(0.1),
            initial_scale: T::one(),
            position_initializer: SwarmPositionInitializer::default(),
            topology: SwarmTopology::default(),
            update_method: SwarmUpdateMethod::default(),
            velocity_initializer: SwarmVelocityInitializer::default(),
            parameter_names: None,
            transform: None,
        }
    }
}

impl<T, B> PSOConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Create a configuration with the default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the inertial weight.
    pub fn with_omega(mut self, value: T) -> GaneshResult<Self> {
        if value < T::zero() {
            return Err(GaneshError::ConfigError(
                "Inertial weight must be greater than 0".to_string(),
            ));
        }
        self.inertia = value;
        Ok(self)
    }

    /// Set the cognitive attraction weight.
    pub fn with_c1(mut self, value: T) -> GaneshResult<Self> {
        if value < T::zero() {
            return Err(GaneshError::ConfigError(
                "Cognitive weight must be greater than 0".to_string(),
            ));
        }
        self.cognitive = value;
        Ok(self)
    }

    /// Set the social attraction weight.
    pub fn with_c2(mut self, value: T) -> GaneshResult<Self> {
        if value < T::zero() {
            return Err(GaneshError::ConfigError(
                "Social weight must be greater than 0".to_string(),
            ));
        }
        self.social = value;
        Ok(self)
    }

    /// Set the particle count.
    pub fn with_particles(mut self, particles: usize) -> GaneshResult<Self> {
        if particles < 2 {
            return Err(GaneshError::ConfigError(
                "Particle count must be at least 2".to_string(),
            ));
        }
        self.particles = particles;
        Ok(self)
    }

    /// Set the half-width of centered initialization.
    pub fn with_initial_scale(mut self, scale: T) -> GaneshResult<Self> {
        if !scale.is_finite() || scale <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Initial scale must be finite and greater than 0".to_string(),
            ));
        }
        self.initial_scale = scale;
        Ok(self)
    }

    /// Select the social-neighborhood topology.
    pub const fn with_topology(mut self, topology: SwarmTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Select synchronous or asynchronous updates.
    pub const fn with_update_method(mut self, update_method: SwarmUpdateMethod) -> Self {
        self.update_method = update_method;
        self
    }

    /// Select the initial velocity distribution.
    pub const fn with_velocity_initializer(
        mut self,
        velocity_initializer: SwarmVelocityInitializer<T>,
    ) -> Self {
        self.velocity_initializer = velocity_initializer;
        self
    }

    /// Convert an internal-coordinate value to external coordinates.
    pub fn to_external(&self, value: &Vector<T, B>) -> Vector<T, B> {
        self.transform
            .as_deref()
            .map_or_else(|| value.clone(), |transform| transform.to_external(value))
    }

    /// Configure a coordinate transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: Transform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }

    /// Draw every initial particle uniformly from external-coordinate bounds.
    ///
    /// # Errors
    /// Returns a configuration error for empty, non-finite, or unordered bounds.
    pub fn with_uniform_initialization<I>(mut self, bounds: I) -> GaneshResult<Self>
    where
        I: IntoIterator<Item = (T, T)>,
    {
        self.position_initializer = SwarmPositionInitializer::uniform(bounds)?;
        Ok(self)
    }
}

/// Scalar- and linear-algebra-generic particle swarm optimizer.
#[derive(Clone, Debug)]
pub struct PSO<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    rng: Rng,
    swarm: Vec<SwarmParticle<T, B>>,
    global_best_x: Vector<T, B>,
    global_best_fx: T,
    _provider: PhantomData<B>,
}

impl<T, B> PSO<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Construct with an optional deterministic seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::new, Rng::with_seed),
            swarm: Vec::new(),
            global_best_x: Vector::zeros(0),
            global_best_fx: T::infinity(),
            _provider: PhantomData,
        }
    }

    /// Current swarm state, suitable for tracking from an observer.
    pub fn particles(&self) -> &[SwarmParticle<T, B>] {
        &self.swarm
    }

    fn social_best(&self, particle_index: usize, topology: SwarmTopology) -> Vector<T, B> {
        match topology {
            SwarmTopology::Global => self.global_best_x.clone(),
            SwarmTopology::Ring { radius } => {
                let count = self.swarm.len();
                let mut best_index = particle_index;
                for offset in 1..=radius.min(count.saturating_sub(1)) {
                    for candidate in [
                        (particle_index + offset) % count,
                        (particle_index + count - offset % count) % count,
                    ] {
                        if self.swarm[candidate].best_fx < self.swarm[best_index].best_fx {
                            best_index = candidate;
                        }
                    }
                }
                self.swarm[best_index].best_x.clone()
            }
        }
    }
}

impl<T, B> Default for PSO<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::new(Some(0))
    }
}

impl<T, B, P, U, E> Algorithm<P, GradientFreeStatus<T, B>, U, E> for PSO<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    type Summary = MinimizationSummary<T, B>;
    type Config = PSOConfig<T, B>;
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
        let center = transformed.to_internal(init);
        let dimension = center.len();
        let count = if config.particles == 0 {
            (10 * dimension).max(20)
        } else {
            config.particles.max(2)
        };
        if let SwarmPositionInitializer::Uniform { bounds } = &config.position_initializer {
            assert_eq!(
                bounds.len(),
                dimension,
                "uniform swarm bounds must match the initial point dimension"
            );
        }
        self.swarm.clear();
        self.global_best_fx = T::infinity();
        for particle_index in 0..count {
            let x = match &config.position_initializer {
                SwarmPositionInitializer::Centered if particle_index == 0 => center.clone(),
                SwarmPositionInitializer::Centered => Vector::from_vec(
                    (0..center.len())
                        .map(|index| {
                            center.get(index)
                                + (T::literal(2.0) * T::random_unit(&mut self.rng) - T::one())
                                    * config.initial_scale
                        })
                        .collect(),
                ),
                SwarmPositionInitializer::Uniform { bounds } => {
                    let external = Vector::from_vec(
                        bounds
                            .iter()
                            .map(|(lower, upper)| {
                                *lower + (*upper - *lower) * T::random_unit(&mut self.rng)
                            })
                            .collect(),
                    );
                    transformed.to_internal(&external)
                }
            };
            let fx = transformed.evaluate(&x, args)?;
            status.evals.record_f();
            if fx < self.global_best_fx {
                self.global_best_fx = fx;
                self.global_best_x = x.clone();
            }
            let velocity = match config.velocity_initializer {
                SwarmVelocityInitializer::Zero => Vector::zeros(dimension),
                SwarmVelocityInitializer::Uniform { scale } => Vector::from_vec(
                    (0..dimension)
                        .map(|_| {
                            (T::literal(2.0) * T::random_unit(&mut self.rng) - T::one()) * scale
                        })
                        .collect(),
                ),
            };
            self.swarm.push(SwarmParticle {
                velocity,
                best_x: x.clone(),
                best_fx: fx,
                x,
            });
        }
        status.initialize(
            transformed.to_external(&self.global_best_x),
            self.global_best_fx,
        );
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
        let social_bests: Vec<Vector<T, B>> = (0..self.swarm.len())
            .map(|index| self.social_best(index, config.topology))
            .collect();
        #[allow(clippy::needless_range_loop)]
        for particle_index in 0..self.swarm.len() {
            let social_best = &social_bests[particle_index];
            let random_scale = match config.update_method {
                SwarmUpdateMethod::Synchronous => T::one(),
                SwarmUpdateMethod::Asynchronous => T::literal(0.1),
            };
            let particle = &mut self.swarm[particle_index];
            let velocity = Vector::from_vec(
                (0..particle.x.len())
                    .map(|index| {
                        config.inertia * particle.velocity.get(index)
                            + config.cognitive
                                * T::random_unit(&mut self.rng)
                                * random_scale
                                * (particle.best_x.get(index) - particle.x.get(index))
                            + config.social
                                * T::random_unit(&mut self.rng)
                                * random_scale
                                * (social_best.get(index) - particle.x.get(index))
                    })
                    .collect(),
            );
            particle.x = particle.x.add(&velocity);
            particle.velocity = velocity;
            let fx = transformed.evaluate(&particle.x, args)?;
            status.evals.record_f();
            if fx < particle.best_fx {
                particle.best_fx = fx;
                particle.best_x = particle.x.clone();
            }
            if fx < self.global_best_fx {
                self.global_best_fx = fx;
                self.global_best_x = particle.x.clone();
            }
        }
        status.set_position(
            transformed.to_external(&self.global_best_x),
            self.global_best_fx,
        );
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
        self.swarm.clear();
        self.global_best_x = Vector::zeros(0);
        self.global_best_fx = T::infinity();
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientFreeStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MaxSteps;
    use crate::test_functions::Rosenbrock;
    use crate::traits::Bounds;

    #[test]
    fn pso_runs_f32_with_provider_bounds() {
        let bounds = Bounds::new([(-3.0_f32, 3.0), (-3.0, 3.0)]).unwrap();
        let config = PSOConfig {
            particles: 40,
            initial_scale: 1.5,
            ..PSOConfig::<f32>::default()
        }
        .with_transform(bounds);
        let mut algorithm = PSO::<f32>::new(Some(19));
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
    fn pso_supports_ring_async_updates_and_tracking_snapshots() {
        let config = PSOConfig {
            particles: 16,
            topology: SwarmTopology::Ring { radius: 2 },
            update_method: SwarmUpdateMethod::Asynchronous,
            velocity_initializer: SwarmVelocityInitializer::Uniform { scale: 0.2 },
            parameter_names: Some(vec!["x".to_string(), "y".to_string()]),
            ..PSOConfig::<f64>::default()
        };
        let mut algorithm = PSO::<f64>::new(Some(3));
        let mut status = GradientFreeStatus::default();
        let init = Vector::from_vec(vec![-1.2, 1.0]);
        algorithm
            .initialize(&Rosenbrock { n: 2 }, &mut status, &(), &init, &config)
            .unwrap();
        assert_eq!(algorithm.particles().len(), 16);
        assert!(algorithm
            .particles()
            .iter()
            .any(|particle| particle.velocity.norm() > 0.0));
        algorithm
            .step(0, &Rosenbrock { n: 2 }, &mut status, &(), &config)
            .unwrap();
    }

    #[test]
    fn uniform_initialization_has_a_stable_seeded_trajectory() {
        let config = PSOConfig {
            particles: 50,
            inertia: 0.8,
            cognitive: 0.1,
            social: 0.1,
            ..PSOConfig::default()
        }
        .with_uniform_initialization([(-20.0, 20.0), (-20.0, 20.0)])
        .unwrap();
        let mut algorithm = PSO::new(Some(0));
        let mut status = GradientFreeStatus::default();
        let init: Vector = [0.0, 0.0].into();
        algorithm
            .initialize(
                &crate::test_functions::Rastrigin { n: 2 },
                &mut status,
                &(),
                &init,
                &config,
            )
            .unwrap();

        algorithm
            .step(
                0,
                &crate::test_functions::Rastrigin { n: 2 },
                &mut status,
                &(),
                &config,
            )
            .unwrap();
        let first = &algorithm.particles()[0].x;
        assert!((first.get(0) - 3.671_882_090_729_494_2).abs() < 1e-12);
        assert!((first.get(1) + 17.716_181_580_514_643).abs() < 1e-12);
        assert!((status.x.get(0) + 2.903_712_090_627_188).abs() < 1e-12);
        assert!((status.x.get(1) - 4.481_572_884_539_613).abs() < 1e-12);

        algorithm
            .step(
                1,
                &crate::test_functions::Rastrigin { n: 2 },
                &mut status,
                &(),
                &config,
            )
            .unwrap();
        let first = &algorithm.particles()[0].x;
        assert!((first.get(0) - 2.721_346_020_556_052).abs() < 1e-12);
        assert!((first.get(1) + 14.314_951_044_754_629).abs() < 1e-12);
        assert!((status.x.get(0) - 3.290_704_952_357_777).abs() < 1e-12);
        assert!((status.x.get(1) + 3.801_592_895_954_696_5).abs() < 1e-12);
    }

    #[test]
    fn uniform_initialization_rejects_invalid_bounds() {
        assert!(SwarmPositionInitializer::<f64>::uniform([]).is_err());
        assert!(SwarmPositionInitializer::uniform([(1.0, 1.0)]).is_err());
        assert!(SwarmPositionInitializer::uniform([(f64::NEG_INFINITY, 1.0)]).is_err());
    }
}
