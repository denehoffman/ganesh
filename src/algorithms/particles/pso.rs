use crate::{
    algorithms::particles::{Swarm, SwarmStatus, SwarmTopology, SwarmUpdateMethod},
    core::{utils::generate_random_vector, Bounds, MinimizationSummary},
    error::{GaneshError, GaneshResult},
    traits::algorithm::{resolve_bounds_and_transform, BoundsHandlingMode},
    traits::{
        Algorithm, CostFunction, Status, SupportsBounds, SupportsParameterNames, SupportsTransform,
        Transform,
    },
    DMatrix, DVector, Float,
};
use fastrand::Rng;
use std::cmp::Ordering;

/// The internal configuration struct for the [`PSO`] algorithm.
#[derive(Clone)]
pub struct PSOConfig {
    bounds: Option<Bounds>,
    bounds_handling: BoundsHandlingMode,
    parameter_names: Option<Vec<String>>,
    transform: Option<Box<dyn Transform>>,
    omega: Float,
    c1: Float,
    c2: Float,
}
impl PSOConfig {
    /// Create a new configuration with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the inertial weight $`\omega`$ (default = `0.8`).
    pub fn with_omega(mut self, value: Float) -> GaneshResult<Self> {
        if value < 0.0 {
            return Err(GaneshError::ConfigError(
                "Inertial weight must be greater than 0".to_string(),
            ));
        }
        self.omega = value;
        Ok(self)
    }
    /// Sets the cognitive weight $`c_1`$ which controls the particle's tendency
    /// to move towards its personal best (default = `0.1`).
    pub fn with_c1(mut self, value: Float) -> GaneshResult<Self> {
        if value < 0.0 {
            return Err(GaneshError::ConfigError(
                "Cognitive weight must be greater than 0".to_string(),
            ));
        }
        self.c1 = value;
        Ok(self)
    }
    /// Sets the social weight $`c_2`$ which controls the particle's tendency
    /// to move towards the global (or neighborhood) best depending on the swarm [`SwarmTopology`]
    /// (default = `0.1`).
    pub fn with_c2(mut self, value: Float) -> GaneshResult<Self> {
        if value < 0.0 {
            return Err(GaneshError::ConfigError(
                "Social weight must be greater than 0".to_string(),
            ));
        }
        self.c2 = value;
        Ok(self)
    }
    /// Set the policy used to handle configured bounds when a transform is also present.
    pub const fn with_bounds_handling(mut self, bounds_handling: BoundsHandlingMode) -> Self {
        self.bounds_handling = bounds_handling;
        self
    }
}
impl Default for PSOConfig {
    fn default() -> Self {
        Self {
            bounds: None,
            bounds_handling: BoundsHandlingMode::Auto,
            parameter_names: None,
            transform: None,
            omega: 0.8,
            c1: 0.1,
            c2: 0.1,
        }
    }
}

impl SupportsBounds for PSOConfig {
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds> {
        &mut self.bounds
    }
}
impl SupportsTransform for PSOConfig {
    fn get_transform_mut(&mut self) -> &mut Option<Box<dyn Transform>> {
        &mut self.transform
    }
}
impl SupportsParameterNames for PSOConfig {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

/// Particle Swarm Optimizer
///
/// The PSO algorithm involves an ensemble of particles which are aware of the position of all or
/// nearby particles in the swarm. The general algorithm involves updating each particle's velocity
/// as follows:
///
/// ```math
/// v_i^{t+1} = \omega v_i^t + c_1 r_{1,i}^{t+1}(p^t_i - x^t_i) + c_2 r_{2,i}^{t+1}(g^t - x^t_i)
/// ```
/// where $`r_1`$ and $`r_2`$ are uniformly distributed random vectors in $`[-1,1]`$, $`\omega`$ is
/// an inertial weight parameter, $`c_1`$ and $`c_2`$ are cognitive and social weights
/// respectively, $`p_i^t`$ is the particle's personal best position, and $`g_i^t`$ is the swarm's best
/// position (or possibly the best position of a subset of particles depending on the swarm
/// topology). See [^1] for more information.
///
/// For bounds handling, see [^2]. The only method not given there is the
/// [`SwarmBoundaryMethod::Transform`](crate::algorithms::particles::SwarmBoundaryMethod) option, which uses the typical nonlinear bounds transformation
/// supplied by this crate.
///
///
/// [^1]: [Houssein, E. H., Gad, A. G., Hussain, K., & Suganthan, P. N. (2021). Major Advances in Particle Swarm Optimization: Theory, Analysis, and Application. In Swarm and Evolutionary Computation (Vol. 63, p. 100868). Elsevier BV.](https://doi.org/10.1016/j.swevo.2021.100868)
/// [^2]: [Chu, W., Gao, X., & Sorooshian, S. (2011). Handling boundary constraints for particle swarm optimization in high-dimensional search space. In Information Sciences (Vol. 181, Issue 20, pp. 4569–4581). Elsevier BV.](https://doi.org/10.1016/j.ins.2010.11.030)
#[derive(Clone)]
pub struct PSO {
    rng: Rng,
}
impl Default for PSO {
    fn default() -> Self {
        Self::new(Some(0))
    }
}
impl PSO {
    /// Create a new Particle Swarm Optimizer with the given seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed),
        }
    }
    fn nbest(&self, i: usize, status: &SwarmStatus) -> DVector<Float> {
        let swarm = &status.swarm;
        match swarm.topology {
            SwarmTopology::Global => status.gbest.x.clone(),
            SwarmTopology::Ring => {
                let ind = swarm.index_of_min_in_circular_window(i, 2);
                swarm.particles[ind].best.x.clone()
            }
        }
    }
    fn update<U, E>(
        &mut self,
        status: &mut SwarmStatus,
        func: &dyn CostFunction<U, E>,
        args: &U,
        config: &PSOConfig,
    ) -> Result<(), E> {
        let swarm = &status.swarm;
        match swarm.update_method {
            SwarmUpdateMethod::Synchronous => self.update_sync(status, func, args, config),
            SwarmUpdateMethod::Asynchronous => self.update_async(status, func, args, config),
        }
    }
    fn update_sync<U, E>(
        &mut self,
        status: &mut SwarmStatus,
        func: &dyn CostFunction<U, E>,
        args: &U,
        config: &PSOConfig,
    ) -> Result<(), E> {
        let (bounds, transform): (Option<Bounds>, Option<Box<dyn Transform>>) =
            resolve_bounds_and_transform(&config.bounds, &config.transform, config.bounds_handling);
        for particle in &mut status.swarm.particles {
            if particle.position.total_cmp(&particle.best) == Ordering::Less {
                particle.best = particle.position.clone();
            }
            if particle.best.total_cmp(&status.gbest) == Ordering::Less {
                status.gbest = particle.best.clone();
            }
        }
        let nbests: Vec<DVector<Float>> = (0..status.swarm.particles.len())
            .map(|i| self.nbest(i, status))
            .collect();

        for (i, particle) in &mut status.swarm.particles.iter_mut().enumerate() {
            let dim = particle.position.x.len();
            let rv1 = generate_random_vector(dim, 0.0, 1.0, &mut self.rng);
            let rv2 = generate_random_vector(dim, 0.0, 1.0, &mut self.rng);
            particle.velocity = particle.velocity.scale(config.omega)
                + rv1
                    .component_mul(&(&particle.best.x - &particle.position.x))
                    .scale(config.c1)
                + rv2
                    .component_mul(&(&nbests[i] - &particle.position.x))
                    .scale(config.c2);
            status.n_f_evals += particle.update_position(
                func,
                args,
                bounds.as_ref(),
                &transform,
                status.swarm.boundary_method,
            )?;
        }
        Ok(())
    }
    fn update_async<U, E>(
        &mut self,
        status: &mut SwarmStatus,
        func: &dyn CostFunction<U, E>,
        args: &U,
        config: &PSOConfig,
    ) -> Result<(), E> {
        let (bounds, transform): (Option<Bounds>, Option<Box<dyn Transform>>) =
            resolve_bounds_and_transform(&config.bounds, &config.transform, config.bounds_handling);
        let nbests: Vec<DVector<Float>> = (0..status.swarm.particles.len())
            .map(|i| self.nbest(i, status))
            .collect();

        for (i, particle) in status.swarm.particles.iter_mut().enumerate() {
            let rv1 = generate_random_vector(particle.position.x.len(), 0.0, 0.1, &mut self.rng);
            let rv2 = generate_random_vector(particle.position.x.len(), 0.0, 0.1, &mut self.rng);
            particle.velocity = particle.velocity.scale(config.omega)
                + rv1
                    .component_mul(&(&particle.best.x - &particle.position.x))
                    .scale(config.c1)
                + rv2
                    .component_mul(&(&nbests[i] - &particle.position.x))
                    .scale(config.c2);
            status.n_f_evals += particle.update_position(
                func,
                args,
                bounds.as_ref(),
                &transform,
                status.swarm.boundary_method,
            )?;
            if particle.position.total_cmp(&particle.best) == Ordering::Less {
                particle.best = particle.position.clone();
            }
            if particle.best.total_cmp(&status.gbest) == Ordering::Less {
                status.gbest = particle.best.clone();
            }
        }
        Ok(())
    }
}

impl<P, U, E> Algorithm<P, SwarmStatus, U, E> for PSO
where
    P: CostFunction<U, E>,
{
    type Summary = MinimizationSummary;
    type Config = PSOConfig;
    type Init = Swarm;
    fn initialize(
        &mut self,
        problem: &P,
        status: &mut SwarmStatus,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let (_bounds, transform): (Option<Bounds>, Option<Box<dyn Transform>>) =
            resolve_bounds_and_transform(&config.bounds, &config.transform, config.bounds_handling);
        status.swarm = init.clone();
        status
            .swarm
            .initialize(&mut self.rng, &transform, problem, args)?;
        status.n_f_evals += status.swarm.particles.len();
        status.gbest = status.swarm.particles[0].best.clone();
        for particle in &mut status.swarm.particles {
            if particle.best.total_cmp(&status.gbest) == Ordering::Less {
                status.gbest = particle.best.clone();
            }
        }
        status.initial_gbest = status.gbest.clone();
        status.set_message().initialize();
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut SwarmStatus,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        self.update(status, problem, args, config)
    }

    fn summarize(
        &self,
        _current_step: usize,
        _func: &P,
        status: &SwarmStatus,
        _args: &U,
        _init: &Self::Init,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        Ok(MinimizationSummary {
            x0: status.initial_gbest.x.clone(),
            x: status.gbest.x.clone(),
            fx: status.gbest.fx_checked(),
            bounds: config.bounds.clone(),
            cost_evals: status.n_f_evals,
            gradient_evals: 0,
            message: status.message.clone(),
            parameter_names: config.parameter_names.clone(),
            std: DVector::from_element(status.gbest.x.len(), 0.0),
            covariance: DMatrix::identity(status.gbest.x.len(), status.gbest.x.len()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algorithms::particles::{SwarmPositionInitializer, TrackingSwarmObserver},
        core::{utils::generate_random_vector, Callbacks, MaxSteps, Point},
        test_functions::Rastrigin,
    };
    use approx::assert_relative_eq;
    use std::convert::Infallible;

    struct Quadratic;
    impl CostFunction<(), Infallible> for Quadratic {
        fn evaluate(&self, x: &DVector<Float>, _: &()) -> Result<Float, Infallible> {
            Ok(x.dot(x))
        }
    }

    #[test]
    fn test_pso() {
        let problem = Rastrigin { n: 2 };
        // Create and seed a random number generator
        let mut rng = Rng::new();
        rng.seed(0);

        let tracker = TrackingSwarmObserver::new();
        let callbacks = Callbacks::empty()
            .with_terminator(MaxSteps(200))
            .with_observer(tracker);

        // Create a new Sampler
        let mut solver = PSO::default();
        let init = Swarm::new(SwarmPositionInitializer::RandomInLimits {
            bounds: vec![(-20.0, 20.0), (-20.0, 20.0)],
            n_particles: 50,
        });
        let config = PSOConfig::default()
            .with_c1(0.1)
            .unwrap()
            .with_c2(0.1)
            .unwrap()
            .with_omega(0.8)
            .unwrap();

        // Run the particle swarm optimizer
        let result = solver
            .process(
                &problem,
                &(),
                init,
                config,
                callbacks,
            )
            .unwrap();

        println!("{}", result);
    }

    #[test]
    fn synchronous_update_uses_unit_random_coefficients() {
        let mut solver = PSO::new(Some(0));
        let particle = crate::algorithms::particles::SwarmParticle {
            position: Point {
                x: DVector::from_row_slice(&[1.0]),
                fx: Some(1.0),
            },
            velocity: DVector::from_row_slice(&[0.0]),
            best: Point {
                x: DVector::from_row_slice(&[2.0]),
                fx: Some(0.0),
            },
        };
        let mut status = SwarmStatus {
            gbest: particle.best.clone(),
            swarm: Swarm::new(SwarmPositionInitializer::Custom(Vec::new())),
            ..Default::default()
        };
        status.swarm.particles = vec![particle];

        let config = PSOConfig::default()
            .with_omega(0.0)
            .unwrap()
            .with_c1(1.0)
            .unwrap()
            .with_c2(0.0)
            .unwrap();

        let mut rng = Rng::with_seed(0);
        let expected = generate_random_vector(1, 0.0, 1.0, &mut rng);

        solver
            .update_sync(&mut status, &Quadratic, &(), &config)
            .unwrap();

        assert_relative_eq!(status.swarm.particles[0].velocity[0], expected[0]);
        assert_relative_eq!(status.swarm.particles[0].position.x[0], 1.0 + expected[0]);
    }

    #[test]
    fn transform_bounds_mode_is_selectable_for_pso() {
        let config = PSOConfig::default()
            .with_bounds([(0.0, 1.0)])
            .with_bounds_handling(BoundsHandlingMode::TransformBounds);

        assert!(matches!(
            config.bounds_handling,
            BoundsHandlingMode::TransformBounds
        ));
    }

    #[test]
    fn summary_reports_initial_eval_count_and_terminal_message() {
        let problem = Rastrigin { n: 2 };
        let callbacks = Callbacks::empty().with_terminator(MaxSteps(2));
        let mut solver = PSO::default();
        let init = Swarm::new(SwarmPositionInitializer::RandomInLimits {
            bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],
            n_particles: 8,
        });
        let config = PSOConfig::default();

        let result = solver
            .process(
                &problem,
                &(),
                init,
                config,
                callbacks,
            )
            .unwrap();

        assert!(result.cost_evals >= 8);
        assert_eq!(result.gradient_evals, 0);
        assert!(result
            .message
            .to_string()
            .contains("Maximum number of steps reached"));
    }
}
