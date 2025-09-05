use crate::{
    algorithms::particles::{Swarm, SwarmStatus, SwarmTopology, SwarmUpdateMethod},
    core::{utils::generate_random_vector, Bounds, MinimizationSummary},
    traits::{Algorithm, Bounded, CostFunction, Status},
    DMatrix, DVector, Float,
};
use fastrand::Rng;
use std::cmp::Ordering;

/// The internal configuration struct for the [`PSO`] algorithm.
#[derive(Clone)]
pub struct PSOConfig {
    swarm: Swarm,
    bounds: Option<Bounds>,
    omega: Float,
    c1: Float,
    c2: Float,
}
impl PSOConfig {
    /// Sets the inertial weight $`\omega`$ (default = `0.8`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\omega < 0`$.
    pub fn with_omega(mut self, value: Float) -> Self {
        assert!(value >= 0.0);
        self.omega = value;
        self
    }
    /// Sets the cognitive weight $`c_1`$ which controls the particle's tendency
    /// to move towards its personal best (default = `0.1`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`c_1 < 0`$.
    pub fn with_c1(mut self, value: Float) -> Self {
        assert!(value >= 0.0);
        self.c1 = value;
        self
    }
    /// Sets the social weight $`c_2`$ which controls the particle's tendency
    /// to move towards the global (or neighborhood) best depending on the swarm [`SwarmTopology`]
    /// (default = `0.1`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`c_2 < 0`$.
    pub fn with_c2(mut self, value: Float) -> Self {
        assert!(value >= 0.0);
        self.c2 = value;
        self
    }
    /// Convenience method to configure the swarm.
    pub fn setup_swarm<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut Swarm) -> &mut Swarm,
    {
        f(&mut self.swarm);
        self
    }
}
impl Bounded for PSOConfig {
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds> {
        &mut self.bounds
    }
}
impl Default for PSOConfig {
    fn default() -> Self {
        Self {
            swarm: Swarm::default(),
            bounds: None,
            omega: 0.8,
            c1: 0.1,
            c2: 0.1,
        }
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
    config: PSOConfig,
    rng: Rng,
    dimension: usize,
}

impl PSO {
    /// Construct a new particle swarm optimizer with `n_particles` particles working in an
    /// `n_dimensions` dimensional space.
    pub fn new(dimension: usize, rng: Rng) -> Self {
        Self {
            config: PSOConfig::default(),
            rng,
            dimension,
        }
    }
    fn nbest(&self, i: usize, status: &SwarmStatus) -> DVector<Float> {
        let swarm = &status.swarm;
        match swarm.topology {
            SwarmTopology::Global => status.gbest.x.clone(),
            SwarmTopology::Ring => {
                let ind = swarm.index_of_max_in_circular_window(i, 2);
                swarm.particles[ind].best.x.clone()
            }
        }
    }
    fn update<U, E>(
        &mut self,
        status: &mut SwarmStatus,
        func: &dyn CostFunction<U, E, Input = DVector<Float>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        let swarm = &status.swarm;
        match swarm.update_method {
            SwarmUpdateMethod::Synchronous => self.update_sync(status, func, user_data),
            SwarmUpdateMethod::Asynchronous => self.update_async(status, func, user_data),
        }
    }
    fn update_sync<U, E>(
        &mut self,
        status: &mut SwarmStatus,
        func: &dyn CostFunction<U, E, Input = DVector<Float>>,
        user_data: &mut U,
    ) -> Result<(), E> {
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
            let rv1 = generate_random_vector(dim, 0.0, 0.1, &mut self.rng);
            let rv2 = generate_random_vector(dim, 0.0, 0.1, &mut self.rng);
            particle.velocity = particle.velocity.scale(self.config.omega)
                + rv1
                    .component_mul(&(&particle.best.x - &particle.position.x))
                    .scale(self.config.c1)
                + rv2
                    .component_mul(&(&nbests[i] - &particle.position.x))
                    .scale(self.config.c2);
            status.n_f_evals += particle.update_position(
                func,
                user_data,
                self.config.bounds.as_ref(),
                status.swarm.boundary_method,
            )?;
        }
        Ok(())
    }
    fn update_async<U, E>(
        &mut self,
        status: &mut SwarmStatus,
        func: &dyn CostFunction<U, E, Input = DVector<Float>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        let nbests: Vec<DVector<Float>> = (0..status.swarm.particles.len())
            .map(|i| self.nbest(i, status))
            .collect();

        for (i, particle) in status.swarm.particles.iter_mut().enumerate() {
            let rv1 = generate_random_vector(particle.position.x.len(), 0.0, 0.1, &mut self.rng);
            let rv2 = generate_random_vector(particle.position.x.len(), 0.0, 0.1, &mut self.rng);
            particle.velocity = particle.velocity.scale(self.config.omega)
                + rv1
                    .component_mul(&(&particle.best.x - &particle.position.x))
                    .scale(self.config.c1)
                + rv2
                    .component_mul(&(&nbests[i] - &particle.position.x))
                    .scale(self.config.c2);
            status.n_f_evals += particle.update_position(
                func,
                user_data,
                self.config.bounds.as_ref(),
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
    P: CostFunction<U, E, Input = DVector<Float>>,
{
    type Summary = MinimizationSummary;
    type Config = PSOConfig;
    fn initialize(
        &mut self,
        config: Self::Config,
        problem: &mut P,
        status: &mut SwarmStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.config = config;
        status.swarm = self.config.swarm.clone();
        status.swarm.initialize(
            &mut self.rng,
            self.dimension,
            self.config.bounds.as_ref(),
            problem,
            user_data,
        )?;
        status.gbest = status.swarm.particles[0].best.clone();
        for particle in &mut status.swarm.particles {
            if particle.best.total_cmp(&status.gbest) == Ordering::Less {
                status.gbest = particle.best.clone();
            }
        }
        status.update_message("Initialized");
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &mut P,
        status: &mut SwarmStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.update(status, problem, user_data)
    }

    fn summarize(
        &self,
        _current_step: usize,
        _func: &P,
        status: &SwarmStatus,
        _user_data: &U,
    ) -> Result<Self::Summary, E> {
        Ok(MinimizationSummary {
            x0: DVector::from_element(status.gbest.x.len(), 0.0),
            x: status.gbest.x.clone(),
            fx: status.gbest.fx,
            bounds: self.config.bounds.clone(),
            converged: status.converged,
            cost_evals: status.n_f_evals,
            gradient_evals: 0,
            message: status.message.clone(),
            parameter_names: None,
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
        core::{Callbacks, MaxSteps},
        test_functions::Rastrigin,
    };

    #[test]
    fn test_pso() {
        let mut problem = Rastrigin { n: 2 };
        // Create and seed a random number generator
        let mut rng = Rng::new();
        rng.seed(0);

        let tracker = TrackingSwarmObserver::new();
        let callbacks = Callbacks::empty()
            .with_terminator(MaxSteps(200))
            .with_observer(tracker);

        // Create a new Sampler
        let mut solver = PSO::new(2, rng);

        // Run the particle swarm optimizer
        let result = solver
            .process(
                &mut problem,
                &mut (),
                PSOConfig::default()
                    .with_c1(0.1)
                    .with_c2(0.1)
                    .with_omega(0.8)
                    .setup_swarm(|swarm| {
                        swarm.with_n_particles(50).with_position_initializer(
                            SwarmPositionInitializer::RandomInLimits(vec![
                                (-20.0, 20.0),
                                (-20.0, 20.0),
                            ]),
                        )
                    }),
                callbacks,
            )
            .unwrap();

        println!("{}", result);
    }
}
