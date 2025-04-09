use std::cmp::Ordering;

use fastrand::Rng;
use nalgebra::DVector;

use super::{
    BoundaryMethod, SwarmAlgorithm, SwarmPositionInitializer, SwarmVelocityInitializer, Topology,
    UpdateMethod,
};
use crate::{generate_random_vector, Bound, Float, Function, Swarm};

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
/// [`BoundaryMethod::Transform`] option, which uses the typical nonlinear bounds transformation
/// supplied by this crate.
///
///
/// [^1]: [Houssein, E. H., Gad, A. G., Hussain, K., & Suganthan, P. N. (2021). Major Advances in Particle Swarm Optimization: Theory, Analysis, and Application. In Swarm and Evolutionary Computation (Vol. 63, p. 100868). Elsevier BV.](https://doi.org/10.1016/j.swevo.2021.100868)
/// [^2]: [Chu, W., Gao, X., & Sorooshian, S. (2011). Handling boundary constraints for particle swarm optimization in high-dimensional search space. In Information Sciences (Vol. 181, Issue 20, pp. 4569–4581). Elsevier BV.](https://doi.org/10.1016/j.ins.2010.11.030)
#[derive(Clone)]
pub struct PSO {
    omega: Float,
    c1: Float,
    c2: Float,
    rng: Rng,
    position_initializer: SwarmPositionInitializer,
    velocity_initializer: SwarmVelocityInitializer,
    topology: Topology,
    boundary_method: BoundaryMethod,
    update_method: UpdateMethod,
}

impl PSO {
    /// Construct a new particle swarm optimizer with `n_particles` particles working in an
    /// `n_dimensions` dimensional space.
    pub fn new<U, E>(position_initializer: SwarmPositionInitializer, rng: Rng) -> Self {
        Self {
            omega: 0.8,
            c1: 0.1,
            c2: 0.1,
            rng,
            position_initializer,
            velocity_initializer: SwarmVelocityInitializer::default(),
            topology: Topology::default(),
            boundary_method: BoundaryMethod::default(),
            update_method: UpdateMethod::default(),
        }
    }
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
    /// to move towards the global (or neighborhood) best depending on the swarm [`Topology`]
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
    /// Sets the method used to initialize the velocity of the swarm's particles (default = [`SwarmVelocityInitializer::Zero`]).
    pub fn with_velocity_initializer(mut self, value: SwarmVelocityInitializer) -> Self {
        self.velocity_initializer = value;
        self
    }
    /// Sets the topology used by the swarm (default = [`Topology::Global`]).
    pub const fn with_topology(mut self, value: Topology) -> Self {
        self.topology = value;
        self
    }
    /// Sets the method used to handle bounds if supplied (default = [`BoundaryMethod::Inf`]).
    pub const fn with_boundary_method(mut self, value: BoundaryMethod) -> Self {
        self.boundary_method = value;
        self
    }
    /// Sets the update method used by the swarm (default = [`UpdateMethod::Synchronous`]).
    pub const fn with_update_method(mut self, value: UpdateMethod) -> Self {
        self.update_method = value;
        self
    }
}

impl PSO {
    fn nbest(&self, i: usize, swarm: &Swarm) -> DVector<Float> {
        match self.topology {
            Topology::Global => swarm.gbest.x.clone(),
            Topology::Ring => {
                let ind = swarm.index_of_max_in_circular_window(i, 2);
                swarm.particles[ind].best.x.clone()
            }
        }
    }
    fn update<U, E>(
        &mut self,
        swarm: &mut Swarm,
        func: &dyn Function<U, E>,
        user_data: &mut U,
    ) -> Result<(), E> {
        match self.update_method {
            UpdateMethod::Synchronous => self.update_sync(swarm, func, user_data),
            UpdateMethod::Asynchronous => self.update_async(swarm, func, user_data),
        }
    }
    fn update_sync<U, E>(
        &mut self,
        swarm: &mut Swarm,
        func: &dyn Function<U, E>,
        user_data: &mut U,
    ) -> Result<(), E> {
        for particle in &mut swarm.particles {
            if particle.position.total_cmp(&particle.best) == Ordering::Less {
                particle.best = particle.position.clone();
            }
            if particle.best.total_cmp(&swarm.gbest) == Ordering::Less {
                swarm.gbest = particle.best.clone();
            }
        }
        let nbests: Vec<DVector<Float>> = (0..swarm.particles.len())
            .map(|i| self.nbest(i, swarm))
            .collect();

        for (i, particle) in &mut swarm.particles.iter_mut().enumerate() {
            let dim = particle.position.x.len();
            let rv1 = generate_random_vector(dim, 0.0, 0.1, &mut self.rng);
            let rv2 = generate_random_vector(dim, 0.0, 0.1, &mut self.rng);
            particle.velocity = particle.velocity.scale(self.omega)
                + rv1
                    .component_mul(&(&particle.best.x - &particle.position.x))
                    .scale(self.c1)
                + rv2
                    .component_mul(&(&nbests[i] - &particle.position.x))
                    .scale(self.c2);
            particle.update_position(
                func,
                user_data,
                swarm.bounds.as_ref(),
                self.boundary_method,
            )?;
        }
        Ok(())
    }
    fn update_async<U, E>(
        &mut self,
        swarm: &mut Swarm,
        func: &dyn Function<U, E>,
        user_data: &mut U,
    ) -> Result<(), E> {
        let nbests: Vec<DVector<Float>> = (0..swarm.particles.len())
            .map(|i| self.nbest(i, swarm))
            .collect();

        for (i, particle) in &mut swarm.particles.iter_mut().enumerate() {
            let dim = particle.position.x.len();
            let rv1 = generate_random_vector(dim, 0.0, 0.1, &mut self.rng);
            let rv2 = generate_random_vector(dim, 0.0, 0.1, &mut self.rng);
            particle.velocity = particle.velocity.scale(self.omega)
                + rv1
                    .component_mul(&(&particle.best.x - &particle.position.x))
                    .scale(self.c1)
                + rv2
                    .component_mul(&(&nbests[i] - &particle.position.x))
                    .scale(self.c2);
            particle.update_position(
                func,
                user_data,
                swarm.bounds.as_ref(),
                self.boundary_method,
            )?;
            if particle.position.total_cmp(&particle.best) == Ordering::Less {
                particle.best = particle.position.clone();
            }
            if particle.best.total_cmp(&swarm.gbest) == Ordering::Less {
                swarm.gbest = particle.best.clone();
            }
        }
        Ok(())
    }
}

impl<U, E> SwarmAlgorithm<U, E> for PSO {
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        swarm: &mut Swarm,
    ) -> Result<(), E> {
        *swarm = Swarm::new(
            func,
            user_data,
            bounds,
            self.position_initializer.clone(),
            self.velocity_initializer.clone(),
            self.boundary_method,
            &mut self.rng,
        )?;
        Ok(())
    }

    fn step(
        &mut self,
        _i_step: usize,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        swarm: &mut Swarm,
    ) -> Result<(), E> {
        self.update(swarm, func, user_data)
    }

    fn check_for_termination(
        &mut self,
        _func: &dyn Function<U, E>,
        _user_data: &mut U,
        _swarm: &mut Swarm,
    ) -> Result<bool, E> {
        Ok(false) // TODO: what does it mean for PSO to terminate?
    }
}
