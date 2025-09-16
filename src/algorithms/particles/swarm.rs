use crate::{
    core::{
        utils::{generate_random_vector_in_limits, SampleFloat},
        Bounds, Point,
    },
    traits::{Boundable, CostFunction, Transform},
    DVector, Float,
};
use fastrand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// A swarm of particles used in particle swarm optimization and similar methods.
#[derive(Clone, Serialize, Deserialize)]
pub struct Swarm {
    /// A list of the particles in the swarm
    pub particles: Vec<SwarmParticle>,
    /// The topology used by the swarm
    pub topology: SwarmTopology,
    /// The update method used by the swarm
    pub update_method: SwarmUpdateMethod,
    /// The boundary method used by the swarm
    pub boundary_method: SwarmBoundaryMethod,
    /// The position initializer used by the swarm
    pub position_initializer: SwarmPositionInitializer,
    /// The velocity initializer used by the swarm
    pub velocity_initializer: SwarmVelocityInitializer,
}

impl Swarm {
    /// Create a new [`Swarm`] from a [`SwarmPositionInitializer`].
    pub fn new(position_initializer: SwarmPositionInitializer) -> Self {
        Self {
            particles: Vec::default(),
            topology: SwarmTopology::default(),
            update_method: SwarmUpdateMethod::default(),
            boundary_method: SwarmBoundaryMethod::default(),
            position_initializer,
            velocity_initializer: SwarmVelocityInitializer::default(),
        }
    }
    /// Get list of the particles in the swarm. If the boundary method is set to
    /// [`SwarmBoundaryMethod::Transform`], this will transform the particles' coordinates to the original bounded space.
    pub fn get_particles(&self) -> Vec<SwarmParticle> {
        self.particles.clone()
    }
    /// Create particles in the swarm using the given random number generator, dimension, bounds, and cost function.
    /// The method uses the configured [`SwarmPositionInitializer`] and [`SwarmVelocityInitializer`] to create the particles.
    /// The [`CostFunction`] and user data are needed to evaluate the value at the particle's position.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    pub fn initialize<U, E>(
        &mut self,
        rng: &mut Rng,
        transform: &Option<Box<dyn Transform>>,
        func: &dyn CostFunction<U, E, Input = DVector<Float>>,
        args: &U,
    ) -> Result<(), E> {
        let mut particle_positions = self.position_initializer.init_positions(rng);
        let mut particle_velocities = self.velocity_initializer.init_velocities(
            rng,
            self.position_initializer.get_dimension(),
            self.position_initializer.get_n_particles(),
        );
        // If we use the Transform method, the particles have been initialized in external space,
        // but we need to convert them to the unbounded internal space
        particle_positions
            .iter_mut()
            .for_each(|point| *point = transform.to_internal(&point.x).into_owned().into());
        particle_velocities
            .iter_mut()
            .for_each(|velocity| *velocity = transform.to_internal(velocity).into_owned());
        self.particles = particle_positions
            .into_iter()
            .zip(particle_velocities.into_iter())
            .map(|(position, velocity)| {
                SwarmParticle::new(position, velocity, func, args, transform)
            })
            .collect::<Result<Vec<SwarmParticle>, E>>()?;
        Ok(())
    }
    /// Sets the topology used by the swarm (default = [`SwarmTopology::Global`]).
    pub const fn with_topology(&mut self, value: SwarmTopology) -> &mut Self {
        self.topology = value;
        self
    }
    /// Sets the update method used by the swarm (default = [`SwarmUpdateMethod::Synchronous`]).
    pub const fn with_update_method(&mut self, value: SwarmUpdateMethod) -> &mut Self {
        self.update_method = value;
        self
    }
    /// Set the [`PSO`](super::PSO)'s [`SwarmVelocityInitializer`].
    pub fn with_velocity_initializer(
        &mut self,
        velocity_initializer: SwarmVelocityInitializer,
    ) -> &mut Self {
        self.velocity_initializer = velocity_initializer;
        self
    }
    /// Set the [`SwarmBoundaryMethod`] for the [`PSO`](super::PSO).
    pub const fn with_boundary_method(
        &mut self,
        boundary_method: SwarmBoundaryMethod,
    ) -> &mut Self {
        self.boundary_method = boundary_method;
        self
    }
    /// Get index of the particle with the maximum value in a circular window around the given index.
    ///
    /// # Panics
    ///
    /// This method panics if the window size is zero.
    pub fn index_of_max_in_circular_window(
        &self,
        center_index: usize,
        window_radius: usize,
    ) -> usize {
        let len = self.particles.len();

        let window_indices = (center_index as isize - window_radius as isize
            ..=center_index as isize + window_radius as isize)
            .map(|i| ((i % len as isize + len as isize) % len as isize) as usize);

        #[allow(clippy::expect_used)]
        window_indices
            .max_by(|&a, &b| self.particles[a].total_cmp(&self.particles[b]))
            .expect("Window has zero size!")
    }
}

/// Methods for handling boundaries in swarm optimizations
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub enum SwarmBoundaryMethod {
    #[default]
    /// Set infeasable values to +inf
    Inf,
    /// Shrink the velocity vector to place the particle on the boundary where it would cross
    Shr,
}

/// Swarm topologies which determine the flow of information
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub enum SwarmTopology {
    /// Each particle is connected to all others
    #[default]
    Global,
    /// Each particle is conected to two others in a chain (with joined endpoints)
    Ring,
}

/// The algorithmic method to update the swarm positions
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub enum SwarmUpdateMethod {
    /// Update the positions and targets in separate loops (slower but sometimes more stable)
    #[default]
    Synchronous,
    /// Update the positions and targets in the same loop (faster but sometimes less stable)
    Asynchronous,
}

/// Methods to initialize the positions of particles in a swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmPositionInitializer {
    /// Random distribution within the given limits for each dimension
    RandomInLimits {
        /// The boundaries for particle generation.
        ///
        /// Note that these need not be the same as the bounds of the problem, but probably
        /// shouldn't exceed them.
        bounds: Vec<(Float, Float)>,
        /// The number of particles to generate.
        n_particles: usize,
    },
    /// Custom distribution from a given vector of positions
    Custom(Vec<DVector<Float>>),
    /// Latin Hypercube sampling within the given limits for each dimension
    LatinHypercube {
        /// The boundaries for particle generation.
        ///
        /// Note that these need not be the same as the bounds of the problem, but probably
        /// shouldn't exceed them.
        bounds: Vec<(Float, Float)>,
        /// The number of particles to generate.
        n_particles: usize,
    },
}
impl SwarmPositionInitializer {
    fn get_dimension(&self) -> usize {
        match self {
            Self::RandomInLimits {
                bounds,
                n_particles: _,
            } => bounds.len(),
            Self::Custom(positions) => positions[0].len(),
            Self::LatinHypercube {
                bounds,
                n_particles: _,
            } => bounds.len(),
        }
    }
    fn get_n_particles(&self) -> usize {
        match self {
            Self::RandomInLimits {
                bounds: _,
                n_particles,
            } => *n_particles,
            Self::Custom(positions) => positions.len(),
            Self::LatinHypercube {
                bounds: _,
                n_particles,
            } => *n_particles,
        }
    }
    /// Initialize the positions of the particles in the swarm
    /// using the given random number generator and dimension.
    pub fn init_positions(&self, rng: &mut Rng) -> Vec<Point<DVector<Float>>> {
        match self {
            Self::RandomInLimits {
                bounds,
                n_particles,
            } => (0..*n_particles)
                .map(|_| generate_random_vector_in_limits(bounds, rng).into())
                .collect(),
            Self::Custom(positions) => positions.iter().map(|p| p.clone().into()).collect(),
            Self::LatinHypercube {
                bounds,
                n_particles,
            } => {
                let dimension = bounds.len();
                let mut lhs_matrix = vec![vec![0.0; dimension]; *n_particles];
                for (d, limit) in bounds.iter().enumerate().take(dimension) {
                    let mut bins: Vec<usize> = (0..*n_particles).collect();
                    rng.shuffle(&mut bins);
                    for (i, &bin) in bins.iter().enumerate() {
                        let (min, max) = limit;
                        let bin_size = (max - min) / *n_particles as Float;
                        let lower = min + bin as Float * bin_size;
                        let upper = lower + bin_size;
                        lhs_matrix[i][d] = rng.range(lower, upper);
                    }
                }
                lhs_matrix.into_iter().map(|coords| coords.into()).collect()
            }
        }
    }
}

/// Methods for setting the initial velocity of particles in a swarm
#[derive(Clone, Default, Serialize, Deserialize)]
pub enum SwarmVelocityInitializer {
    /// Initialize all velocities to zero
    #[default]
    Zero,
    /// Initialize velocities randomly within the given limits
    RandomInLimits(Vec<(Float, Float)>),
}
impl SwarmVelocityInitializer {
    /// Initialize the velocities of the particles in the swarm
    /// using the given random number generator and dimension.
    pub fn init_velocities(
        &self,
        rng: &mut Rng,
        dimension: usize,
        n_particles: usize,
    ) -> Vec<DVector<Float>> {
        match self {
            Self::Zero => (0..n_particles)
                .map(|_| DVector::zeros(dimension))
                .collect(),
            Self::RandomInLimits(limits) => (0..n_particles)
                .map(|_| generate_random_vector_in_limits(limits, rng))
                .collect(),
        }
    }
}

/// A particle with a position, velocity, and best known position
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct SwarmParticle {
    /// The position of the particle (in unbounded space)
    pub position: Point<DVector<Float>>,
    /// The velocity of the particle (in unbounded space)
    pub velocity: DVector<Float>,
    /// The best position of the particle (as measured by the minimum value of `fx`)
    pub best: Point<DVector<Float>>,
}
impl SwarmParticle {
    /// Create a new particle with the given position, velocity, and cost function
    /// using the given random number generator and dimension.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    pub fn new<U, E>(
        position: Point<DVector<Float>>,
        velocity: DVector<Float>,
        func: &dyn CostFunction<U, E, Input = DVector<Float>>,
        args: &U,
        transform: &Option<Box<dyn Transform>>,
    ) -> Result<Self, E> {
        let mut position = position;
        position.evaluate_transformed(func, transform, args)?;
        Ok(Self {
            position: position.clone(),
            velocity,
            best: position,
        })
    }
    /// Compare the best position to another particle
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        self.best.total_cmp(&other.best)
    }
    /// Update the particle's position and velocity using the given cost function and user data.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    pub fn update_position<U, E>(
        &mut self,
        func: &dyn CostFunction<U, E, Input = DVector<Float>>,
        args: &U,
        bounds: Option<&Bounds>,
        transform: &Option<Box<dyn Transform>>,
        boundary_method: SwarmBoundaryMethod,
    ) -> Result<usize, E> {
        let internal_bounds = bounds.map(|b| b.apply(transform));
        let position_internal = self.position.to_internal(transform);
        let velocity_internal = transform.to_internal(&self.velocity);
        let new_position_internal = position_internal.x + velocity_internal.as_ref();
        let mut evals = 0;
        if let Some(internal_bounds) = internal_bounds {
            match boundary_method {
                SwarmBoundaryMethod::Inf => {
                    self.position
                        .set_position(transform.to_external(&new_position_internal).into_owned());
                    if !new_position_internal.is_in(&internal_bounds) {
                        self.position.fx = Some(Float::INFINITY);
                    } else {
                        self.position.evaluate(func, args)?;
                        evals += 1;
                    }
                }
                SwarmBoundaryMethod::Shr => {
                    let bounds_excess = new_position_internal.excess_from(&internal_bounds);
                    self.position.set_position(
                        transform
                            .to_external(&(new_position_internal - bounds_excess))
                            .into_owned(),
                    );
                    self.position.evaluate(func, args)?;
                    evals += 1;
                }
            }
        } else {
            self.position
                .set_position(transform.to_external(&new_position_internal).into_owned());
            self.position.evaluate(func, args)?;
            evals += 1;
        }
        Ok(evals)
    }
}
