use std::cmp::Ordering;

use fastrand::Rng;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::{
    core::{Bound, Config, Point},
    traits::{CostFunction, Status},
    utils::{generate_random_vector_in_limits, SampleFloat},
    Float,
};

/// Methods for handling boundaries in swarm optimizations
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub enum SwarmBoundaryMethod {
    #[default]
    /// Set infeasable values to +inf
    Inf,
    /// Shrink the velocity vector to place the particle on the boundary where it would cross
    Shr,
    /// Transform the function inputs nonlinearly to map the infinite plane to a bounded subset
    Transform,
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
    /// Start all particles at the origin
    Zero {
        /// The number of particles
        n_particles: usize,
    },
    /// Random distribution within the given limits for each dimension
    RandomInLimits {
        /// The number of particles
        n_particles: usize,
        /// Limits for each dimension of parameter space
        limits: Vec<(Float, Float)>,
    },
    /// Custom distribution from a given vector of positions
    Custom(Vec<DVector<Float>>),
    /// Latin Hypercube sampling
    LatinHypercube {
        /// The number of particles
        n_particles: usize,
        /// Limits for each dimension of parameter space
        limits: Vec<(Float, Float)>,
    },
}
impl Default for SwarmPositionInitializer {
    fn default() -> Self {
        Self::Zero { n_particles: 0 }
    }
}
impl SwarmPositionInitializer {
    /// Initialize the positions of the particles in the swarm
    /// using the given random number generator and dimension.
    pub fn init_positions(&self, rng: &mut Rng, dimension: usize) -> Vec<Point> {
        match self {
            Self::Zero { n_particles } => (0..*n_particles)
                .map(|_| DVector::zeros(dimension).into())
                .collect(),
            Self::RandomInLimits {
                n_particles,
                limits,
            } => (0..*n_particles)
                .map(|_| generate_random_vector_in_limits(limits, rng).into())
                .collect(),
            Self::Custom(positions) => {
                for position in positions {
                    assert_eq!(
                        dimension,
                        position.len(),
                        "All initial positions must be the same dimension!"
                    );
                }
                positions.iter().map(|p| p.clone().into()).collect()
            }
            Self::LatinHypercube {
                n_particles,
                limits,
            } => {
                let mut lhs_matrix = vec![vec![0.0; dimension]; *n_particles];
                for (d, limit) in limits.iter().enumerate().take(dimension) {
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
        n_particles: usize,
        dim: usize,
        rng: &mut Rng,
    ) -> Vec<DVector<Float>> {
        match self {
            Self::Zero => (0..n_particles).map(|_| DVector::zeros(dim)).collect(),
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
    pub position: Point,
    /// The velocity of the particle (in unbounded space)
    pub velocity: DVector<Float>,
    /// The best position of the particle (as measured by the minimum value of `fx`)
    pub best: Point,
}
impl SwarmParticle {
    /// Create a new particle with the given position, velocity, and cost function
    /// using the given random number generator and dimension.
    pub fn new<U, E>(
        position: Point,
        velocity: DVector<Float>,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        bounds: Option<&Vec<Bound>>,
        boundary_method: SwarmBoundaryMethod,
    ) -> Result<Self, E> {
        let mut position = position;
        if matches!(boundary_method, SwarmBoundaryMethod::Transform) {
            position.evaluate_bounded(func, bounds, user_data)?;
        } else {
            position.evaluate(func, user_data)?;
        }
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
    pub fn update_position<U, E>(
        &mut self,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        bounds: Option<&Vec<Bound>>,
        boundary_method: SwarmBoundaryMethod,
    ) -> Result<(), E> {
        let new_position = self.position.x.clone() + self.velocity.clone();
        if let Some(bounds) = bounds {
            match boundary_method {
                SwarmBoundaryMethod::Inf => {
                    self.position.set_position(new_position);
                    if Bound::contains_vec(bounds, &self.position.x) {
                        self.position.fx = Float::INFINITY;
                    }
                }
                SwarmBoundaryMethod::Shr => {
                    let bounds_excess = Bound::bounds_excess(bounds, &new_position);
                    self.position.set_position(new_position - bounds_excess);
                    self.position.evaluate(func, user_data)?;
                }
                SwarmBoundaryMethod::Transform => {
                    self.position.set_position(new_position);
                    self.position
                        .evaluate_bounded(func, Some(bounds), user_data)?;
                }
            }
        } else {
            self.position.set_position(new_position);
            self.position.evaluate(func, user_data)?;
        }
        Ok(())
    }
    /// Convert the particle's coordinates from the unbounded space to the bounded space using a
    /// nonlinear transformation.
    pub fn to_bounded(&self, bounds: Option<&Vec<Bound>>) -> Self {
        Self {
            position: self.position.to_bounded(bounds),
            velocity: Bound::to_bounded(self.velocity.as_slice(), bounds),
            best: self.best.to_bounded(bounds),
        }
    }
}
/// A swarm of particles used in particle swarm optimization and similar methods.
#[derive(Clone, Serialize, Deserialize, Default)]
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
    /// Create particles in the swarm using the given random number generator and [`Config`].
    /// The method uses the configured [`SwarmPositionInitializer`] and [`SwarmVelocityInitializer`] to create the particles.
    /// The [`CostFunction`] and user data are needed to evaluate the value at the particle's position.
    pub fn initialize<U, E>(
        &mut self,
        rng: &mut Rng,
        config: &Config,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
    ) -> Result<(), E> {
        let bounds = config.bounds.as_ref();
        let mut particle_positions = self
            .position_initializer
            .init_positions(rng, config.dimension);
        let mut particle_velocities = self.velocity_initializer.init_velocities(
            particle_positions.len(),
            config.dimension,
            rng,
        );
        // If we use the Transform method, the particles have been initialized in external space,
        // but we need to convert them to the unbounded internal space
        if matches!(self.boundary_method, SwarmBoundaryMethod::Transform) {
            particle_positions
                .iter_mut()
                .for_each(|point| *point = Bound::to_unbounded(point.x.as_slice(), bounds).into());
            particle_velocities
                .iter_mut()
                .for_each(|velocity| *velocity = Bound::to_unbounded(velocity.as_slice(), bounds));
        }
        self.particles = particle_positions
            .into_iter()
            .zip(particle_velocities.into_iter())
            .map(|(position, velocity)| {
                SwarmParticle::new(
                    position,
                    velocity,
                    func,
                    user_data,
                    bounds,
                    self.boundary_method,
                )
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
    /// Set the [`PSO`](super::PSO)'s [`SwarmPositionInitializer`].
    pub fn with_position_initializer(
        &mut self,
        position_initializer: SwarmPositionInitializer,
    ) -> &mut Self {
        self.position_initializer = position_initializer;
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

/// A status for particle swarm optimization and similar methods.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct SwarmStatus {
    /// The global best position found by all particles (in unbounded space)
    pub gbest: Point,
    /// An indicator of whether the swarm has converged
    pub converged: bool,
    /// A message containing information about the condition of the swarm or convergence
    pub message: String,
    /// The swarm
    pub swarm: Swarm,
}

impl SwarmStatus {
    /// Get list of the particles in the swarm. If the boundary method is set to
    /// [`SwarmBoundaryMethod::Transform`], this will transform the particles' coordinates to the original bounded space.
    pub fn get_particles(&self, bounds: Option<&Vec<Bound>>) -> Vec<SwarmParticle> {
        if matches!(self.swarm.boundary_method, SwarmBoundaryMethod::Transform) {
            self.swarm
                .particles
                .iter()
                .map(|p| p.to_bounded(bounds))
                .collect()
        } else {
            self.swarm.particles.clone()
        }
    }
    /// Get the global best position found by the swarm. If the boundary method is set to
    /// [`SwarmBoundaryMethod::Transform`], this will return the position in the original bounded space.
    pub fn get_best(
        &self,
        bounds: Option<&Vec<Bound>>,
        boundary_method: SwarmBoundaryMethod,
    ) -> Point {
        if matches!(boundary_method, SwarmBoundaryMethod::Transform) {
            self.gbest.to_bounded(bounds)
        } else {
            self.gbest.clone()
        }
    }
}

impl Status for SwarmStatus {
    fn reset(&mut self) {
        self.converged = false;
        self.message = String::new();
        self.gbest = Point::default();
        self.swarm.particles = vec![];
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn message(&self) -> &str {
        &self.message
    }
    fn update_message(&mut self, message: &str) {
        self.message = message.to_string();
    }
}
