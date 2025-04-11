/// Module containing standard particle swarm optimization.
pub mod pso;
use std::{cmp::Ordering, fmt::Display, sync::Arc};

use fastrand::Rng;
use nalgebra::DVector;
use parking_lot::RwLock;
pub use pso::PSO;
use serde::{Deserialize, Serialize};

use crate::{
    generate_random_vector_in_limits, init_ctrl_c_handler, is_ctrl_c_pressed,
    observers::SwarmObserver, reset_ctrl_c_handler, Bound, Float, Function, Point, SampleFloat,
};

/// A particle with a position, velocity, and best known position
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct Particle {
    /// The position of the particle (in unbounded space)
    pub position: Point,
    /// The velocity of the particle (in unbounded space)
    pub velocity: DVector<Float>,
    /// The best position of the particle (as measured by the minimum value of `fx`)
    pub best: Point,
}

impl Particle {
    fn new<U, E>(
        position: Point,
        velocity: DVector<Float>,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        bounds: Option<&Vec<Bound>>,
        boundary_method: BoundaryMethod,
    ) -> Result<Self, E> {
        let mut position = position;
        if matches!(boundary_method, BoundaryMethod::Transform) {
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
    fn total_cmp(&self, other: &Self) -> Ordering {
        self.best.total_cmp(&other.best)
    }
    fn update_position<U, E>(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        bounds: Option<&Vec<Bound>>,
        boundary_method: BoundaryMethod,
    ) -> Result<(), E> {
        let new_position = self.position.x.clone() + self.velocity.clone();
        if let Some(bounds) = bounds {
            match boundary_method {
                BoundaryMethod::Inf => {
                    self.position.set_position(new_position);
                    if Bound::contains_vec(bounds, &self.position.x) {
                        self.position.fx = Float::INFINITY;
                    }
                }
                BoundaryMethod::Shr => {
                    let bounds_excess = Bound::bounds_excess(bounds, &new_position);
                    self.position.set_position(new_position - bounds_excess);
                    self.position.evaluate(func, user_data)?;
                }
                BoundaryMethod::Transform => {
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
#[derive(Clone, Serialize, Deserialize)]
pub struct Swarm {
    /// The dimension of the parameter space
    pub dimension: usize,
    /// A list of the particles in the swarm
    pub particles: Vec<Particle>,
    /// The global best position found by all particles (in unbounded space)
    pub gbest: Point,
    /// An indicator of whether the swarm has converged
    pub converged: bool,
    /// A message containing information about the condition of the swarm or convergence
    pub message: String,
    /// The bounds placed on the minimization space
    pub bounds: Option<Vec<Bound>>,
    boundary_method: BoundaryMethod,
    position_initializer: SwarmPositionInitializer,
    velocity_initializer: SwarmVelocityInitializer,
}

impl Display for Swarm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let title = format!(
            "╒══════════════════════════════════════════════════════════════════════════════════════════════╕
│{:^94}│",
            "SWARM STATUS",
        );
        let status = format!(
            "╞════════════════════════════════════════════════════════════════╤═════════════════════════════╡
│ Status: {}                                        │ fval: {:+12.3E}          │",
            if self.converged {
                "Converged      "
            } else {
                "Invalid Minimum"
            },
            self.gbest.fx,
        );
        let message = format!(
            "├────────────────────────────────────────────────────────────────┴─────────────────────────────┤
│ Message: {:<83} │",
            self.message,
        );
        let header =
            "├───────╥────────────────────────────────────────────╥──────────────┬──────────────┬───────────┤
│ Par # ║ Value                                      ║       -Bound │       +Bound │ At Limit? │
├───────╫────────────────────────────────────────────╫──────────────┼──────────────┼───────────┤"
                .to_string();
        let mut res_list: Vec<String> = vec![];
        let bounds = self
            .bounds
            .clone()
            .unwrap_or_else(|| vec![Bound::NoBound; self.gbest.x.len()]);
        for (i, xi) in self.gbest.x.iter().enumerate() {
            let row = format!(
                "│ {:>5} ║ {:>+12.8E}                             ║ {:>+12.3E} │ {:>+12.3E} │ {:^9} │",
                i,
                xi,
                bounds[i].lower(),
                bounds[i].upper(),
                if bounds[i].at_bound(*xi) { "yes" } else { "" }
            );
            res_list.push(row);
        }
        let bottom = "└───────╨────────────────────────────────────────────╨──────────────┴──────────────┴───────────┘".to_string();
        let out = [title, status, message, header, res_list.join("\n"), bottom].join("\n");
        write!(f, "{}", out)
    }
}

impl Swarm {
    /// Get the global best position found by the swarm. If the boundary method is set to
    /// [`BoundaryMethod::Transform`], this will return the position in the original bounded space.
    pub fn get_best(&self) -> Point {
        if matches!(self.boundary_method, BoundaryMethod::Transform) {
            self.gbest.to_bounded(self.bounds.as_ref())
        } else {
            self.gbest.clone()
        }
    }
    /// Get list of the particles in the swarm. If the boundary method is set to
    /// [`BoundaryMethod::Transform`], this will transform the particles' coordinates to the original bounded space.
    pub fn get_particles(&self) -> Vec<Particle> {
        if matches!(self.boundary_method, BoundaryMethod::Transform) {
            self.particles
                .iter()
                .map(|p| p.to_bounded(self.bounds.as_ref()))
                .collect()
        } else {
            self.particles.clone()
        }
    }
    /// Updates the [`Swarm::message`] field.
    pub fn update_message(&mut self, message: &str) {
        self.message = message.to_string();
    }
    fn index_of_max_in_circular_window(&self, center_index: usize, window_radius: usize) -> usize {
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

/// Methods to initialize the positions of particles in a swarm.
#[derive(Clone, Serialize, Deserialize)]
pub enum SwarmPositionInitializer {
    /// Start all particles at the origin
    Zero {
        /// The number of particles
        n_particles: usize,
        /// The dimension of the parameter space
        n_dimensions: usize,
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
impl SwarmPositionInitializer {
    fn get_positions(&self, rng: &mut Rng) -> Vec<Point> {
        match self {
            Self::Zero {
                n_particles,
                n_dimensions,
            } => (0..*n_particles)
                .map(|_| DVector::zeros(*n_dimensions).into())
                .collect(),
            Self::RandomInLimits {
                n_particles,
                limits,
            } => (0..*n_particles)
                .map(|_| generate_random_vector_in_limits(limits, rng).into())
                .collect(),
            Self::Custom(positions) => {
                let dim = positions[0].len();
                for position in positions {
                    assert_eq!(
                        dim,
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
                let dim = limits.len();
                let mut lhs_matrix = vec![vec![0.0; dim]; *n_particles];
                for (d, limit) in limits.iter().enumerate().take(dim) {
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
    fn get_velocities(&self, n_particles: usize, dim: usize, rng: &mut Rng) -> Vec<DVector<Float>> {
        match self {
            Self::Zero => (0..n_particles).map(|_| DVector::zeros(dim)).collect(),
            Self::RandomInLimits(limits) => (0..n_particles)
                .map(|_| generate_random_vector_in_limits(limits, rng))
                .collect(),
        }
    }
}

impl Swarm {
    /// Construct a new [`Swarm`] from a [`SwarmPositionInitializer`].
    pub fn new(position_initializer: SwarmPositionInitializer) -> Self {
        Self {
            dimension: 0,
            particles: Vec::default(),
            gbest: Point::default(),
            converged: false,
            message: "Uninitialized".to_string(),
            bounds: None,
            boundary_method: BoundaryMethod::default(),
            position_initializer,
            velocity_initializer: SwarmVelocityInitializer::default(),
        }
    }
    /// Set the [`Swarm`]'s [`SwarmVelocityInitializer`].
    pub fn with_velocity_initializer(
        mut self,
        velocity_initializer: SwarmVelocityInitializer,
    ) -> Self {
        self.velocity_initializer = velocity_initializer;
        self
    }
    /// Set the [`BoundaryMethod`] for the [`Swarm`].
    pub const fn with_boundary_method(mut self, boundary_method: BoundaryMethod) -> Self {
        self.boundary_method = boundary_method;
        self
    }
    /// Initialize the swarm with a given function, bounds, and random number generator.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    pub fn initialize<U, E>(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        bounds: Option<&Vec<Bound>>,
        rng: &mut Rng,
    ) -> Result<(), E> {
        self.bounds = bounds.cloned();
        let mut particle_positions = self.position_initializer.get_positions(rng);
        let mut particle_velocities = self.velocity_initializer.get_velocities(
            particle_positions.len(),
            particle_positions[0].x.len(),
            rng,
        );
        // If we use the Transform method, the particles have been initialized in external space,
        // but we need to convert them to the unbounded internal space
        if matches!(self.boundary_method, BoundaryMethod::Transform) {
            particle_positions.iter_mut().for_each(|point| {
                *point = Bound::to_unbounded(point.x.as_slice(), self.bounds.as_ref()).into()
            });
            particle_velocities.iter_mut().for_each(|velocity| {
                *velocity = Bound::to_unbounded(velocity.as_slice(), self.bounds.as_ref())
            });
        }
        self.particles = particle_positions
            .into_iter()
            .zip(particle_velocities.into_iter())
            .map(|(position, velocity)| {
                Particle::new(
                    position,
                    velocity,
                    func,
                    user_data,
                    self.bounds.as_ref(),
                    self.boundary_method,
                )
            })
            .collect::<Result<Vec<Particle>, E>>()?;
        self.gbest = self.particles[0].best.clone();
        for particle in &mut self.particles {
            if particle.best.total_cmp(&self.gbest) == Ordering::Less {
                self.gbest = particle.best.clone();
            }
        }
        self.dimension = self.particles[0].best.x.len();
        self.update_message("Initialized");
        Ok(())
    }
}

/// Methods for handling boundaries in swarm optimizations
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub enum BoundaryMethod {
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
pub enum Topology {
    /// Each particle is connected to all others
    #[default]
    Global,
    /// Each particle is conected to two others in a chain (with joined endpoints)
    Ring,
}

/// The algorithmic method to update the swarm positions
#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub enum UpdateMethod {
    /// Update the positions and targets in separate loops (slower but sometimes more stable)
    #[default]
    Synchronous,
    /// Update the positions and targets in the same loop (faster but sometimes less stable)
    Asynchronous,
}

/// A trait representing a swarm algorithm.
///
/// This trait is implemented for the algorithms found in the [`swarms`](`super`) module, and contains
/// all the methods needed to be run by a [`SwarmMinimizer`].
pub trait SwarmAlgorithm<U, E> {
    /// Any setup work done before the main steps of the algorithm should be done here.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        swarm: &mut Swarm,
    ) -> Result<(), E>;
    /// The main "step" of an algorithm, which is repeated until termination conditions are met or
    /// the max number of steps have been taken.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn step(
        &mut self,
        i_step: usize,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        swarm: &mut Swarm,
    ) -> Result<(), E>;
    /// Runs any termination/convergence checks and returns true if the algorithm has converged.
    /// Developers should also update the internal [`Swarm`] of the algorithm here if converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn check_for_termination(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        swarm: &mut Swarm,
    ) -> Result<bool, E>;
    /// Runs any steps needed by the [`SwarmAlgorithm`] after termination or convergence. This will run
    /// regardless of whether the [`SwarmAlgorithm`] converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    #[allow(unused_variables)]
    fn postprocessing(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        swarm: &mut Swarm,
    ) -> Result<(), E> {
        Ok(())
    }
}

/// The main struct used for running [`SwarmAlgorithm`]s on [`Function`]s.
pub struct SwarmMinimizer<U, E> {
    /// The [`Swarm`] of the [`SwarmMinimizer`], usually read after minimization.
    pub swarm: Swarm,
    algorithm: Box<dyn SwarmAlgorithm<U, E>>,
    max_steps: usize,
    observers: Vec<Arc<RwLock<dyn SwarmObserver<U>>>>,
    bounds: Option<Vec<Bound>>,
}
impl<U, E> Display for SwarmMinimizer<U, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.swarm)
    }
}
impl<U, E> SwarmMinimizer<U, E> {
    const DEFAULT_MAX_STEPS: usize = 4000;
    /// Creates a new [`SwarmMinimizer`] with the given (boxed) [`SwarmAlgorithm`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new(algorithm: Box<dyn SwarmAlgorithm<U, E>>, swarm: Swarm) -> Self {
        Self {
            swarm,
            algorithm,
            max_steps: Self::DEFAULT_MAX_STEPS,
            observers: Vec::default(),
            bounds: None,
        }
    }
    // fn reset_status(&mut self) {
    //     let new_status = Swarm {
    //         bounds: self.swarm.bounds.clone(),
    //         ..Default::default()
    //     };
    //     self.swarm = swarm;
    // }
    /// Sets all [`Bound`]s of the [`SwarmMinimizer`]. This can be [`None`] for an unbounded problem, or
    /// [`Some`] [`Vec<(T, T)>`] with length equal to the number of free parameters. Individual
    /// upper or lower bounds can be unbounded by setting them equal to `T::infinity()` or
    /// `T::neg_infinity()` (e.g. `f64::INFINITY` and `f64::NEG_INFINITY`).
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bounds is not equal to the number of free
    /// parameters.
    pub fn with_bounds<I: IntoIterator<Item = B>, B: Into<Bound>>(mut self, bounds: I) -> Self {
        let bounds = bounds.into_iter().map(Into::into).collect::<Vec<Bound>>();
        assert!(bounds.len() == self.swarm.dimension);
        self.bounds = Some(bounds);
        self
    }

    /// Set the maximum number of steps to perform before failure (default: 4000).
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
    /// Adds a single [`SwarmObserver`] to the [`SwarmMinimizer`].
    pub fn with_observer(mut self, observer: Arc<RwLock<dyn SwarmObserver<U>>>) -> Self {
        self.observers.push(observer);
        self
    }
    /// Minimize the given [`Function`] starting at the point `x0`.
    ///
    /// This method first runs [`SwarmAlgorithm::initialize`], then runs [`SwarmAlgorithm::step`] in a loop,
    /// terminating if [`SwarmAlgorithm::check_for_termination`] returns `true` or if
    /// the maximum number of allowed steps is exceeded. Each step will be followed by a sequential
    /// call to all given [`SwarmObserver`]s' callback functions. Finally, regardless of convergence,
    /// [`SwarmAlgorithm::postprocessing`] is called. If the algorithm did not converge in the given
    /// step limit, the [`Swarm::message`] will be set to `"MAX EVALS"` at termination.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    ///
    /// # Panics
    ///
    /// This method will panic if the length of `x0` is not equal to the dimension of the problem
    /// (number of free parameters) or if any values of `x0` are outside the [`Bound`]s given to the
    /// [`SwarmMinimizer`].
    pub fn minimize(&mut self, func: &dyn Function<U, E>, user_data: &mut U) -> Result<(), E> {
        init_ctrl_c_handler();
        reset_ctrl_c_handler();
        // self.reset_status();
        self.algorithm
            .initialize(func, self.bounds.as_ref(), user_data, &mut self.swarm)?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step <= self.max_steps
            && !observer_termination
            && !self
                .algorithm
                .check_for_termination(func, user_data, &mut self.swarm)?
            && !is_ctrl_c_pressed()
        {
            self.algorithm
                .step(current_step, func, user_data, &mut self.swarm)?;
            current_step += 1;
            if !self.observers.is_empty() {
                for observer in self.observers.iter_mut() {
                    observer_termination =
                        observer
                            .write()
                            .callback(current_step, &mut self.swarm, user_data)
                            || observer_termination;
                }
            }
        }
        self.algorithm
            .postprocessing(func, user_data, &mut self.swarm)?;
        if current_step > self.max_steps && !self.swarm.converged {
            self.swarm.update_message("MAX EVALS");
        }
        if is_ctrl_c_pressed() {
            self.swarm.update_message("Ctrl-C Pressed");
        }
        Ok(())
    }
}
