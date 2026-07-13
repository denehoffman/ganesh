//! Scalar- and backend-generic particle swarm optimization.

use crate::algorithms::gradient_free::BackendGradientFreeStatus;
use crate::core::{
    BackendMinimizationSummary, Callbacks, LinearAlgebra, Matrix, MaxSteps, NalgebraBackend,
    RandomScalar, Vector,
};
use crate::traits::{Algorithm, BackendTransform, BackendTransformedProblem, CostFunction, Status};
use fastrand::Rng;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
struct Particle<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    x: Vector<T, B>,
    velocity: Vector<T, B>,
    best_x: Vector<T, B>,
    best_fx: T,
}

/// Configuration for backend-generic particle swarm optimization.
pub struct BackendPSOConfig<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Number of particles; zero selects `max(20, 10 * dimension)`.
    pub particles: usize,
    /// Velocity inertia coefficient.
    pub inertia: T,
    /// Cognitive attraction coefficient.
    pub cognitive: T,
    /// Social attraction coefficient.
    pub social: T,
    /// Half-width of the initial cloud in internal coordinates.
    pub initial_scale: T,
    /// Objective improvement tolerance.
    pub improvement_tolerance: T,
    /// Stable iterations required for convergence.
    pub patience: usize,
    /// Optional transform or smooth bounds.
    pub transform: Option<Box<dyn BackendTransform<T, B>>>,
}

impl<T, B> Default for BackendPSOConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            particles: 0,
            inertia: T::literal(0.7298),
            cognitive: T::literal(1.49618),
            social: T::literal(1.49618),
            initial_scale: T::one(),
            improvement_tolerance: T::epsilon().sqrt(),
            patience: 100,
            transform: None,
        }
    }
}

impl<T, B> BackendPSOConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Configure a coordinate transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: BackendTransform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

/// Scalar- and backend-generic particle swarm optimizer.
#[derive(Clone, Debug)]
pub struct BackendPSO<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    rng: Rng,
    swarm: Vec<Particle<T, B>>,
    global_best_x: Vector<T, B>,
    global_best_fx: T,
    previous_best_fx: T,
    stable_steps: usize,
    _backend: PhantomData<B>,
}

impl<T, B> BackendPSO<T, B>
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
            previous_best_fx: T::infinity(),
            stable_steps: 0,
            _backend: PhantomData,
        }
    }
}

impl<T, B> Default for BackendPSO<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::new(None)
    }
}

impl<T, B, P, U, E> Algorithm<P, BackendGradientFreeStatus<T, B>, U, E> for BackendPSO<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    type Summary = BackendMinimizationSummary<T, B>;
    type Config = BackendPSOConfig<T, B>;
    type Init = Vector<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut BackendGradientFreeStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        let center = transformed.to_internal(init);
        let count = if config.particles == 0 {
            (10 * center.len()).max(20)
        } else {
            config.particles.max(2)
        };
        self.swarm.clear();
        self.global_best_fx = T::infinity();
        for particle_index in 0..count {
            let x = if particle_index == 0 {
                center.clone()
            } else {
                Vector::from_vec(
                    (0..center.len())
                        .map(|index| {
                            center.get(index)
                                + (T::literal(2.0) * T::random_unit(&mut self.rng) - T::one())
                                    * config.initial_scale
                        })
                        .collect(),
                )
            };
            let fx = transformed.evaluate(&x, args)?;
            status.evals.record_f();
            if fx < self.global_best_fx {
                self.global_best_fx = fx;
                self.global_best_x = x.clone();
            }
            self.swarm.push(Particle {
                velocity: Vector::zeros(center.len()),
                best_x: x.clone(),
                best_fx: fx,
                x,
            });
        }
        self.previous_best_fx = self.global_best_fx;
        self.stable_steps = 0;
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
        status: &mut BackendGradientFreeStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        if self.stable_steps >= config.patience {
            status.set_message().succeed_with_message("SWARM CONVERGED");
            return Ok(());
        }
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        let global = self.global_best_x.clone();
        for particle in &mut self.swarm {
            let velocity = Vector::from_vec(
                (0..particle.x.len())
                    .map(|index| {
                        config.inertia * particle.velocity.get(index)
                            + config.cognitive
                                * T::random_unit(&mut self.rng)
                                * (particle.best_x.get(index) - particle.x.get(index))
                            + config.social
                                * T::random_unit(&mut self.rng)
                                * (global.get(index) - particle.x.get(index))
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
        if (self.previous_best_fx - self.global_best_fx).abs() <= config.improvement_tolerance {
            self.stable_steps += 1;
        } else {
            self.stable_steps = 0;
        }
        self.previous_best_fx = self.global_best_fx;
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
        status: &BackendGradientFreeStatus<T, B>,
        _args: &U,
        init: &Self::Init,
        _config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        let dimension = status.x.len();
        Ok(BackendMinimizationSummary {
            parameter_names: None,
            message: status.message.clone(),
            x0: init.clone(),
            x: status.x.clone(),
            std: Vector::zeros(dimension),
            fx: status.fx,
            evals: status.evals,
            covariance: Matrix::identity(dimension),
        })
    }

    fn reset(&mut self) {
        self.swarm.clear();
        self.global_best_x = Vector::zeros(0);
        self.global_best_fx = T::infinity();
        self.previous_best_fx = T::infinity();
        self.stable_steps = 0;
    }

    fn default_callbacks() -> Callbacks<Self, P, BackendGradientFreeStatus<T, B>, U, E, Self::Config>
    {
        Callbacks::empty().with_terminator(MaxSteps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::Rosenbrock;
    use crate::traits::BackendBounds;

    #[test]
    fn pso_runs_f32_with_backend_bounds() {
        let bounds = BackendBounds::new([(-3.0_f32, 3.0), (-3.0, 3.0)]).unwrap();
        let config = BackendPSOConfig {
            particles: 40,
            initial_scale: 1.5,
            patience: 150,
            ..BackendPSOConfig::default()
        }
        .with_transform(bounds);
        let mut algorithm = BackendPSO::<f32>::new(Some(19));
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
}
