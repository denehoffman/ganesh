//! Scalar- and backend-generic simulated annealing.

use crate::algorithms::gradient_free::BackendGradientFreeStatus;
use crate::core::utils::sample_standard_normal;
use crate::core::{
    BackendMinimizationSummary, Callbacks, LinearAlgebra, Matrix, MaxSteps, NalgebraBackend,
    RandomScalar, Vector,
};
use crate::traits::{Algorithm, BackendTransform, BackendTransformedProblem, CostFunction, Status};
use fastrand::Rng;
use std::marker::PhantomData;

/// Configuration for backend-generic simulated annealing.
pub struct BackendSimulatedAnnealingConfig<
    T: RandomScalar = f64,
    B: LinearAlgebra<T> = NalgebraBackend,
> {
    /// Initial temperature.
    pub initial_temperature: T,
    /// Multiplicative cooling factor applied after every proposal.
    pub cooling_rate: T,
    /// Proposal standard deviation in internal coordinates.
    pub proposal_scale: T,
    /// Temperature at which the run succeeds.
    pub minimum_temperature: T,
    /// Optional coordinate transform.
    pub transform: Option<Box<dyn BackendTransform<T, B>>>,
}

impl<T, B> Default for BackendSimulatedAnnealingConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            initial_temperature: T::one(),
            cooling_rate: T::literal(0.995),
            proposal_scale: T::literal(0.1),
            minimum_temperature: T::literal(1e-6),
            transform: None,
        }
    }
}

impl<T, B> BackendSimulatedAnnealingConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Configure a coordinate transform or bounds transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: BackendTransform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

/// Scalar- and backend-generic simulated-annealing optimizer.
#[derive(Clone, Debug)]
pub struct BackendSimulatedAnnealing<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    rng: Rng,
    current_x: Vector<T, B>,
    current_fx: T,
    best_x: Vector<T, B>,
    best_fx: T,
    temperature: T,
    _backend: PhantomData<B>,
}

impl<T, B> BackendSimulatedAnnealing<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Construct with an optional deterministic seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::new, Rng::with_seed),
            current_x: Vector::zeros(0),
            current_fx: T::zero(),
            best_x: Vector::zeros(0),
            best_fx: T::zero(),
            temperature: T::one(),
            _backend: PhantomData,
        }
    }
}

impl<T, B> Default for BackendSimulatedAnnealing<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::new(None)
    }
}

impl<T, B, P, U, E> Algorithm<P, BackendGradientFreeStatus<T, B>, U, E>
    for BackendSimulatedAnnealing<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    type Summary = BackendMinimizationSummary<T, B>;
    type Config = BackendSimulatedAnnealingConfig<T, B>;
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
        self.current_x = transformed.to_internal(init);
        self.current_fx = transformed.evaluate(&self.current_x, args)?;
        self.best_x = self.current_x.clone();
        self.best_fx = self.current_fx;
        self.temperature = config.initial_temperature;
        status.evals.record_f();
        status.initialize(init.clone(), self.current_fx);
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
        if self.temperature <= config.minimum_temperature {
            status
                .set_message()
                .succeed_with_message("TEMPERATURE CONVERGED");
            return Ok(());
        }
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        let proposal = Vector::from_vec(
            (0..self.current_x.len())
                .map(|index| {
                    self.current_x.get(index)
                        + config.proposal_scale * sample_standard_normal(&mut self.rng)
                })
                .collect(),
        );
        let proposal_fx = transformed.evaluate(&proposal, args)?;
        status.evals.record_f();
        let delta = proposal_fx - self.current_fx;
        let accept =
            delta <= T::zero() || T::random_unit(&mut self.rng) < (-delta / self.temperature).exp();
        if accept {
            self.current_x = proposal;
            self.current_fx = proposal_fx;
            if self.current_fx < self.best_fx {
                self.best_x = self.current_x.clone();
                self.best_fx = self.current_fx;
            }
        }
        self.temperature = self.temperature * config.cooling_rate;
        status.set_position(transformed.to_external(&self.best_x), self.best_fx);
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
        self.current_x = Vector::zeros(0);
        self.best_x = Vector::zeros(0);
        self.current_fx = T::zero();
        self.best_fx = T::zero();
        self.temperature = T::one();
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
    fn simulated_annealing_runs_f32_with_bounds() {
        let bounds = BackendBounds::new([(-3.0_f32, 3.0), (-3.0, 3.0)]).unwrap();
        let config = BackendSimulatedAnnealingConfig {
            initial_temperature: 2.0,
            cooling_rate: 0.995,
            proposal_scale: 0.15,
            minimum_temperature: 1e-5,
            ..BackendSimulatedAnnealingConfig::default()
        }
        .with_transform(bounds);
        let mut algorithm = BackendSimulatedAnnealing::<f32>::new(Some(13));
        let result = algorithm
            .process(
                &Rosenbrock { n: 2 },
                &(),
                Vector::from_vec(vec![-1.2, 1.0]),
                config,
                Callbacks::empty().with_terminator(MaxSteps(10_000)),
            )
            .unwrap();
        assert!(result.fx < 0.2);
    }
}
