//! Scalar- and linear-algebra-generic simulated annealing.

use crate::algorithms::gradient_free::GradientFreeStatus;
use crate::core::utils::sample_standard_normal;
use crate::core::{
    Callbacks, LinearAlgebra, Matrix, MinimizationSummary, NalgebraProvider, RandomScalar, Vector,
};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{
    Algorithm, CostFunction, SupportsParameterNames, Terminator, Transform, TransformedProblem,
};
use dyn_clone::DynClone;
use fastrand::Rng;
use std::fmt::Debug;
use std::{marker::PhantomData, ops::ControlFlow};

/// Temperature-activated terminator for [`SimulatedAnnealing`].
#[derive(Copy, Clone)]
pub struct SimulatedAnnealingTerminator<T: RandomScalar = f64> {
    /// Minimum temperature (default = `1e-3`).
    pub min_temperature: T,
}

impl<T: RandomScalar, B: LinearAlgebra<T>> SupportsParameterNames
    for SimulatedAnnealingConfig<T, B>
{
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

impl<T: RandomScalar> Default for SimulatedAnnealingTerminator<T> {
    fn default() -> Self {
        Self {
            min_temperature: T::literal(1e-3),
        }
    }
}

/// Generic proposal generator for simulated annealing.
pub trait SimulatedAnnealingGenerator<T, B>: DynClone + Debug + Send + Sync
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Generate a proposal in internal coordinates.
    fn generate(
        &mut self,
        current: &Vector<T, B>,
        temperature: T,
        scale: T,
        rng: &mut Rng,
    ) -> Vector<T, B>;
}

dyn_clone::clone_trait_object!(<T, B> SimulatedAnnealingGenerator<T, B> where T: RandomScalar, B: LinearAlgebra<T>);

/// Independent Gaussian random-walk proposals.
#[derive(Clone, Copy, Debug, Default)]
pub struct GaussianAnnealingGenerator;

impl<T, B> SimulatedAnnealingGenerator<T, B> for GaussianAnnealingGenerator
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn generate(
        &mut self,
        current: &Vector<T, B>,
        _temperature: T,
        scale: T,
        rng: &mut Rng,
    ) -> Vector<T, B> {
        Vector::from_vec(
            (0..current.len())
                .map(|index| current.get(index) + scale * sample_standard_normal(rng))
                .collect(),
        )
    }
}

/// Configuration for linear-algebra-generic simulated annealing.
pub struct SimulatedAnnealingConfig<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Initial temperature.
    initial_temperature: T,
    /// Multiplicative cooling factor applied after every proposal.
    cooling_rate: T,
    /// Proposal standard deviation in internal coordinates.
    proposal_scale: T,
    /// Optional custom proposal generator; `None` uses the inlined Gaussian fast path.
    proposal_generator: Option<Box<dyn SimulatedAnnealingGenerator<T, B>>>,
    /// Optional names for the optimized parameters.
    parameter_names: Option<Vec<String>>,
    /// Optional coordinate transform.
    transform: Option<Box<dyn Transform<T, B>>>,
}

impl<T, B> Default for SimulatedAnnealingConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            initial_temperature: T::one(),
            cooling_rate: T::literal(0.999),
            proposal_scale: T::literal(0.1),
            proposal_generator: None,
            parameter_names: None,
            transform: None,
        }
    }
}

impl<T, B> SimulatedAnnealingConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Create a configuration with validated temperature parameters.
    pub fn new(initial_temperature: T, cooling_rate: T) -> GaneshResult<Self> {
        if initial_temperature <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Initial temperature must be greater than 0".to_string(),
            ));
        }
        if cooling_rate <= T::zero() || cooling_rate >= T::one() {
            return Err(GaneshError::ConfigError(
                "Cooling rate must be in (0, 1)".to_string(),
            ));
        }
        Ok(Self {
            initial_temperature,
            cooling_rate,
            ..Self::default()
        })
    }

    /// Set the proposal standard deviation.
    pub fn with_proposal_scale(mut self, scale: T) -> GaneshResult<Self> {
        if !scale.is_finite() || scale <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Proposal scale must be finite and greater than 0".to_string(),
            ));
        }
        self.proposal_scale = scale;
        Ok(self)
    }

    /// Replace the proposal generator.
    pub fn with_proposal_generator<G>(mut self, proposal_generator: G) -> Self
    where
        G: SimulatedAnnealingGenerator<T, B> + 'static,
    {
        self.proposal_generator = Some(Box::new(proposal_generator));
        self
    }

    /// Configure a coordinate transform or bounds transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: Transform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

impl<T, B, P, U, E>
    Terminator<
        SimulatedAnnealing<T, B>,
        P,
        GradientFreeStatus<T, B>,
        U,
        E,
        SimulatedAnnealingConfig<T, B>,
    > for SimulatedAnnealingTerminator<T>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut SimulatedAnnealing<T, B>,
        _problem: &P,
        _status: &mut GradientFreeStatus<T, B>,
        _args: &U,
        _config: &SimulatedAnnealingConfig<T, B>,
    ) -> ControlFlow<()> {
        if algorithm.temperature < self.min_temperature {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

/// Scalar- and linear-algebra-generic simulated-annealing optimizer.
#[derive(Clone, Debug)]
pub struct SimulatedAnnealing<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    rng: Rng,
    current_x: Vector<T, B>,
    current_fx: T,
    best_x: Vector<T, B>,
    best_fx: T,
    temperature: T,
    proposal_generator: Option<Box<dyn SimulatedAnnealingGenerator<T, B>>>,
    _provider: PhantomData<B>,
}

impl<T, B> SimulatedAnnealing<T, B>
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
            proposal_generator: None,
            _provider: PhantomData,
        }
    }
}

impl<T, B> Default for SimulatedAnnealing<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::new(Some(0))
    }
}

impl<T, B, P, U, E> Algorithm<P, GradientFreeStatus<T, B>, U, E> for SimulatedAnnealing<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    type Summary = MinimizationSummary<T, B>;
    type Config = SimulatedAnnealingConfig<T, B>;
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
        self.current_x = transformed.to_internal(init);
        self.current_fx = transformed.evaluate(&self.current_x, args)?;
        self.best_x = self.current_x.clone();
        self.best_fx = self.current_fx;
        self.temperature = config.initial_temperature;
        self.proposal_generator = config
            .proposal_generator
            .as_deref()
            .map(dyn_clone::clone_box);
        status.evals.record_f();
        status.initialize(init.clone(), self.current_fx);
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
        let proposal = if let Some(generator) = self.proposal_generator.as_mut() {
            generator.generate(
                &self.current_x,
                self.temperature,
                config.proposal_scale,
                &mut self.rng,
            )
        } else {
            Vector::from_vec(
                (0..self.current_x.len())
                    .map(|index| {
                        self.current_x.get(index)
                            + config.proposal_scale * sample_standard_normal(&mut self.rng)
                    })
                    .collect(),
            )
        };
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
        self.current_x = Vector::zeros(0);
        self.best_x = Vector::zeros(0);
        self.current_fx = T::zero();
        self.best_fx = T::zero();
        self.temperature = T::one();
        self.proposal_generator = None;
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientFreeStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty().with_terminator(SimulatedAnnealingTerminator::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MaxSteps;
    use crate::test_functions::Rosenbrock;
    use crate::traits::Bounds;

    #[derive(Clone, Debug)]
    struct TowardOne;

    impl<T, B> SimulatedAnnealingGenerator<T, B> for TowardOne
    where
        T: RandomScalar,
        B: LinearAlgebra<T>,
    {
        fn generate(
            &mut self,
            _current: &Vector<T, B>,
            _temperature: T,
            _scale: T,
            _rng: &mut Rng,
        ) -> Vector<T, B> {
            Vector::from_vec((0.._current.len()).map(|_| T::one()).collect())
        }
    }

    #[test]
    fn simulated_annealing_runs_f32_with_bounds() {
        let bounds = Bounds::new([(-3.0_f32, 3.0), (-3.0, 3.0)]).unwrap();
        let config = SimulatedAnnealingConfig {
            initial_temperature: 2.0,
            cooling_rate: 0.995,
            proposal_scale: 0.15,
            ..SimulatedAnnealingConfig::<f32>::default()
        }
        .with_transform(bounds);
        let mut algorithm = SimulatedAnnealing::<f32>::new(Some(13));
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

    #[test]
    fn simulated_annealing_accepts_custom_provider_generator() {
        let config = SimulatedAnnealingConfig::<f64>::default().with_proposal_generator(TowardOne);
        let result = SimulatedAnnealing::<f64>::new(Some(4))
            .process(
                &Rosenbrock { n: 2 },
                &(),
                Vector::from_vec(vec![0.0, 0.0]),
                config,
                Callbacks::empty().with_terminator(MaxSteps(20)),
            )
            .unwrap();
        assert!(result.fx < 1.0);
    }
}
