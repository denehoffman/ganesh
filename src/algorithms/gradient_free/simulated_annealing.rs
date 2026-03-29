use crate::{
    Float,
    core::{Callbacks, Point, SimulatedAnnealingSummary, utils::SampleFloat},
    error::{GaneshError, GaneshResult},
    traits::{
        Algorithm, GenericCostFunction, Status, StatusMessage, SupportsTransform, Terminator,
        Transform,
    },
};
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

/// A temperature-activated terminator for [`SimulatedAnnealing`].
#[derive(Copy, Clone)]
pub struct SimulatedAnnealingTerminator {
    /// The minimum temperature for the simulated annealing algorithm.
    pub min_temperature: Float,
}
impl Default for SimulatedAnnealingTerminator {
    fn default() -> Self {
        Self {
            min_temperature: 1e-3,
        }
    }
}
impl<P, U, E, I>
    Terminator<SimulatedAnnealing, P, SimulatedAnnealingStatus<I>, U, E, SimulatedAnnealingConfig>
    for SimulatedAnnealingTerminator
where
    P: SimulatedAnnealingGenerator<U, E, Input = I>,
    I: Serialize + for<'a> Deserialize<'a> + Clone + Default,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        _algorithm: &mut SimulatedAnnealing,
        _problem: &P,
        status: &mut SimulatedAnnealingStatus<I>,
        _args: &U,
        _config: &SimulatedAnnealingConfig,
    ) -> ControlFlow<()> {
        if status.temperature < self.min_temperature {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// A trait for generating new points in the simulated annealing algorithm.
pub trait SimulatedAnnealingGenerator<U, E>: GenericCostFunction<U, E> {
    /// Returns the initial state of the algorithm.
    fn initial(
        &self,
        transform: &Option<Box<dyn Transform>>,
        status: &mut SimulatedAnnealingStatus<Self::Input>,
        args: &U,
    ) -> Self::Input;
    /// Generates a new state based on the current state, cost function and the status.
    fn generate(
        &self,
        transform: &Option<Box<dyn Transform>>,
        status: &mut SimulatedAnnealingStatus<Self::Input>,
        args: &U,
    ) -> Self::Input;
}

/// The internal configuration struct for the [`SimulatedAnnealing`] algorithm.
pub struct SimulatedAnnealingConfig {
    transform: Option<Box<dyn Transform>>,
    /// The initial temperature for the simulated annealing algorithm.
    pub initial_temperature: Float,
    /// The cooling rate for the simulated annealing algorithm.
    pub cooling_rate: Float,
}
impl Default for SimulatedAnnealingConfig {
    fn default() -> Self {
        Self {
            transform: None,
            initial_temperature: 1.0,
            cooling_rate: 0.999,
        }
    }
}
impl SimulatedAnnealingConfig {
    /// Create a new [`SimulatedAnnealingConfig`] with the given parameters.
    pub fn new(initial_temperature: Float, cooling_rate: Float) -> GaneshResult<Self> {
        if initial_temperature <= 0.0 {
            return Err(GaneshError::ConfigError(
                "Initial temperature must be greater than 0".to_string(),
            ));
        }
        if cooling_rate <= 0.0 || cooling_rate >= 1.0 {
            return Err(GaneshError::ConfigError(
                "Cooling rate must be in (0, 1)".to_string(),
            ));
        }
        Ok(Self {
            transform: None,
            initial_temperature,
            cooling_rate,
        })
    }
}
impl SupportsTransform for SimulatedAnnealingConfig {
    fn get_transform_mut(&mut self) -> &mut Option<Box<dyn Transform>> {
        &mut self.transform
    }
}

/// A struct for the status of the simulated annealing algorithm.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulatedAnnealingStatus<I> {
    /// The current temperature of the simulated annealing algorithm.
    pub temperature: Float,
    /// The initial point in the simulated annealing algorithm.
    pub initial: Point<I>,
    /// The best point in the simulated annealing algorithm.
    pub best: Point<I>,
    /// The current point in the simulated annealing algorithm.
    pub current: Point<I>,
    /// The number of iterations.
    pub iteration: usize,
    /// Flag indicating whether the algorithm has converged.
    pub converged: bool,
    /// The message to be displayed at the end of the algorithm.
    pub message: StatusMessage,
    /// The number of function evaluations.
    pub n_f_evals: usize,
}

impl<I> Status for SimulatedAnnealingStatus<I>
where
    I: Serialize + for<'a> Deserialize<'a> + Clone + Default,
{
    fn reset(&mut self) {
        self.temperature = Default::default();
        self.best = Default::default();
        self.current = Default::default();
        self.iteration = Default::default();
        self.converged = Default::default();
        self.message = Default::default();
        self.n_f_evals = Default::default();
    }

    fn message(&self) -> &StatusMessage {
        &self.message
    }

    fn set_message(&mut self) -> &mut StatusMessage {
        &mut self.message
    }
}

/// A struct for the simulated annealing algorithm.
pub struct SimulatedAnnealing {
    rng: fastrand::Rng,
}

impl SimulatedAnnealing {
    /// Creates a new instance of the simulated annealing algorithm.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed),
        }
    }
}

impl<P, U, E, I> Algorithm<P, SimulatedAnnealingStatus<I>, U, E> for SimulatedAnnealing
where
    P: SimulatedAnnealingGenerator<U, E, Input = I>,
    I: Serialize + for<'a> Deserialize<'a> + Clone + Default,
{
    type Summary = SimulatedAnnealingSummary<I>;
    type Config = SimulatedAnnealingConfig;

    #[allow(clippy::expect_used)]
    fn initialize(
        &mut self,
        problem: &P,
        status: &mut SimulatedAnnealingStatus<I>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let x0 = problem.initial(&config.transform, status, args);
        let fx0 = problem.evaluate_generic(&x0, args)?;
        status.temperature = config.initial_temperature;
        status.current = Point {
            x: x0,
            fx: Some(fx0),
        };
        status.initial = status.current.clone();
        status.best = status.current.clone();
        status.iteration = 0;
        status.set_message().initialize();
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut SimulatedAnnealingStatus<I>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let x = problem.generate(&config.transform, status, args);
        let fx = problem.evaluate_generic(&x, args)?;
        status.n_f_evals += 1;

        status.temperature *= config.cooling_rate;

        if fx < status.best.fx_checked() {
            status.current = Point { x, fx: Some(fx) };
            status.best = status.current.clone();
            return Ok(());
        }

        let d_fx = fx - status.current.fx_checked();
        let acceptance_probability = (-d_fx / status.temperature).exp();

        if acceptance_probability > self.rng.float() {
            status.current = Point { x, fx: Some(fx) };
        }

        status.iteration += 1;

        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &SimulatedAnnealingStatus<I>,
        _args: &U,
        _config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        Ok(SimulatedAnnealingSummary {
            bounds: None,
            message: status.message.clone(),
            x0: status.initial.x.clone(),
            x: status.best.x.clone(),
            fx: status.best.fx_checked(),
            cost_evals: status.n_f_evals,
        })
    }

    fn default_callbacks() -> Callbacks<Self, P, SimulatedAnnealingStatus<I>, U, E, Self::Config>
    where
        Self: Sized,
    {
        Callbacks::empty().with_terminator(SimulatedAnnealingTerminator::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DVector,
        core::{Bounds, Callbacks, MaxSteps},
        test_functions::Rosenbrock,
        traits::cost_function::GenericGradient,
    };
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use std::{cell::RefCell, convert::Infallible, fmt::Debug};

    pub struct GradientAnnealingProblem<U, E>(
        Box<dyn GenericGradient<U, E, Input = DVector<Float>>>,
        DVector<Float>,
    );
    impl<U, E> GradientAnnealingProblem<U, E> {
        pub fn new<P>(problem: P, x0: &[Float]) -> Self
        where
            P: GenericGradient<U, E, Input = DVector<Float>> + 'static,
        {
            Self(Box::new(problem), DVector::from_row_slice(x0))
        }
    }
    impl<U, E> GenericCostFunction<U, E> for GradientAnnealingProblem<U, E> {
        type Input = DVector<Float>;

        fn evaluate_generic(&self, x: &Self::Input, args: &U) -> Result<Float, E> {
            self.0.evaluate_generic(x, args)
        }
    }
    impl<U, E> GenericGradient<U, E> for GradientAnnealingProblem<U, E> {
        fn gradient_generic(&self, x: &Self::Input, args: &U) -> Result<DVector<Float>, E> {
            self.0.gradient_generic(x, args)
        }

        fn hessian_generic(&self, x: &Self::Input, args: &U) -> Result<DMatrix<Float>, E> {
            self.0.hessian_generic(x, args)
        }
    }
    impl<U, E: Debug> SimulatedAnnealingGenerator<U, E> for GradientAnnealingProblem<U, E>
    where
        Self: GenericGradient<U, E, Input = DVector<Float>>,
    {
        fn generate(
            &self,
            transform: &Option<Box<dyn Transform>>,
            status: &mut SimulatedAnnealingStatus<Self::Input>,
            args: &U,
        ) -> Self::Input {
            let x_int = transform.to_owned_internal(&status.current.x);
            #[allow(clippy::expect_used)]
            let g_ext = self
                .gradient_generic(&status.current.x, args)
                .expect("This should never fail");
            let g_int = transform.pullback_gradient(&x_int, &g_ext);
            let x_int_new = x_int - &(status.temperature * 1e-4 * g_int);
            transform.to_owned_external(&x_int_new)
        }

        fn initial(
            &self,
            _transform: &Option<Box<dyn Transform>>,
            _status: &mut SimulatedAnnealingStatus<Self::Input>,
            _args: &U,
        ) -> Self::Input {
            self.1.clone()
        }
    }

    #[test]
    fn test_simulated_annealing() {
        let mut solver = SimulatedAnnealing::new(Some(0));
        let problem = GradientAnnealingProblem::new(Rosenbrock { n: 2 }, &[0.0, 0.0]);
        let result = solver
            .process(
                &problem,
                &(),
                SimulatedAnnealingConfig::new(1.0, 0.999)
                    .unwrap()
                    .with_transform(&Bounds::from([(-5.0, 5.0), (-5.0, 5.0)])),
                SimulatedAnnealing::default_callbacks(),
            )
            .unwrap();
        assert_relative_eq!(result.fx, 0.0, epsilon = 0.5);
    }

    struct SequenceAnnealingProblem {
        initial: DVector<Float>,
        proposals: RefCell<Vec<DVector<Float>>>,
    }
    impl SequenceAnnealingProblem {
        fn new(initial: &[Float], proposals: Vec<&[Float]>) -> Self {
            Self {
                initial: DVector::from_row_slice(initial),
                proposals: RefCell::new(
                    proposals
                        .into_iter()
                        .map(DVector::from_row_slice)
                        .collect::<Vec<_>>(),
                ),
            }
        }
    }
    impl GenericCostFunction<(), Infallible> for SequenceAnnealingProblem {
        type Input = DVector<Float>;

        fn evaluate_generic(&self, x: &Self::Input, _: &()) -> Result<Float, Infallible> {
            Ok(x[0])
        }
    }
    impl SimulatedAnnealingGenerator<(), Infallible> for SequenceAnnealingProblem {
        fn initial(
            &self,
            _: &Option<Box<dyn Transform>>,
            _: &mut SimulatedAnnealingStatus<Self::Input>,
            _: &(),
        ) -> Self::Input {
            self.initial.clone()
        }

        fn generate(
            &self,
            _: &Option<Box<dyn Transform>>,
            _: &mut SimulatedAnnealingStatus<Self::Input>,
            _: &(),
        ) -> Self::Input {
            self.proposals.borrow_mut().remove(0)
        }
    }

    #[test]
    fn accepts_improving_proposal_even_if_not_new_best() {
        let mut solver = SimulatedAnnealing::new(Some(0));
        let problem = SequenceAnnealingProblem::new(&[2.0], vec![&[1.0]]);
        let config = SimulatedAnnealingConfig::new(0.01, 0.9).unwrap();
        let mut status = SimulatedAnnealingStatus::default();

        solver.initialize(&problem, &mut status, &(), &config).unwrap();
        status.best = Point {
            x: DVector::from_row_slice(&[0.0]),
            fx: Some(0.0),
        };
        status.current = Point {
            x: DVector::from_row_slice(&[2.0]),
            fx: Some(2.0),
        };

        solver.step(0, &problem, &mut status, &(), &config).unwrap();

        assert_relative_eq!(status.current.x[0], 1.0);
        assert_relative_eq!(status.current.fx_checked(), 1.0);
        assert_relative_eq!(status.best.x[0], 0.0);
        assert_relative_eq!(status.best.fx_checked(), 0.0);
    }

    #[test]
    fn rejected_proposal_does_not_advance_current() {
        let mut solver = SimulatedAnnealing::new(Some(0));
        let problem = SequenceAnnealingProblem::new(&[0.0], vec![&[1.0]]);
        let config = SimulatedAnnealingConfig::new(1e-6, 0.9).unwrap();
        let mut status = SimulatedAnnealingStatus::default();

        solver.initialize(&problem, &mut status, &(), &config).unwrap();
        let current_before = status.current.clone();
        let best_before = status.best.clone();

        solver.step(0, &problem, &mut status, &(), &config).unwrap();

        assert_eq!(status.current.x, current_before.x);
        assert_eq!(status.current.fx, current_before.fx);
        assert_eq!(status.best.x, best_before.x);
        assert_eq!(status.best.fx, best_before.fx);
    }

    #[test]
    fn summary_reports_nonzero_evals_and_terminal_message() {
        let mut solver = SimulatedAnnealing::new(Some(0));
        let problem = GradientAnnealingProblem::new(Rosenbrock { n: 2 }, &[0.0, 0.0]);
        let result = solver
            .process(
                &problem,
                &(),
                SimulatedAnnealingConfig::new(1.0, 0.999).unwrap(),
                Callbacks::empty().with_terminator(MaxSteps(2)),
            )
            .unwrap();

        assert!(result.cost_evals > 0);
        assert!(result.message.to_string().contains("Maximum number of steps reached"));
    }
}
