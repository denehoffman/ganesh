use crate::{
    core::{utils::SampleFloat, Bounds, Callbacks, Point, SimulatedAnnealingSummary},
    traits::{Algorithm, Bounded, CostFunction, Status, Terminator},
    Float,
};
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

/// A temperature-activated terminator for [`SimulatedAnnealing`].
pub struct SimulatedAnnealingTerminator;
impl<P, U, E, I> Terminator<SimulatedAnnealing, P, SimulatedAnnealingStatus<I>, U, E>
    for SimulatedAnnealingTerminator
where
    P: SimulatedAnnealingGenerator<U, E, Input = I>,
    I: Serialize + for<'a> Deserialize<'a> + Clone + Default,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut SimulatedAnnealing,
        _problem: &P,
        status: &mut SimulatedAnnealingStatus<I>,
        _user_data: &U,
    ) -> ControlFlow<()> {
        if status.temperature < algorithm.config.min_temperature {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// A trait for generating new points in the simulated annealing algorithm.
pub trait SimulatedAnnealingGenerator<U, E>: CostFunction<U, E> {
    /// Returns the initial state of the algorithm.
    fn initial(
        &mut self,
        bounds: Option<&Bounds>,
        status: &mut SimulatedAnnealingStatus<Self::Input>,
        user_data: &mut U,
    ) -> Self::Input;
    /// Generates a new state based on the current state, cost function and the status.
    fn generate(
        &mut self,
        bounds: Option<&Bounds>,
        status: &mut SimulatedAnnealingStatus<Self::Input>,
        user_data: &mut U,
    ) -> Self::Input;
}

/// The internal configuration struct for the [`SimulatedAnnealing`] algorithm.
pub struct SimulatedAnnealingConfig {
    bounds: Option<Bounds>,
    /// The initial temperature for the simulated annealing algorithm.
    pub initial_temperature: Float,
    /// The cooling rate for the simulated annealing algorithm.
    pub cooling_rate: Float,
    /// The minimum temperature for the simulated annealing algorithm.
    pub min_temperature: Float,
}
impl SimulatedAnnealingConfig {
    /// Create a new [`SimulatedAnnealingConfig`] with the given parameters.
    pub const fn new(
        initial_temperature: Float,
        cooling_rate: Float,
        min_temperature: Float,
    ) -> Self {
        Self {
            bounds: None,
            initial_temperature,
            cooling_rate,
            min_temperature,
        }
    }
}
impl Bounded for SimulatedAnnealingConfig {
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds> {
        &mut self.bounds
    }
}
/// A struct for the simulated annealing algorithm.
pub struct SimulatedAnnealing {
    config: SimulatedAnnealingConfig,
    rng: fastrand::Rng,
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
    pub message: String,
    /// The number of function evaluations.
    pub cost_evals: usize,
}

impl<I> Status for SimulatedAnnealingStatus<I>
where
    I: Serialize + for<'a> Deserialize<'a> + Clone + Default,
{
    fn reset(&mut self) {
        self.converged = false;
        self.message = String::new();
        self.best = Point::default();
        self.current = Point::default();
        self.iteration = 0;
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

impl SimulatedAnnealing {
    /// Creates a new instance of the simulated annealing algorithm.
    pub fn new(config: SimulatedAnnealingConfig, seed: Option<u64>) -> Self {
        Self {
            config,
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
        config: Self::Config,
        problem: &mut P,
        status: &mut SimulatedAnnealingStatus<I>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.config = config;
        let bounds = self.config.bounds.as_ref();
        let x0 = problem.initial(bounds, status, user_data);
        let fx0 = problem.evaluate(&x0, user_data)?;
        status.temperature = self.config.initial_temperature;
        status.current = Point { x: x0, fx: fx0 };
        status.initial = status.current.clone();
        status.best = status.current.clone();
        status.iteration = 0;
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &mut P,
        status: &mut SimulatedAnnealingStatus<I>,
        user_data: &mut U,
    ) -> Result<(), E> {
        let x = problem.generate(self.config.bounds.as_ref(), status, user_data);
        let fx = problem.evaluate(&x, user_data)?;
        status.cost_evals += 1;

        status.current = Point { x, fx };
        status.temperature *= self.config.cooling_rate;
        status.iteration += 1;

        let acceptance_probability = if status.current.fx < status.best.fx {
            1.0
        } else {
            let d_fx = status.current.fx - status.best.fx;
            (-d_fx / status.temperature).exp()
        };

        if acceptance_probability > self.rng.float() {
            status.best = status.current.clone();
        }
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &SimulatedAnnealingStatus<I>,
        _user_data: &U,
    ) -> Result<Self::Summary, E> {
        Ok(SimulatedAnnealingSummary {
            bounds: self.config.bounds.clone(),
            message: status.message.clone(),
            x0: status.initial.x.clone(),
            x: status.best.x.clone(),
            fx: status.best.fx,
            cost_evals: status.cost_evals,
            converged: status.converged,
        })
    }

    fn default_callbacks() -> Callbacks<Self, P, SimulatedAnnealingStatus<I>, U, E>
    where
        Self: Sized,
    {
        Callbacks::empty().with_terminator(SimulatedAnnealingTerminator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::MaxSteps,
        test_functions::Rosenbrock,
        traits::{Boundable, Gradient},
        DVector,
    };
    use approx::assert_relative_eq;
    use std::fmt::Debug;

    pub struct GradientAnnealingProblem<U, E>(Box<dyn Gradient<U, E>>);
    impl<U, E> GradientAnnealingProblem<U, E> {
        pub fn new<P>(problem: P) -> Self
        where
            P: Gradient<U, E> + 'static,
        {
            Self(Box::new(problem))
        }
    }
    impl<U, E> CostFunction<U, E> for GradientAnnealingProblem<U, E> {
        type Input = DVector<Float>;

        fn evaluate(&self, x: &Self::Input, user_data: &mut U) -> Result<Float, E> {
            self.0.evaluate(x, user_data)
        }
    }
    impl<U, E> Gradient<U, E> for GradientAnnealingProblem<U, E> {
        fn gradient(&self, x: &Self::Input, user_data: &mut U) -> Result<DVector<Float>, E> {
            self.0.gradient(x, user_data)
        }

        fn hessian(
            &self,
            x: &Self::Input,
            user_data: &mut U,
        ) -> Result<nalgebra::DMatrix<Float>, E> {
            self.0.hessian(x, user_data)
        }
    }
    impl<U, E: Debug> SimulatedAnnealingGenerator<U, E> for GradientAnnealingProblem<U, E>
    where
        Self: Gradient<U, E>,
    {
        fn generate(
            &mut self,
            bounds: Option<&Bounds>,
            status: &mut SimulatedAnnealingStatus<Self::Input>,
            user_data: &mut U,
        ) -> Self::Input {
            #[allow(clippy::expect_used)]
            let g = self
                .gradient(&status.current.x, user_data)
                .expect("This should never fail");
            let x = &status.current.x - &(status.temperature * 1e0 * g);
            x.constrain_to(bounds)
        }

        fn initial(
            &mut self,
            bounds: Option<&Bounds>,
            _status: &mut SimulatedAnnealingStatus<Self::Input>,
            _user_data: &mut U,
        ) -> Self::Input {
            #[allow(clippy::expect_used)]
            DVector::zeros(bounds.expect("This generator requires bounds to be explicitly specified, even if all parameters are unbounded!").len()).constrain_to(bounds)
        }
    }

    #[test]
    fn test_simulated_annealing() {
        let mut solver =
            SimulatedAnnealing::new(SimulatedAnnealingConfig::new(1.0, 0.999, 1e-3), Some(0));
        let mut problem = GradientAnnealingProblem::new(Rosenbrock { n: 2 });
        let result = solver
            .process(
                &mut problem,
                &mut (),
                SimulatedAnnealingConfig::new(1.0, 0.999, 1e-3)
                    .with_bounds([(-5.0, 5.0), (-5.0, 5.0)]),
                SimulatedAnnealing::default_callbacks().with_terminator(MaxSteps(5_000)),
            )
            .unwrap();
        assert_relative_eq!(result.fx, 0.0, epsilon = 0.5);
    }
}
