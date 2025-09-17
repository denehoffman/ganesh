use crate::{
    core::{utils::SampleFloat, Callbacks, Point, SimulatedAnnealingSummary},
    traits::{
        algorithm::SupportsTransform, Algorithm, GenericCostFunction, Status, Terminator, Transform,
    },
    Float,
};
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

/// A temperature-activated terminator for [`SimulatedAnnealing`].
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
    pub const fn new(initial_temperature: Float, cooling_rate: Float) -> Self {
        Self {
            transform: None,
            initial_temperature,
            cooling_rate,
        }
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
        status.cost_evals += 1;

        status.current = Point { x, fx: Some(fx) };
        status.temperature *= config.cooling_rate;
        status.iteration += 1;

        let acceptance_probability = if status.current.fx < status.best.fx {
            1.0
        } else {
            let d_fx = status.current.fx_checked() - status.best.fx_checked();
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
        _args: &U,
        _config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        Ok(SimulatedAnnealingSummary {
            bounds: None,
            message: status.message.clone(),
            x0: status.initial.x.clone(),
            x: status.best.x.clone(),
            fx: status.best.fx_checked(),
            cost_evals: status.cost_evals,
            converged: status.converged,
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
        core::Bounds,
        test_functions::Rosenbrock,
        traits::{cost_function::GenericGradient, DiffOps},
        DVector,
    };
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use std::fmt::Debug;

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
            #[allow(clippy::expect_used)]
            let x_int = transform.to_owned_internal(&status.current.x);
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
                    .with_transform(&Bounds::from([(-5.0, 5.0), (-5.0, 5.0)])),
                SimulatedAnnealing::default_callbacks(),
            )
            .unwrap();
        assert_relative_eq!(result.fx, 0.0, epsilon = 0.5);
    }
}
