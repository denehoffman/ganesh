use crate::{
    core::{bound::Bounds, MinimizationSummary, Point},
    traits::{Algorithm, Bounded, Callback, CostFunction, Status},
    utils::SampleFloat,
    Float,
};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

/// A temperature-activated terminator for [`SimulatedAnnealing`].
pub struct SimulatedAnnealingTerminator;
impl<P, U, E> Callback<SimulatedAnnealing<U, E>, P, SimulatedAnnealingStatus, U, E>
    for SimulatedAnnealingTerminator
where
    P: CostFunction<U, E>,
{
    fn callback(
        &mut self,
        _current_step: usize,
        algorithm: &mut SimulatedAnnealing<U, E>,
        _problem: &P,
        status: &mut SimulatedAnnealingStatus,
        _user_data: &mut U,
    ) -> ControlFlow<()> {
        if status.temperature < algorithm.config.min_temperature {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// A trait for generating new points in the simulated annealing algorithm.
pub trait SimulatedAnnealingGenerator<U, E> {
    /// Generates a new point based on the current point, cost function and the status.
    fn generate(
        &mut self,
        func: &dyn CostFunction<U, E>,
        bounds: Option<&Bounds>,
        status: &mut SimulatedAnnealingStatus,
        user_data: &mut U,
    ) -> DVector<Float>;
}

/// The internal configuration struct for the [`SimulatedAnnealing`] algorithm.
pub struct SimulatedAnnealingConfig<U, E> {
    bounds: Option<Bounds>,
    /// The initial temperature for the simulated annealing algorithm.
    pub initial_temperature: Float,
    /// The cooling rate for the simulated annealing algorithm.
    pub cooling_rate: Float,
    /// The minimum temperature for the simulated annealing algorithm.
    pub min_temperature: Float,
    /// The generator for generating new points in the simulated annealing algorithm.
    pub generator: Box<dyn SimulatedAnnealingGenerator<U, E>>,
    _user_data: std::marker::PhantomData<U>,
    _error: std::marker::PhantomData<E>,
}
impl<U, E> SimulatedAnnealingConfig<U, E> {
    /// Create a new [`SimulatedAnnealingConfig`] with the given parameters.
    pub fn new<G: SimulatedAnnealingGenerator<U, E> + 'static>(
        initial_temperature: Float,
        cooling_rate: Float,
        min_temperature: Float,
        generator: G,
    ) -> Self {
        Self {
            bounds: None,
            initial_temperature,
            cooling_rate,
            min_temperature,
            generator: Box::new(generator),
            _user_data: std::marker::PhantomData,
            _error: std::marker::PhantomData,
        }
    }
}
impl<U, E> Bounded for SimulatedAnnealingConfig<U, E> {
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds> {
        &mut self.bounds
    }
}
/// A struct for the simulated annealing algorithm.
pub struct SimulatedAnnealing<U, E> {
    config: SimulatedAnnealingConfig<U, E>,
    rng: fastrand::Rng,
}

/// A struct for the status of the simulated annealing algorithm.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulatedAnnealingStatus {
    /// The current temperature of the simulated annealing algorithm.
    pub temperature: Float,
    /// The best point in the simulated annealing algorithm.
    pub best: Point,
    /// The current point in the simulated annealing algorithm.
    pub current: Point,
    /// The number of iterations.
    pub iteration: usize,
    /// Flag indicating whether the algorithm has converged.
    pub converged: bool,
    /// The message to be displayed at the end of the algorithm.
    pub message: String,
    /// The number of function evaluations.
    pub cost_evals: usize,
}

impl Status for SimulatedAnnealingStatus {
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

impl<U, E> SimulatedAnnealing<U, E> {
    /// Creates a new instance of the simulated annealing algorithm.
    pub fn new(config: SimulatedAnnealingConfig<U, E>) -> Self {
        Self {
            config,
            rng: fastrand::Rng::new(),
        }
    }
}

impl<P, U, E> Algorithm<P, SimulatedAnnealingStatus, U, E> for SimulatedAnnealing<U, E>
where
    P: CostFunction<U, E>,
{
    type Summary = MinimizationSummary;
    type Config = SimulatedAnnealingConfig<U, E>;

    #[allow(clippy::expect_used)]
    fn initialize(
        &mut self,
        config: Self::Config,
        func: &P,
        status: &mut SimulatedAnnealingStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.config = config;
        let bounds = self.config.bounds.as_ref();
        status.current.x = DVector::zeros(bounds.expect("The simulated annealing algorithm requires bounds to be explicitly specified, even if all parameters are unbounded!").len());
        let x0 = self
            .config
            .generator
            .generate(func, bounds, status, user_data);
        let fx0 = func.evaluate(x0.as_slice(), user_data)?;
        status.temperature = self.config.initial_temperature;
        status.current = Point { x: x0, fx: fx0 };
        status.best = status.current.clone();
        status.iteration = 0;
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut SimulatedAnnealingStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        let x =
            self.config
                .generator
                .generate(problem, self.config.bounds.as_ref(), status, user_data);
        let fx = problem.evaluate(x.as_slice(), user_data)?;
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

    fn postprocessing(
        &mut self,
        _problem: &P,
        _status: &mut SimulatedAnnealingStatus,
        _user_data: &mut U,
    ) -> Result<(), E> {
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &SimulatedAnnealingStatus,
        _user_data: &U,
    ) -> Result<Self::Summary, E> {
        let result = MinimizationSummary {
            x0: vec![Float::NAN; status.best.x.nrows()],
            x: status.best.x.iter().cloned().collect(),
            fx: status.best.fx,
            bounds: self.config.bounds.clone(),
            converged: status.converged,
            cost_evals: status.cost_evals,
            gradient_evals: 0,
            message: status.message.clone(),
            parameter_names: None,
            std: vec![0.0; status.best.x.len()],
        };

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use approx::assert_relative_eq;
    use nalgebra::DVector;

    use crate::{
        algorithms::gradient_free::{
            simulated_annealing::{SimulatedAnnealingConfig, SimulatedAnnealingTerminator},
            SimulatedAnnealing,
        },
        core::{bound::Boundable, Bounds},
        test_functions::Rosenbrock,
        traits::{callback::MaxSteps, Algorithm, Bounded, Callback, CostFunction},
        Float,
    };

    use super::{SimulatedAnnealingGenerator, SimulatedAnnealingStatus};

    pub struct AnnealingGenerator;
    impl<U, E: Debug> SimulatedAnnealingGenerator<U, E> for AnnealingGenerator {
        fn generate(
            &mut self,
            func: &dyn CostFunction<U, E>,
            bounds: Option<&Bounds>,
            status: &mut SimulatedAnnealingStatus,
            user_data: &mut U,
        ) -> DVector<Float> {
            let g = func
                .gradient(status.current.x.as_slice(), user_data)
                .expect("This should never fail");
            let x = &status.current.x - &(status.temperature * 1e0 * g);
            let x = x.constrain_to(bounds);
            x
        }
    }

    #[test]
    fn test_simulated_annealing() {
        let mut solver = SimulatedAnnealing::new(SimulatedAnnealingConfig::new(
            1.0,
            0.999,
            1e-3,
            AnnealingGenerator,
        )); // TODO: We need to sort this out, the config should go later
        let mut problem = Rosenbrock { n: 2 };
        let result = solver
            .process(
                &mut problem,
                &mut (),
                SimulatedAnnealingConfig::new(1.0, 0.999, 1e-3, AnnealingGenerator)
                    .with_bounds([(-5.0, 5.0), (-5.0, 5.0)]),
                &[
                    SimulatedAnnealingTerminator.build(),
                    MaxSteps(5_000).build(),
                ],
            )
            .unwrap();
        println!("{}", result);
        assert_relative_eq!(result.fx, 0.0, epsilon = 0.5);
    }
}
