use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::{
    core::{bound::Bounds, Point, Summary},
    traits::{Algorithm, CostFunction, Status},
    utils::SampleFloat,
    Float,
};

/// A trait for generating new points in the simulated annealing algorithm.
pub trait SimulatedAnnealingGenerator<U, E> {
    /// Generates a new point based on the current point, cost function and the status.
    fn generate(
        &mut self,
        _func: &dyn CostFunction<U, E>,
        _bounds: Option<&Bounds>,
        _status: &mut SimulatedAnnealingStatus,
        _user_data: &mut U,
    ) -> DVector<Float>;
}

/// A struct for the simulated annealing algorithm.
pub struct SimulatedAnnealing<G, U, E>
where
    G: SimulatedAnnealingGenerator<U, E>,
{
    /// The initial temperature for the simulated annealing algorithm.
    pub initial_temperature: Float,
    /// The cooling rate for the simulated annealing algorithm.
    pub cooling_rate: Float,
    /// The minimum temperature for the simulated annealing algorithm.
    pub min_temperature: Float,
    /// The generator for generating new points in the simulated annealing algorithm.
    pub generator: G,
    rng: fastrand::Rng,
    _user_data: std::marker::PhantomData<U>,
    _error: std::marker::PhantomData<E>,
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

impl<G, U, E> SimulatedAnnealing<G, U, E>
where
    G: SimulatedAnnealingGenerator<U, E>,
{
    /// Creates a new instance of the simulated annealing algorithm.
    pub fn new(
        initial_temperature: Float,
        cooling_rate: Float,
        min_temperature: Float,
        generator: G,
    ) -> Self {
        Self {
            rng: fastrand::Rng::new(),
            generator,
            initial_temperature,
            cooling_rate,
            min_temperature,
            _user_data: std::marker::PhantomData,
            _error: std::marker::PhantomData,
        }
    }
}

impl<G, U, E> Algorithm<SimulatedAnnealingStatus, U, E> for SimulatedAnnealing<G, U, E>
where
    G: SimulatedAnnealingGenerator<U, E>,
{
    fn initialize(
        &mut self,
        func: &dyn CostFunction<U, E>,
        bounds: Option<&Bounds>,
        status: &mut SimulatedAnnealingStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        status.current.x = DVector::zeros(bounds.unwrap().len());
        let x0 = self.generator.generate(func, bounds, status, user_data);
        let fx0 = func.evaluate(x0.as_slice(), user_data)?;
        status.temperature = self.initial_temperature;
        status.current = Point { x: x0, fx: fx0 };
        status.best = status.current.clone();
        status.iteration = 0;
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        _func: &dyn CostFunction<U, E>,
        _bounds: Option<&Bounds>,
        status: &mut SimulatedAnnealingStatus,
        _user_data: &mut U,
    ) -> Result<bool, E> {
        if status.temperature < self.min_temperature {
            return Ok(true);
        }
        Ok(false)
    }

    fn step(
        &mut self,
        _i_step: usize,
        func: &dyn CostFunction<U, E>,
        bounds: Option<&Bounds>,
        status: &mut SimulatedAnnealingStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        let x = self.generator.generate(func, bounds, status, user_data);
        let fx = func.evaluate(x.as_slice(), user_data)?;
        status.cost_evals += 1;

        status.current = Point { x, fx };
        status.temperature *= self.cooling_rate;
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
        _func: &dyn CostFunction<U, E>,
        _bounds: Option<&Bounds>,
        _status: &mut SimulatedAnnealingStatus,
        _user_data: &mut U,
    ) -> Result<(), E> {
        Ok(())
    }

    fn summarize(
        &self,
        _func: &dyn CostFunction<U, E>,
        bounds: Option<&Bounds>,
        parameter_names: Option<&Vec<String>>,
        status: &SimulatedAnnealingStatus,
        _user_data: &U,
    ) -> Result<Summary, E> {
        let result = Summary {
            x0: vec![Float::NAN; status.best.x.nrows()],
            x: status.best.x.iter().cloned().collect(),
            fx: status.best.fx,
            bounds: bounds.cloned(),
            converged: status.converged,
            cost_evals: status.cost_evals,
            gradient_evals: 0,
            message: status.message.clone(),
            parameter_names: parameter_names.as_ref().map(|names| names.to_vec()),
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
        core::{Bound, Bounds, CtrlCAbortSignal, Engine},
        solvers::gradient_free::SimulatedAnnealing,
        test_functions::Rosenbrock,
        traits::{CostFunction, Gradient},
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
            let x = Bound::to_bounded(x.as_slice(), bounds);
            x
        }
    }

    #[test]
    fn test_simulated_annealing() {
        let solver = SimulatedAnnealing::new(1.0, 0.999, 1e-3, AnnealingGenerator);
        let mut m = Engine::new(solver).setup(|m| {
            m.with_abort_signal(CtrlCAbortSignal::new())
                .with_bounds([(-5.0, 5.0), (-5.0, 5.0)])
                .with_max_steps(5_000)
        });
        let problem = Rosenbrock { n: 2 };
        m.minimize(&problem).unwrap();
        println!("{}", m.result.unwrap());
        assert_relative_eq!(m.status.best.fx, 0.0, epsilon = 0.5);
    }
}
