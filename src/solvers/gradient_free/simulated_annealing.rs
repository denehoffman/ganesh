use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::{
    core::{bound::Bounds, Point, Summary},
    traits::{CostFunction, Solver, Status},
    utils::SampleFloat,
    Float,
};

pub trait SimulatedAnnealingGenerator<U, E> {
    fn generate(
        &mut self,
        _func: &dyn CostFunction<U, E>,
        _bounds: Option<&Bounds>,
        _status: &mut SimulatedAnnealingStatus,
        _user_data: &mut U,
    ) -> DVector<Float>;
}

pub struct SimulatedAnnealing<G, U, E>
where
    G: SimulatedAnnealingGenerator<U, E>,
{
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    pub min_temperature: f64,
    pub generator: G,
    pub rng: fastrand::Rng,
    _user_data: std::marker::PhantomData<U>,
    _error: std::marker::PhantomData<E>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulatedAnnealingStatus {
    pub temperature: f64,
    pub best: Point,
    pub current: Point,
    pub iteration: usize,
    pub converged: bool,
    pub message: String,
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
    pub fn new(
        initial_temperature: f64,
        cooling_rate: f64,
        min_temperature: f64,
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

impl<G, U, E> Solver<SimulatedAnnealingStatus, U, E> for SimulatedAnnealing<G, U, E>
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
        let acceptance_probability;

        let x = self.generator.generate(func, bounds, status, user_data);
        let fx = func.evaluate(x.as_slice(), user_data)?;
        status.cost_evals += 1;

        status.current = Point { x, fx };
        status.temperature *= self.cooling_rate;
        status.iteration += 1;

        if status.current.fx < status.best.fx {
            acceptance_probability = 1.0;
        } else {
            let d_fx = status.current.fx - status.best.fx;
            acceptance_probability = (-d_fx / status.temperature).exp();
        }

        if acceptance_probability > self.rng.f64() {
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
            x0: vec![],
            x: status.best.x.iter().cloned().collect(),
            fx: status.best.fx,
            bounds: bounds.cloned(),
            converged: status.converged,
            cost_evals: status.cost_evals,
            gradient_evals: 0,
            message: status.message.clone(),
            parameter_names: parameter_names
                .as_ref()
                .map(|names| names.iter().cloned().collect()),
            std: vec![0.0; status.best.x.len()],
        };

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::{fmt::Debug, sync::Arc};

    use approx::assert_relative_eq;
    use nalgebra::DVector;
    use parking_lot::RwLock;

    use crate::{
        core::{Bound, Bounds, CtrlCAbortSignal, Minimizer},
        solvers::gradient_free::SimulatedAnnealing,
        test_functions::Rosenbrock,
        traits::{CostFunction, Gradient, Observer},
        Float,
    };

    use super::{SimulatedAnnealingGenerator, SimulatedAnnealingStatus};

    pub struct AnnealingObserver;
    impl AnnealingObserver {
        /// Finalize the [`Observer`] by wrapping it in an [`Arc`] and [`RwLock`]
        pub fn build() -> Arc<RwLock<Self>> {
            Arc::new(RwLock::new(Self))
        }
    }
    impl<U> Observer<SimulatedAnnealingStatus, U> for AnnealingObserver {
        fn callback(
            &mut self,
            step: usize,
            _bounds: Option<&Bounds>,
            status: &mut SimulatedAnnealingStatus,
            _user_data: &mut U,
        ) -> bool {
            println!(
                "[{step} | {:.3}], {}, best: {}",
                status.temperature, status.current, status.best
            );
            false
        }
    }

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
            let x = &status.current.x + &(status.temperature * g);
            let x = Bound::to_bounded(x.as_slice(), bounds);
            x
        }
    }

    #[test]
    fn test_simulated_annealing() {
        let observer = AnnealingObserver::build();
        let solver = SimulatedAnnealing::new(1.0, 0.9999, 1e-3, AnnealingGenerator);
        let mut m = Minimizer::new(solver).setup(|m| {
            m.with_abort_signal(CtrlCAbortSignal::new())
                .with_bounds([(-5.0, 5.0), (-5.0, 5.0)])
                .add_observer(observer.clone())
                .with_max_steps(10_000)
        });
        let problem = Rosenbrock { n: 2 };
        m.minimize(&problem).unwrap();
        println!("{}", m.result.unwrap());
        assert_relative_eq!(m.status.best.fx, 0.0, epsilon = Float::EPSILON.sqrt());
    }
}
