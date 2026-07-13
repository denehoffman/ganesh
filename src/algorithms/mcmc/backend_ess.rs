//! Scalar- and backend-generic ensemble slice sampling.

use crate::algorithms::mcmc::BackendEnsembleStatus;
use crate::core::{
    BackendMCMCSummary, Callbacks, LinearAlgebra, MaxSteps, NalgebraBackend, RandomScalar, Vector,
};
use crate::traits::{Algorithm, BackendTransform, BackendTransformedProblem, LogDensity};
use fastrand::Rng;
use std::marker::PhantomData;

/// Configuration for backend-generic ensemble slice sampling.
pub struct BackendESSConfig<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Initial one-dimensional slice bracket width.
    pub bracket_width: T,
    /// Maximum bracket-shrink evaluations per walker and ensemble step.
    pub max_shrink_steps: usize,
    /// Optional coordinate transform.
    pub transform: Option<Box<dyn BackendTransform<T, B>>>,
}

impl<T, B> Default for BackendESSConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            bracket_width: T::one(),
            max_shrink_steps: 1_000,
            transform: None,
        }
    }
}

impl<T, B> BackendESSConfig<T, B>
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

/// Scalar- and backend-generic differential-direction ensemble slice sampler.
#[derive(Clone, Debug)]
pub struct BackendESS<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    rng: Rng,
    _backend: PhantomData<(T, B)>,
}

impl<T, B> BackendESS<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Construct with an optional deterministic seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::new, Rng::with_seed),
            _backend: PhantomData,
        }
    }

    fn positive_uniform(&mut self) -> T {
        let mut value = T::random_unit(&mut self.rng);
        while value <= T::zero() {
            value = T::random_unit(&mut self.rng);
        }
        value
    }
}

impl<T, B> Default for BackendESS<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::new(None)
    }
}

impl<T, B, P, U, E> Algorithm<P, BackendEnsembleStatus<T, B>, U, E> for BackendESS<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    P: LogDensity<T, B, U, E>,
{
    type Summary = BackendMCMCSummary<T, B>;
    type Config = BackendESSConfig<T, B>;
    type Init = Vec<Vector<T, B>>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut BackendEnsembleStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        status.walkers = init
            .iter()
            .map(|walker| transformed.to_internal(walker))
            .collect();
        status.log_density.clear();
        status.chain = vec![Vec::new(); init.len()];
        for (index, walker) in status.walkers.iter().enumerate() {
            status
                .log_density
                .push(transformed.log_density(walker, args)?);
            status.evals.record_f();
            status.chain[index].push(transformed.to_external(walker));
        }
        status.message.initialize();
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut BackendEnsembleStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        let snapshot = status.walkers.clone();
        for walker_index in 0..snapshot.len() {
            let mut partner = self.rng.usize(0..snapshot.len());
            while partner == walker_index {
                partner = self.rng.usize(0..snapshot.len());
            }
            let direction = snapshot[walker_index].sub(&snapshot[partner]);
            let slice_level = status.log_density[walker_index] + self.positive_uniform().ln();
            let offset = T::random_unit(&mut self.rng);
            let mut left = -offset * config.bracket_width;
            let mut right = left + config.bracket_width;
            for _ in 0..config.max_shrink_steps {
                let coordinate = left + (right - left) * T::random_unit(&mut self.rng);
                let proposal = snapshot[walker_index].add_scaled(&direction, coordinate);
                let proposal_log_density = transformed.log_density(&proposal, args)?;
                status.evals.record_f();
                if proposal_log_density >= slice_level {
                    status.walkers[walker_index] = proposal;
                    status.log_density[walker_index] = proposal_log_density;
                    break;
                }
                if coordinate < T::zero() {
                    left = coordinate;
                } else {
                    right = coordinate;
                }
            }
        }
        for (index, walker) in status.walkers.iter().enumerate() {
            status.chain[index].push(transformed.to_external(walker));
        }
        status.message.step();
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &BackendEnsembleStatus<T, B>,
        _args: &U,
        _init: &Self::Init,
        _config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        let walkers = status.chain.len();
        let steps = status.chain.first().map_or(0, Vec::len);
        let variables = status
            .chain
            .first()
            .and_then(|walker| walker.first())
            .map_or(0, Vector::len);
        Ok(BackendMCMCSummary {
            parameter_names: None,
            message: status.message.clone(),
            chain: status.chain.clone(),
            evals: status.evals,
            dimension: (walkers, steps, variables),
        })
    }

    fn reset(&mut self) {}

    fn default_callbacks() -> Callbacks<Self, P, BackendEnsembleStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty().with_terminator(MaxSteps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    struct StandardNormal;

    impl<T, B> LogDensity<T, B> for StandardNormal
    where
        T: RandomScalar,
        B: LinearAlgebra<T>,
    {
        fn log_density(&self, x: &Vector<T, B>, _: &()) -> Result<T, Infallible> {
            Ok(-T::literal(0.5) * x.dot(x))
        }
    }

    #[test]
    fn ess_retains_backend_native_f32_chain() {
        let init = (0..8)
            .map(|index| {
                Vector::from_vec(vec![
                    0.25_f32.mul_add(index as f32, -1.0),
                    (-0.1_f32).mul_add(index as f32, 0.5),
                ])
            })
            .collect();
        let mut sampler = BackendESS::<f32>::new(Some(29));
        let result = sampler
            .process(
                &StandardNormal,
                &(),
                init,
                BackendESSConfig::default(),
                Callbacks::empty().with_terminator(MaxSteps(100)),
            )
            .unwrap();
        assert_eq!(result.dimension, (8, 101, 2));
        assert!(result.evals.f() >= 808);
    }
}
