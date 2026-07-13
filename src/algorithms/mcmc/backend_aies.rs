//! Scalar- and backend-generic affine-invariant ensemble sampling.

use crate::core::{
    BackendMCMCSummary, Callbacks, EvalCounts, LinearAlgebra, MaxSteps, NalgebraBackend,
    RandomScalar, Vector,
};
use crate::traits::{
    Algorithm, BackendTransform, BackendTransformedProblem, LogDensity, ProgressStatus, Status,
    StatusMessage,
};
use fastrand::Rng;
use std::marker::PhantomData;
use std::ops::ControlFlow;

/// Generic ensemble sampler status and retained chain.
#[derive(Clone, Debug)]
pub struct BackendEnsembleStatus<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Current status message.
    pub message: StatusMessage,
    /// Current internal walker positions.
    pub walkers: Vec<Vector<T, B>>,
    /// Current walker log densities.
    pub log_density: Vec<T>,
    /// Retained external positions grouped by walker.
    pub chain: Vec<Vec<Vector<T, B>>>,
    /// Log-density evaluation counts.
    pub evals: EvalCounts,
}

impl<T, B> Default for BackendEnsembleStatus<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            message: StatusMessage::default(),
            walkers: Vec::new(),
            log_density: Vec::new(),
            chain: Vec::new(),
            evals: EvalCounts::default(),
        }
    }
}

impl<T, B> Status for BackendEnsembleStatus<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn message(&self) -> &StatusMessage {
        &self.message
    }

    fn set_message(&mut self) -> &mut StatusMessage {
        &mut self.message
    }

    fn check_invariants(&mut self) -> ControlFlow<()> {
        if self.log_density.iter().any(|value| value.is_nan()) {
            self.message.fail_with_message("walker log density is NaN");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

impl<T, B> ProgressStatus for BackendEnsembleStatus<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn write_progress(&self, out: &mut String) -> std::fmt::Result {
        use std::fmt::Write;
        write!(
            out,
            "status={} walkers={} density_evals={}",
            self.message,
            self.walkers.len(),
            self.evals.f()
        )
    }
}

/// Configuration for backend-generic AIES.
pub struct BackendAIESConfig<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Stretch-move scale, required to be greater than one.
    pub stretch_scale: T,
    /// Optional coordinate transform.
    pub transform: Option<Box<dyn BackendTransform<T, B>>>,
}

impl<T, B> Default for BackendAIESConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            stretch_scale: T::literal(2.0),
            transform: None,
        }
    }
}

impl<T, B> BackendAIESConfig<T, B>
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

/// Scalar- and backend-generic affine-invariant ensemble sampler.
#[derive(Clone, Debug)]
pub struct BackendAIES<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    rng: Rng,
    _backend: PhantomData<(T, B)>,
}

impl<T, B> BackendAIES<T, B>
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

    fn stretch(&mut self, scale: T) -> T {
        let root = T::one() + (scale - T::one()) * T::random_unit(&mut self.rng);
        root * root / scale
    }
}

impl<T, B> Default for BackendAIES<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::new(None)
    }
}

impl<T, B, P, U, E> Algorithm<P, BackendEnsembleStatus<T, B>, U, E> for BackendAIES<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    P: LogDensity<T, B, U, E>,
{
    type Summary = BackendMCMCSummary<T, B>;
    type Config = BackendAIESConfig<T, B>;
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
        let dimension = snapshot[0].len();
        for walker_index in 0..status.walkers.len() {
            let mut partner = self.rng.usize(0..snapshot.len());
            while partner == walker_index {
                partner = self.rng.usize(0..snapshot.len());
            }
            let z = self.stretch(config.stretch_scale);
            let proposal =
                snapshot[partner].add_scaled(&snapshot[walker_index].sub(&snapshot[partner]), z);
            let proposal_log_density = transformed.log_density(&proposal, args)?;
            status.evals.record_f();
            let log_acceptance = T::literal(dimension.saturating_sub(1) as f64) * z.ln()
                + proposal_log_density
                - status.log_density[walker_index];
            let log_uniform = {
                let mut uniform = T::random_unit(&mut self.rng);
                while uniform <= T::zero() {
                    uniform = T::random_unit(&mut self.rng);
                }
                uniform.ln()
            };
            if log_uniform < log_acceptance {
                status.walkers[walker_index] = proposal;
                status.log_density[walker_index] = proposal_log_density;
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
    fn aies_retains_backend_native_f32_chain() {
        let init = (0..8)
            .map(|index| {
                Vector::from_vec(vec![
                    0.25_f32.mul_add(index as f32, -1.0),
                    (-0.1_f32).mul_add(index as f32, 0.5),
                ])
            })
            .collect();
        let mut sampler = BackendAIES::<f32>::new(Some(23));
        let result = sampler
            .process(
                &StandardNormal,
                &(),
                init,
                BackendAIESConfig::default(),
                Callbacks::empty().with_terminator(MaxSteps(100)),
            )
            .unwrap();
        assert_eq!(result.dimension, (8, 101, 2));
        assert_eq!(result.evals.f(), 808);
    }
}
