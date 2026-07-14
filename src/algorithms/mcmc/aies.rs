//! Scalar- and linear-algebra-generic affine-invariant ensemble sampling.

use crate::algorithms::mcmc::ChainStorageMode;
use crate::core::{
    utils::sample_standard_normal, Callbacks, EvalCounts, LinearAlgebra, MCMCSummary,
    NalgebraProvider, RandomScalar, Vector,
};
use crate::traits::{
    Algorithm, LogDensity, ProgressStatus, Status, StatusMessage, SupportsParameterNames,
    Transform, TransformedProblem,
};
use fastrand::Rng;
use std::marker::PhantomData;
use std::ops::ControlFlow;

/// Generic ensemble sampler status and retained chain.
#[derive(Clone, Debug)]
pub struct EnsembleStatus<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
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
    /// Chain retention policy for the current run.
    pub chain_storage: ChainStorageMode,
    /// Number of completed ensemble steps, excluding initialization.
    pub chain_steps: usize,
}

impl<T, B> Default for EnsembleStatus<T, B>
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
            chain_storage: ChainStorageMode::default(),
            chain_steps: 0,
        }
    }
}

impl<T, B> EnsembleStatus<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    pub(crate) fn retain_walkers(&mut self, external: Vec<Vector<T, B>>) {
        let retain = match self.chain_storage {
            ChainStorageMode::Full | ChainStorageMode::Rolling { .. } => true,
            ChainStorageMode::Sampled { keep_every, .. } => {
                keep_every != 0 && self.chain_steps % keep_every == 0
            }
        };
        if !retain {
            return;
        }
        for (chain, walker) in self.chain.iter_mut().zip(external) {
            chain.push(walker);
            if let Some(limit) = self.chain_storage.history_limit() {
                if chain.len() > limit {
                    let excess = chain.len() - limit;
                    chain.drain(..excess);
                }
            }
        }
    }
}

/// Proposal move used by linear-algebra-generic AIES.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AIESMove<T: RandomScalar = f64> {
    /// Goodman–Weare stretch move.
    Stretch {
        /// Stretch distribution scale, greater than one.
        scale: T,
    },
    /// Goodman–Weare ensemble walk move.
    Walk,
}

impl<T: RandomScalar> AIESMove<T> {
    /// A default stretch move paired with a selection weight.
    pub fn stretch(weight: T) -> (Self, T) {
        (
            Self::Stretch {
                scale: T::literal(2.0),
            },
            weight,
        )
    }

    /// A stretch move with a custom scale paired with a selection weight.
    ///
    /// # Errors
    /// Returns a configuration error when `scale` is not finite and greater than one.
    pub fn custom_stretch(scale: T, weight: T) -> crate::error::GaneshResult<(Self, T)> {
        if !scale.is_finite() || scale <= T::one() {
            return Err(crate::error::GaneshError::ConfigError(
                "AIES stretch scale must be finite and greater than one".to_string(),
            ));
        }
        Ok((Self::Stretch { scale }, weight))
    }

    /// A walk move paired with a selection weight.
    pub const fn walk(weight: T) -> (Self, T) {
        (Self::Walk, weight)
    }
}

impl<T, B> Status for EnsembleStatus<T, B>
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

impl<T, B> ProgressStatus for EnsembleStatus<T, B>
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

/// Configuration for linear-algebra-generic AIES.
pub struct AIESConfig<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Stretch-move scale, required to be greater than one.
    stretch_scale: T,
    /// Weighted proposal moves; an empty list uses `stretch_scale` for compatibility.
    moves: Vec<(AIESMove<T>, T)>,
    /// Chain retention policy.
    chain_storage: ChainStorageMode,
    /// Optional names for the sampled parameters.
    parameter_names: Option<Vec<String>>,
    /// Optional coordinate transform.
    transform: Option<Box<dyn Transform<T, B>>>,
}

impl<T: RandomScalar, B: LinearAlgebra<T>> SupportsParameterNames for AIESConfig<T, B> {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

impl<T, B> Default for AIESConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            stretch_scale: T::literal(2.0),
            moves: Vec::new(),
            chain_storage: ChainStorageMode::default(),
            parameter_names: None,
            transform: None,
        }
    }
}

impl<T, B> AIESConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Create a configuration with default move settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the weighted proposal-move mixture.
    ///
    /// # Errors
    /// Returns a configuration error when weights are invalid or all zero.
    pub fn with_moves<I>(mut self, moves: I) -> crate::error::GaneshResult<Self>
    where
        I: IntoIterator<Item = (AIESMove<T>, T)>,
    {
        let moves: Vec<_> = moves.into_iter().collect();
        if moves.is_empty()
            || moves
                .iter()
                .any(|(_, weight)| !weight.is_finite() || *weight < T::zero())
            || moves.iter().all(|(_, weight)| *weight == T::zero())
        {
            return Err(crate::error::GaneshError::ConfigError(
                "AIES move weights must be finite, non-negative, and include a positive entry"
                    .to_string(),
            ));
        }
        self.moves = moves;
        Ok(self)
    }

    /// Select how much chain history is retained.
    pub const fn with_chain_storage(mut self, chain_storage: ChainStorageMode) -> Self {
        self.chain_storage = chain_storage;
        self
    }

    /// Configure a coordinate transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: Transform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

/// Validated starting walkers for an [`AIES`] run.
#[derive(Clone, Debug)]
pub struct AIESInit<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    walkers: Vec<Vector<T, B>>,
}

impl<T: RandomScalar, B: LinearAlgebra<T>> AIESInit<T, B> {
    /// Validate and store starting walker positions.
    pub fn new(walkers: Vec<Vector<T, B>>) -> crate::error::GaneshResult<Self> {
        validate_walkers(&walkers, "AIES", 2)?;
        Ok(Self { walkers })
    }
}

pub(crate) fn validate_walkers<T: RandomScalar, B: LinearAlgebra<T>>(
    walkers: &[Vector<T, B>],
    family: &str,
    minimum: usize,
) -> crate::error::GaneshResult<()> {
    if walkers.len() < minimum {
        return Err(crate::error::GaneshError::ConfigError(format!(
            "{family} requires at least {minimum} walkers"
        )));
    }
    let dimension = walkers[0].len();
    if dimension == 0 {
        return Err(crate::error::GaneshError::ConfigError(format!(
            "{family} walker dimension must be at least 1"
        )));
    }
    if walkers.iter().any(|walker| walker.len() != dimension) {
        return Err(crate::error::GaneshError::ConfigError(format!(
            "{family} walkers must all have the same dimension"
        )));
    }
    Ok(())
}

/// Scalar- and linear-algebra-generic affine-invariant ensemble sampler.
#[derive(Clone, Debug)]
pub struct AIES<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    rng: Rng,
    _provider: PhantomData<(T, B)>,
}

impl<T, B> AIES<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Construct with an optional deterministic seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::new, Rng::with_seed),
            _provider: PhantomData,
        }
    }

    fn stretch(&mut self, scale: T) -> T {
        let root = T::one() + (scale - T::one()) * T::random_unit(&mut self.rng);
        root * root / scale
    }

    fn choose_move(&mut self, moves: &[(AIESMove<T>, T)]) -> AIESMove<T> {
        let total = moves
            .iter()
            .fold(T::zero(), |sum, (_, weight)| sum + *weight);
        let mut draw = T::random_unit(&mut self.rng) * total;
        for (proposal, weight) in moves {
            if draw < *weight {
                return *proposal;
            }
            draw = draw - *weight;
        }
        moves[moves.len() - 1].0
    }
}

impl<T, B> Default for AIES<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::new(Some(0))
    }
}

impl<T, B, P, U, E> Algorithm<P, EnsembleStatus<T, B>, U, E> for AIES<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    P: LogDensity<T, B, U, E>,
{
    type Summary = MCMCSummary<T, B>;
    type Config = AIESConfig<T, B>;
    type Init = AIESInit<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut EnsembleStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        status.walkers = init
            .walkers
            .iter()
            .map(|walker| transformed.to_internal(walker))
            .collect();
        status.log_density.clear();
        status.chain = vec![Vec::new(); init.walkers.len()];
        status.chain_storage = config.chain_storage;
        status.chain_steps = 0;
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
        status: &mut EnsembleStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        let snapshot = status.walkers.clone();
        let dimension = snapshot[0].len();
        let selected_move = if config.moves.is_empty() {
            AIESMove::Stretch {
                scale: config.stretch_scale,
            }
        } else {
            self.choose_move(&config.moves)
        };
        match selected_move {
            AIESMove::Stretch { .. } => status.message.step_with_message("Stretch Move"),
            AIESMove::Walk => status.message.step_with_message("Walk Move"),
        }
        for walker_index in 0..status.walkers.len() {
            let (proposal, jacobian) = match selected_move {
                AIESMove::Stretch { scale } => {
                    let mut partner = self.rng.usize(0..snapshot.len());
                    while partner == walker_index {
                        partner = self.rng.usize(0..snapshot.len());
                    }
                    let z = self.stretch(scale);
                    (
                        snapshot[partner]
                            .add_scaled(&snapshot[walker_index].sub(&snapshot[partner]), z),
                        T::literal(dimension.saturating_sub(1) as f64) * z.ln(),
                    )
                }
                AIESMove::Walk => {
                    let count = snapshot.len().saturating_sub(1);
                    let mean = snapshot
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| *index != walker_index)
                        .fold(Vector::zeros(dimension), |sum, (_, walker)| sum.add(walker))
                        .scale(T::one() / T::literal(count as f64));
                    let displacement = snapshot
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| *index != walker_index)
                        .fold(Vector::zeros(dimension), |sum, (_, walker)| {
                            sum.add_scaled(
                                &walker.sub(&mean),
                                sample_standard_normal(&mut self.rng),
                            )
                        });
                    (snapshot[walker_index].add(&displacement), T::zero())
                }
            };
            let proposal_log_density = transformed.log_density(&proposal, args)?;
            status.evals.record_f();
            let log_acceptance = jacobian + proposal_log_density - status.log_density[walker_index];
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
        status.chain_steps += 1;
        let external = status
            .walkers
            .iter()
            .map(|walker| transformed.to_external(walker))
            .collect();
        status.retain_walkers(external);
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &EnsembleStatus<T, B>,
        _args: &U,
        _init: &Self::Init,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        let walkers = status.chain.len();
        let steps = status.chain.first().map_or(0, Vec::len);
        let variables = status
            .chain
            .first()
            .and_then(|walker| walker.first())
            .map_or(0, Vector::len);
        let mut message = status.message.clone();
        if matches!(message.status_type, crate::traits::StatusType::Custom)
            && message
                .text()
                .is_some_and(|text| text.contains("Maximum number of steps reached"))
        {
            let text = message.text_or_empty().to_string();
            message.succeed_with_message(text);
        }
        Ok(MCMCSummary {
            parameter_names: config.parameter_names.clone(),
            message,
            chain: status.chain.clone(),
            evals: status.evals,
            dimension: (walkers, steps, variables),
        })
    }

    fn reset(&mut self) {}

    fn default_callbacks() -> Callbacks<Self, P, EnsembleStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MaxSteps;
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
    fn aies_retains_provider_native_f32_chain() {
        let init: Vec<Vector<f32>> = (0..8)
            .map(|index| {
                Vector::from_vec(vec![
                    0.25_f32.mul_add(index as f32, -1.0),
                    (-0.1_f32).mul_add(index as f32, 0.5),
                ])
            })
            .collect();
        let mut sampler = AIES::<f32>::new(Some(23));
        let result = sampler
            .process(
                &StandardNormal,
                &(),
                AIESInit::new(init).unwrap(),
                AIESConfig::<f32>::default(),
                Callbacks::empty().with_terminator(MaxSteps(100)),
            )
            .unwrap();
        assert_eq!(result.dimension, (8, 101, 2));
        assert_eq!(result.evals.f(), 808);
    }

    #[test]
    fn aies_supports_weighted_moves_metadata_and_rolling_storage() {
        let init: Vec<Vector> = (0..8)
            .map(|index| Vector::from_vec(vec![index as f64 * 0.1, index as f64 * -0.05]))
            .collect();
        let names = vec!["x".to_string(), "y".to_string()];
        let config = AIESConfig {
            parameter_names: Some(names.clone()),
            ..AIESConfig::<f64>::default()
        }
        .with_moves([AIESMove::stretch(0.25), AIESMove::walk(0.75)])
        .unwrap()
        .with_chain_storage(ChainStorageMode::Rolling { window: 3 });
        let result = AIES::<f64>::new(Some(9))
            .process(
                &StandardNormal,
                &(),
                AIESInit::new(init).unwrap(),
                config,
                Callbacks::empty().with_terminator(MaxSteps(20)),
            )
            .unwrap();
        assert_eq!(result.parameter_names, Some(names));
        assert!(result.chain.iter().all(|chain| chain.len() == 3));
    }

    #[test]
    fn aies_initialization_reports_invalid_ensembles() {
        assert!(AIESInit::<f64>::new(vec![[0.0].into()]).is_err());
        assert!(AIESInit::<f64>::new(vec![[0.0].into(), [0.0, 1.0].into()]).is_err());
    }
}
