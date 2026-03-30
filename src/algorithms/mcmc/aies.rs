use crate::{
    DVector, Float,
    algorithms::mcmc::{
        ChainStorageMode, EnsembleStatus, Walker, validate_walker_inputs,
        validate_weighted_moves,
    },
    core::{
        MCMCSummary, Point,
        utils::{RandChoice, SampleFloat},
    },
    error::{GaneshError, GaneshResult},
    traits::{
        Algorithm, LogDensity, Status, SupportsParameterNames, SupportsTransform, Transform,
        status::StatusType,
    },
};
use fastrand::Rng;

/// A move used by the the [`AIES`] algorithm
///
/// See Goodman & Weare[^1] for move implementation algorithms
///
/// [^1]: Goodman, J., & Weare, J. (2010). Ensemble samplers with affine invariance. In Communications in Applied Mathematics and Computational Science (Vol. 5, Issue 1, pp. 65–80). Mathematical Sciences Publishers. <https://doi.org/10.2140/camcos.2010.5.65>
#[derive(Copy, Clone)]
pub enum AIESMove {
    /// The stretch step described in Equation (7) of Goodman & Weare
    Stretch {
        /// The scaling parameter (higher values encourage exploration) (default: 2.0)
        a: Float,
    },
    /// The walk move described in Equation (11) of Goodman & Weare
    Walk,
}
impl AIESMove {
    /// Create a new [`AIESMove::Stretch`] with a usage weight and default scaling parameter
    pub const fn stretch(weight: Float) -> WeightedAIESMove {
        (Self::Stretch { a: 2.0 }, weight)
    }
    /// Create a new [`AIESMove::Stretch`] with a usage weight and custom scaling parameter
    pub fn custom_stretch(a: Float, weight: Float) -> GaneshResult<WeightedAIESMove> {
        if a <= 0.0 {
            return Err(GaneshError::ConfigError(
                "Scaling parameter must be greater than 0".to_string(),
            ));
        }
        Ok((Self::Stretch { a }, weight))
    }
    /// Create a new [`AIESMove::Walk`] with a usage weight
    pub const fn walk(weight: Float) -> WeightedAIESMove {
        (Self::Walk, weight)
    }
    fn step<P, U, E>(
        &self,
        problem: &P,
        transform: &Option<Box<dyn Transform>>,
        args: &U,
        ensemble: &mut EnsembleStatus,
        rng: &mut Rng,
    ) -> Result<(), E>
    where
        P: LogDensity<U, E>,
    {
        let mut positions = Vec::with_capacity(ensemble.len());
        match self {
            Self::Stretch { a } => {
                ensemble
                    .set_message()
                    .step_with_message(&format!("Stretch Move (a = {})", &a));
            }
            Self::Walk => {
                ensemble.set_message().step_with_message("Walk Move");
            }
        }
        for (i, walker) in ensemble.iter().enumerate() {
            let x_k = walker.get_latest();
            let (proposal, r) = match self {
                Self::Stretch { a } => {
                    // g(z) ∝ 1/√z if z ∈ [1/a, a] or 0 otherwise
                    // where a > 1 can be adjusted (higher values mean more exploration)
                    //
                    // Normalization on g:
                    //  a
                    //  ∫ 1/√z dz = 2(a - 1)/√a
                    // 1/a
                    //
                    // let f(z) = √a/(2(a - 1)√z) if z ∈ [1/a, a] or 0 otherwise
                    //
                    // The CDF of f is then
                    //        x           x
                    // F(x) = ∫ f(z) dz = ∫ f(z) dz = (√(ax) - 1) / (a - 1)
                    //       -∞          1/a
                    //
                    // The inverse of the CDF is then
                    //
                    // F⁻¹(u) = ((a-1) u + 1)² / a
                    let z = (a - 1.0).mul_add(rng.float(), 1.0).powi(2) / a;
                    let x_l = ensemble
                        .walkers[ensemble.get_compliment_walker_index(i, rng)]
                        .get_latest();
                    // Xₖ -> Y = Xₗ + Z(Xₖ(t) - Xₗ)
                    let mut proposal = Point::from(
                        transform.to_internal(&x_l.x).as_ref()
                            + (transform.to_internal(&x_k.x).as_ref()
                                - transform.to_internal(&x_l.x).as_ref())
                            .scale(z),
                    );
                    proposal.log_density_transformed(problem, transform, args)?;
                    // The acceptance probability should then be (in an n-dimensional problem),
                    //
                    // Pr[stretch] = min { 1, Zⁿ⁻¹ π(Y) / π(Xₖ(t))}
                    //
                    // Then if Pr[stretch] > U[0,1], Xₖ(t+1) = Y else Xₖ(t+1) = Xₖ(t)
                    let n = x_l.x.len();
                    let r = z.ln().mul_add((n - 1) as Float, proposal.fx_checked())
                        - x_k.fx_checked();
                    (proposal, r)
                }
                Self::Walk => {
                    // Cₛ is the the covariance of the positions of all the walkers in S:
                    //
                    // X̅ₛ = 1/|S|   ⅀ Xₗ
                    //            Xₗ∈S
                    //
                    // Cₛ = 1/|S|   ⅀ (Xₗ - X̅ₛ)(Xₗ - X̅ₛ)†
                    //            Xₗ∈S
                    //
                    // We can do this faster by selecting Zₗ ~ Norm(μ=0, σ=1) and
                    //
                    // W = ⅀ Zₗ(Xₗ - X̅ₛ)
                    //   Xₗ∈S
                    let x_s = ensemble.internal_mean_compliment(i, transform);
                    let w = ensemble
                        .iter_compliment(i)
                        .map(|x_l| {
                            (transform.to_internal(&x_l.x).as_ref() - &x_s)
                                .scale(rng.normal(0.0, 1.0))
                        })
                        .sum::<DVector<Float>>();
                    let mut proposal = Point::from(transform.to_internal(&x_k.x).as_ref() + w);
                    // Xₖ -> Y = Xₖ + W
                    // where W ~ Norm(μ=0, σ=Cₛ)
                    proposal.log_density_transformed(problem, transform, args)?;
                    // Pr[walk] = min { 1, π(Y) / π(Xₖ(t))}
                    let r = proposal.fx_checked() - x_k.fx_checked();
                    (proposal, r)
                }
            };
            if r > rng.float().ln() {
                positions.push(proposal.to_external(transform))
            } else {
                positions.push(x_k.clone())
            }
        }
        ensemble.n_f_evals += ensemble.walkers.len();
        ensemble.push(positions);
        Ok(())
    }
}

/// The internal configuration struct for the [`AIES`] algorithm.
#[derive(Clone)]
pub struct AIESConfig {
    parameter_names: Option<Vec<String>>,
    transform: Option<Box<dyn Transform>>,
    walkers: Vec<Walker>,
    moves: Vec<WeightedAIESMove>,
    chain_storage: ChainStorageMode,
}
impl SupportsTransform for AIESConfig {
    fn get_transform_mut(&mut self) -> &mut Option<Box<dyn Transform>> {
        &mut self.transform
    }
}
impl SupportsParameterNames for AIESConfig {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}
impl AIESConfig {
    /// Create a new configuratione with the initial positions of the walkers.
    ///
    /// This sets the default move list to use a [`AIESMove::Stretch`] move 100% of the time.
    ///
    /// # See Also
    /// [`Walker::new`]
    pub fn new(x0: Vec<DVector<Float>>) -> GaneshResult<Self> {
        validate_walker_inputs(&x0, "AIES", 2)?;
        Ok(Self {
            parameter_names: None,
            transform: None,
            walkers: x0.into_iter().map(Walker::new).collect(),
            moves: vec![AIESMove::stretch(1.0)],
            chain_storage: ChainStorageMode::default(),
        })
    }
    /// Set the moves for the [`AIES`] algorithm to use.
    pub fn with_moves<T: AsRef<[WeightedAIESMove]>>(mut self, moves: T) -> GaneshResult<Self> {
        validate_weighted_moves(
            &moves.as_ref().iter().map(|move_weight| move_weight.1).collect::<Vec<_>>(),
            "AIES",
        )?;
        self.moves = moves.as_ref().to_vec();
        Ok(self)
    }
    /// Set how much chain history to retain in memory during sampling.
    pub const fn with_chain_storage(mut self, chain_storage: ChainStorageMode) -> Self {
        self.chain_storage = chain_storage;
        self
    }
}

/// The Affine Invariant Ensemble Sampler
///
/// This sampler follows the AIES algorithm defined in Goodman & Weare[^1].
///
/// [^1]: Goodman, J., & Weare, J. (2010). Ensemble samplers with affine invariance. In Communications in Applied Mathematics and Computational Science (Vol. 5, Issue 1, pp. 65–80). Mathematical Sciences Publishers. <https://doi.org/10.2140/camcos.2010.5.65>
#[derive(Clone)]
pub struct AIES {
    rng: Rng,
}
impl Default for AIES {
    fn default() -> Self {
        Self::new(Some(0))
    }
}

/// A [`AIESMove`] coupled with a weight
pub type WeightedAIESMove = (AIESMove, Float);

impl AIES {
    /// Create a new Affine Invariant Ensemble Sampler with the given seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed),
        }
    }
}

impl<P, U, E> Algorithm<P, EnsembleStatus, U, E> for AIES
where
    P: LogDensity<U, E>,
{
    type Summary = MCMCSummary;
    type Config = AIESConfig;
    fn initialize(
        &mut self,
        problem: &P,
        status: &mut EnsembleStatus,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        status.walkers = config.walkers.clone();
        let history_limit = config.chain_storage.history_limit();
        for walker in status.walkers.iter_mut() {
            walker.set_history_limit(history_limit);
        }
        status.log_density_latest(problem, args)?;
        status.set_message().initialize();
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut EnsembleStatus,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let step_type_index = self
            .rng
            .choice_weighted(&config.moves.iter().map(|s| s.1).collect::<Vec<Float>>())
            .expect("AIES move weights should be validated by AIESConfig::with_moves");
        let step_type = config.moves[step_type_index].0;
        step_type.step(problem, &config.transform, args, status, &mut self.rng)
    }

    fn summarize(
        &self,
        _current_step: usize,
        _func: &P,
        status: &EnsembleStatus,
        _args: &U,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        let mut message = status.message().clone();
        if matches!(message.status_type, StatusType::Custom)
            && message.text.contains("Maximum number of steps reached")
        {
            message.succeed_with_message(&message.text.clone());
        }
        Ok(MCMCSummary {
            bounds: None,
            parameter_names: config.parameter_names.clone(),
            message,
            chain: status.get_chain(None, None),
            chain_storage: config.chain_storage,
            cost_evals: status.n_f_evals,
            gradient_evals: status.n_g_evals,
            dimension: status.dimension(),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{Callbacks, MaxSteps},
        test_functions::Rosenbrock,
        traits::Algorithm,
    };
    use approx::assert_relative_eq;
    use std::convert::Infallible;

    fn make_walkers(n_walkers: usize, dim: usize) -> Vec<DVector<Float>> {
        (0..n_walkers)
            .map(|i| DVector::from_element(dim, i as Float + 1.0))
            .collect()
    }

    struct CenteredLogDensity {
        target: Float,
    }
    impl crate::traits::LogDensity<(), Infallible> for CenteredLogDensity {
        fn log_density(&self, x: &DVector<Float>, _: &()) -> Result<Float, Infallible> {
            Ok(-Float::powi(x[0] - self.target, 2))
        }
    }

    #[test]
    fn test_aies_config_builders() {
        let walkers = make_walkers(3, 2);
        let moves = vec![AIESMove::stretch(0.5), AIESMove::walk(0.5)];

        let config = AIESConfig::new(walkers.clone())
            .unwrap()
            .with_moves(moves.clone())
            .unwrap();

        assert_eq!(config.walkers.len(), walkers.len());
        assert_eq!(config.moves.len(), moves.len());
    }

    #[test]
    fn test_aies_rejects_invalid_move_weights() {
        let walkers = make_walkers(3, 2);

        let err = match AIESConfig::new(walkers.clone())
            .unwrap()
            .with_moves([AIESMove::stretch(-1.0), AIESMove::walk(1.0)])
        {
            Err(err) => err,
            Ok(_) => panic!("negative AIES move weights should be rejected"),
        };
        assert!(err.to_string().contains("finite and non-negative"));

        let err = match AIESConfig::new(walkers).unwrap().with_moves([
            AIESMove::stretch(0.0),
            AIESMove::walk(0.0),
        ]) {
            Err(err) => err,
            Ok(_) => panic!("zero-sum AIES move weights should be rejected"),
        };
        assert!(err.to_string().contains("sum to a positive finite value"));
    }

    #[test]
    fn test_aies_rejects_invalid_walker_inputs() {
        let err = match AIESConfig::new(Vec::new()) {
            Err(err) => err,
            Ok(_) => panic!("empty AIES walker lists should be rejected"),
        };
        assert!(err.to_string().contains("at least 2 walkers"));

        let err = match AIESConfig::new(vec![DVector::from_row_slice(&[1.0])]) {
            Err(err) => err,
            Ok(_) => panic!("single-walker AIES inputs should be rejected"),
        };
        assert!(err.to_string().contains("at least 2 walkers"));

        let err = match AIESConfig::new(vec![
            DVector::from_row_slice(&[1.0, 2.0]),
            DVector::from_row_slice(&[3.0]),
        ]) {
            Err(err) => err,
            Ok(_) => panic!("mixed-dimension AIES walkers should be rejected"),
        };
        assert!(err.to_string().contains("same dimension"));
    }

    #[test]
    fn test_aiesmove_updates_message() {
        let mut rng = Rng::with_seed(0);
        let problem = Rosenbrock { n: 2 };
        let mut status = EnsembleStatus::default();

        AIESMove::Stretch { a: 2.0 }
            .step(&problem, &None, &(), &mut status, &mut rng)
            .unwrap();
        assert!(status.message().to_string().contains("Stretch Move"));

        AIESMove::Walk
            .step(&problem, &None, &(), &mut status, &mut rng)
            .unwrap();
        assert!(status.message().to_string().contains("Walk Move"));
    }

    #[test]
    fn test_aies_initialize_and_summarize() {
        let mut aies = AIES::default();

        let walkers = make_walkers(3, 2);
        let config = AIESConfig::new(walkers.clone()).unwrap();
        let problem = Rosenbrock { n: 2 };
        let mut status = EnsembleStatus::default();

        aies.initialize(&problem, &mut status, &(), &config)
            .unwrap();
        assert_eq!(status.walkers.len(), walkers.len());

        let summary = aies.summarize(0, &problem, &status, &(), &config).unwrap();
        assert_eq!(summary.dimension, status.dimension());
    }

    #[test]
    fn test_aies_step_runs() {
        let mut aies = AIES::default();
        let problem = Rosenbrock { n: 2 };

        let walkers = make_walkers(3, 2);
        let moves = vec![AIESMove::stretch(1.0), AIESMove::walk(1.0)];
        let config = AIESConfig::new(walkers).unwrap().with_moves(moves).unwrap();

        let mut status = EnsembleStatus::default();
        aies.initialize(&problem, &mut status, &(), &config)
            .unwrap();

        assert!(aies.step(0, &problem, &mut status, &(), &config).is_ok());
    }

    #[test]
    fn stretch_move_proposes_toward_current_from_compliment() {
        let mut rng = Rng::with_seed(0);
        let a: Float = 2.0;
        let z = (a - 1.0).mul_add(rng.float(), 1.0).powi(2) / a;
        let expected = 1.0 + z * (2.0 - 1.0);
        let problem = CenteredLogDensity { target: expected };
        let mut ensemble = EnsembleStatus {
            walkers: vec![
                Walker::new(DVector::from_row_slice(&[2.0])),
                Walker::new(DVector::from_row_slice(&[1.0])),
            ],
            ..Default::default()
        };
        ensemble.log_density_latest(&problem, &()).unwrap();

        AIESMove::Stretch { a }
            .step(&problem, &None, &(), &mut ensemble, &mut Rng::with_seed(0))
            .unwrap();

        let x0 = ensemble.walkers[0].get_latest();
        assert_relative_eq!(x0.x[0], expected);
    }

    #[test]
    fn summary_marks_max_steps_as_success_and_counts_initial_evals() {
        let mut aies = AIES::default();
        let walkers = make_walkers(4, 2);
        let config = AIESConfig::new(walkers).unwrap();

        let result = aies
            .process(
                &Rosenbrock { n: 2 },
                &(),
                config,
                Callbacks::empty().with_terminator(MaxSteps(2)),
            )
            .unwrap();

        assert!(result.cost_evals >= 4);
        assert_eq!(result.gradient_evals, 0);
        assert!(result.message.success());
        assert!(result.message.text.contains("Maximum number of steps reached"));
    }

    #[test]
    fn summary_uses_parameter_names_from_config() {
        let mut aies = AIES::default();
        let result = aies
            .process(
                &Rosenbrock { n: 2 },
                &(),
                AIESConfig::new(make_walkers(4, 2))
                    .unwrap()
                    .with_parameter_names(["alpha", "beta"]),
                Callbacks::empty().with_terminator(MaxSteps(2)),
            )
            .unwrap();

        assert_eq!(
            result.parameter_names,
            Some(vec!["alpha".to_string(), "beta".to_string()])
        );
    }

    #[test]
    fn rolling_chain_storage_limits_retained_history() {
        let mut aies = AIES::default();
        let config = AIESConfig::new(make_walkers(4, 2))
            .unwrap()
            .with_chain_storage(ChainStorageMode::Rolling { window: 2 });

        let result = aies
            .process(
                &Rosenbrock { n: 2 },
                &(),
                config,
                Callbacks::empty().with_terminator(MaxSteps(4)),
            )
            .unwrap();

        assert_eq!(result.chain_storage, ChainStorageMode::Rolling { window: 2 });
        assert!(result.chain.iter().all(|walker| walker.len() <= 2));
        assert_eq!(result.dimension.1, 2);
    }
}
