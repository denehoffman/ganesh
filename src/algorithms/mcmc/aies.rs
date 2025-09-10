use crate::{
    algorithms::mcmc::{EnsembleStatus, Walker},
    core::{
        utils::{RandChoice, SampleFloat},
        Bounds, MCMCSummary, Point,
    },
    traits::{Algorithm, Bounded, LogDensity, Status},
    DVector, Float,
};
use fastrand::Rng;
use parking_lot::RwLock;
use std::sync::Arc;

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
    /// Create a new [`AIESMove::Walk`] with a usage weight
    pub const fn walk(weight: Float) -> WeightedAIESMove {
        (Self::Walk, weight)
    }
    fn step<P, U, E>(
        &self,
        problem: &P,
        args: &U,
        ensemble: &mut EnsembleStatus,
        rng: &mut Rng,
    ) -> Result<(), E>
    where
        P: LogDensity<U, E, Input = DVector<Float>>,
    {
        let mut positions = Vec::with_capacity(ensemble.len());
        match self {
            Self::Stretch { a } => {
                ensemble.update_message(&format!("Stretch Move (a = {})", &a));
            }
            Self::Walk => {
                ensemble.update_message("Walk Move");
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
                    let x_l = &ensemble.get_compliment_walker(i, rng).get_latest();
                    // Xₖ -> Y = Xₗ + Z(Xₖ(t) - Xₗ)
                    let mut proposal =
                        Point::from(&x_l.read().x - (&x_k.read().x - &x_l.read().x).scale(z));
                    proposal.log_density(problem, args)?;
                    // The acceptance probability should then be (in an n-dimensional problem),
                    //
                    // Pr[stretch] = min { 1, Zⁿ⁻¹ π(Y) / π(Xₖ(t))}
                    //
                    // Then if Pr[stretch] > U[0,1], Xₖ(t+1) = Y else Xₖ(t+1) = Xₖ(t)
                    let n = x_l.read().x.len();
                    let r = z.ln().mul_add((n - 1) as Float, proposal.fx_checked())
                        - x_k.read().fx_checked();
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
                    let x_s = ensemble.mean_compliment(i);
                    let w = ensemble
                        .iter_compliment(i)
                        .map(|x_l| (&x_l.read().x - &x_s).scale(rng.normal(0.0, 1.0)))
                        .sum::<DVector<Float>>();
                    let mut proposal = Point::from(&x_k.read().x + w);
                    // Xₖ -> Y = Xₖ + W
                    // where W ~ Norm(μ=0, σ=Cₛ)
                    proposal.log_density(problem, args)?;
                    // Pr[walk] = min { 1, π(Y) / π(Xₖ(t))}
                    let r = proposal.fx_checked() - x_k.read().fx_checked();
                    (proposal, r)
                }
            };
            if r > rng.float().ln() {
                positions.push(Arc::new(RwLock::new(proposal)))
            } else {
                positions.push(x_k)
            }
        }
        ensemble.push(positions);
        Ok(())
    }
}

/// The internal configuration struct for the [`AIES`] algorithm.
#[derive(Clone)]
pub struct AIESConfig {
    bounds: Option<Bounds>,
    walkers: Vec<Walker>,
    moves: Vec<WeightedAIESMove>,
}
impl Bounded for AIESConfig {
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds> {
        &mut self.bounds
    }
}
impl AIESConfig {
    /// Create a new configuratione with the initial positions of the walkers.
    ///
    /// This sets the default move list to use a [`AIESMove::Stretch`] move 100% of the time.
    ///
    /// # See Also
    /// [`Walker::new`]
    pub fn new(x0: Vec<DVector<Float>>) -> Self {
        Self {
            bounds: Default::default(),
            walkers: x0.into_iter().map(Walker::new).collect(),
            moves: vec![AIESMove::stretch(1.0)],
        }
    }
    /// Set the moves for the [`AIES`] algorithm to use.
    pub fn with_moves<T: AsRef<[WeightedAIESMove]>>(mut self, moves: T) -> Self {
        self.moves = moves.as_ref().to_vec();
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
    P: LogDensity<U, E, Input = DVector<Float>>,
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
        status.log_density_latest(problem, args)
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
            .unwrap_or(0);
        let step_type = config.moves[step_type_index].0;
        step_type.step(problem, args, status, &mut self.rng)
    }

    fn summarize(
        &self,
        _current_step: usize,
        _func: &P,
        status: &EnsembleStatus,
        _args: &U,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        Ok(MCMCSummary {
            bounds: config.bounds.clone(),
            parameter_names: None,
            message: status.message().to_string(),
            chain: status.get_chain(None, None),
            cost_evals: status.n_f_evals,
            gradient_evals: status.n_g_evals,
            converged: status.converged(),
            dimension: status.dimension(),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::Rosenbrock;

    fn make_walkers(n_walkers: usize, dim: usize) -> Vec<DVector<Float>> {
        (0..n_walkers)
            .map(|i| DVector::from_element(dim, i as Float + 1.0))
            .collect()
    }

    #[test]
    fn test_aies_config_builders() {
        let walkers = make_walkers(3, 2);
        let moves = vec![AIESMove::stretch(0.5), AIESMove::walk(0.5)];

        let config = AIESConfig::new(walkers.clone()).with_moves(moves.clone());

        assert_eq!(config.walkers.len(), walkers.len());
        assert_eq!(config.moves.len(), moves.len());
    }

    #[test]
    fn test_aiesmove_updates_message() {
        let mut rng = Rng::with_seed(0);
        let problem = Rosenbrock { n: 2 };
        let mut status = EnsembleStatus::default();

        AIESMove::Stretch { a: 2.0 }
            .step(&problem, &(), &mut status, &mut rng)
            .unwrap();
        assert!(status.message().contains("Stretch Move"));

        AIESMove::Walk
            .step(&problem, &(), &mut status, &mut rng)
            .unwrap();
        assert!(status.message().contains("Walk Move"));
    }

    #[test]
    fn test_aies_initialize_and_summarize() {
        let mut aies = AIES::default();

        let walkers = make_walkers(3, 2);
        let config = AIESConfig::new(walkers.clone());
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
        let config = AIESConfig::new(walkers).with_moves(moves);

        let mut status = EnsembleStatus::default();
        aies.initialize(&problem, &mut status, &(), &config)
            .unwrap();

        assert!(aies.step(0, &problem, &mut status, &(), &config).is_ok());
    }
}
