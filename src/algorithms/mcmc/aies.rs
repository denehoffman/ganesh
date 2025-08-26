use std::sync::Arc;

use fastrand::Rng;
use nalgebra::DVector;
use parking_lot::RwLock;

use crate::{
    algorithms::mcmc::Walker,
    core::{Bounds, MCMCSummary, Point},
    traits::{Algorithm, Bounded, CostFunction, Status},
    utils::{RandChoice, SampleFloat},
    Float,
};

use super::ensemble_status::EnsembleStatus;

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
    fn step<U, E>(
        &self,
        func: &dyn CostFunction<U, E, Input = DVector<Float>>,
        user_data: &mut U,
        ensemble: &mut EnsembleStatus,
        rng: &mut Rng,
    ) -> Result<(), E> {
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
                    proposal.evaluate(func, user_data)?;
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
                    proposal.evaluate(func, user_data)?;
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
#[derive(Clone, Default)]
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
    /// Set the moves for the [`AIES`] algorithm to use.
    pub fn with_moves<T: AsRef<[WeightedAIESMove]>>(mut self, moves: T) -> Self {
        self.moves = moves.as_ref().to_vec();
        self
    }
    /// Set the initial positions of the walkers
    ///
    /// # See Also
    /// [`Walker::new`]
    pub fn with_walkers(mut self, x0: Vec<DVector<Float>>) -> Self {
        self.walkers = x0.into_iter().map(Walker::new).collect();
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
    config: AIESConfig,
    rng: Rng,
}

/// A [`AIESMove`] coupled with a weight
pub type WeightedAIESMove = (AIESMove, Float);

impl AIES {
    /// Create a new Affine Invariant Ensemble Sampler
    pub fn new(rng: Rng) -> Self {
        Self {
            config: AIESConfig::default(),
            rng,
        }
    }
}

impl<P, U, E> Algorithm<P, EnsembleStatus, U, E> for AIES
where
    P: CostFunction<U, E, Input = DVector<Float>>,
{
    type Summary = MCMCSummary;
    type Config = AIESConfig;
    fn initialize(
        &mut self,
        config: Self::Config,
        problem: &P,
        status: &mut EnsembleStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.config = config;
        status.walkers = self.config.walkers.clone();
        status.evaluate_latest(problem, user_data)
    }

    fn step(
        &mut self,
        current_step: usize,
        problem: &P,
        status: &mut EnsembleStatus,
        user_data: &mut U,
    ) -> Result<(), E> {
        let step_type_index = self
            .rng
            .choice_weighted(
                &self
                    .config
                    .moves
                    .iter()
                    .map(|s| s.1)
                    .collect::<Vec<Float>>(),
            )
            .unwrap_or(0);
        let step_type = self.config.moves[step_type_index].0;
        step_type.step(problem, user_data, status, &mut self.rng)
    }

    fn summarize(
        &self,
        current_step: usize,
        func: &P,
        status: &EnsembleStatus,
        user_data: &U,
    ) -> Result<Self::Summary, E> {
        Ok(MCMCSummary {
            bounds: self.config.bounds.clone(),
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
