// https://arxiv.org/abs/2002.06212
use std::sync::Arc;

use fastrand::Rng;
use nalgebra::DVector;
use parking_lot::RwLock;

use crate::{algorithms::Point, Bound, Float, Function, RandChoice, SampleFloat};

use super::{Ensemble, MCMCAlgorithm};

/// A move used by the [`ESS`] algorithm
///
/// See Karamanis & Beutler[^1] for step implementation algorithms
///
/// [^1]: Karamanis, M., & Beutler, F. (2020). Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions. arXiv Preprint arXiv: 2002.06212.
#[derive(Copy, Clone)]
pub enum ESSMove {
    /// The Differential move described in Algorithm 2 of Karamanis & Beutler
    Differential,
    /// The Gaussian move described in Algorithm 3 of Karamanis & Beutler
    Gaussian,
}
impl ESSMove {
    /// Create a new [`ESSMove::Differential`] with a usage weight
    pub const fn differential(weight: Float) -> WeightedESSMove {
        (Self::Differential, weight)
    }
    /// Create a new [`ESSMove::Gaussian`] with a usage weight
    pub const fn gaussian(weight: Float) -> WeightedESSMove {
        (Self::Gaussian, weight)
    }
    #[allow(clippy::too_many_arguments)]
    fn step<U, E>(
        &self,
        step: usize,
        n_adaptive: usize,
        max_steps: usize,
        mu: &mut Float,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        ensemble: &mut Ensemble,
        rng: &mut Rng,
    ) -> Result<(), E> {
        let mut positions = Vec::with_capacity(ensemble.len());
        let mut n_expand = 0;
        let mut n_contract = 0;
        let n = ensemble.walkers[0].get_latest().read().x.len();
        for (i, walker) in ensemble.iter().enumerate() {
            let x_k = walker.get_latest();
            let eta = match self {
                Self::Differential => {
                    // Given a walker Xₖ and complementary set of walkers S, pick two walkers Xₗ and Xₘ from S (without
                    // replacement) and compute direction vector ηₖ = μ(Xₗ - Xₘ)
                    let s = &ensemble.get_compliment_walkers(i, 2, rng);
                    let x_l = s[0].get_latest();
                    let x_m = s[1].get_latest();
                    let eta = (&x_l.read().x - &x_m.read().x).scale(*mu);
                    eta
                }
                Self::Gaussian => {
                    // Cₛ = 1/|S|   ⅀ (Xₗ - X̅ₛ)(Xₗ - X̅ₛ)†
                    //            Xₗ∈S
                    // sample ηₖ/(2μ) ∝ Norm(0, Cₛ)
                    //
                    // We can do this faster by selecting Zₗ ~ Norm(μ=0, σ=1) and
                    //
                    // W = ⅀ Zₗ(Xₗ - X̅ₛ)
                    //   Xₗ∈S
                    let x_s = ensemble.mean_compliment(i);
                    let n_s = ensemble.len();
                    ensemble
                        .iter_compliment(i)
                        .map(|x_l| (&x_l.read().x - &x_s).scale(rng.normal(0.0, 1.0)))
                        .sum::<DVector<Float>>()
                        .scale(2.0 * *mu)
                }
            };
            // Y ~ U(0, f(Xₖ(t)))
            let y = x_k.read().get_fx_checked() + rng.float().ln();
            // U ~ U(0, 1)
            // L <- -U
            let mut l = -rng.float();
            let mut p_l = Point::from(&x_k.read().x + eta.scale(l));
            p_l.evaluate(func, user_data)?;
            // R <- L + 1
            let mut r = l + 1.0;
            let mut p_r = Point::from(&x_k.read().x + eta.scale(r));
            p_r.evaluate(func, user_data)?;
            // while Y < f(L) do
            while y < p_l.get_fx_checked() && n_expand < max_steps {
                // L <- L - 1
                l -= 1.0;
                p_l.set_position(&x_k.read().x + eta.scale(l));
                p_l.evaluate(func, user_data)?;
                // N₊(t) <- N₊(t) + 1
                n_expand += 1;
            }
            // while Y < f(R) do
            while y < p_r.get_fx_checked() && n_expand < max_steps {
                // R <- R + 1
                r += 1.0;
                p_r.set_position(&x_k.read().x + eta.scale(r));
                p_r.evaluate(func, user_data)?;
                // N₊(t) <- N₊(t) + 1
                n_expand += 1;
            }
            // while True do
            let xprime = loop {
                // X' ~ U(L, R)
                let xprime = rng.range(l, r);
                // Y' <- f(X'ηₖ + Xₖ(t))
                let mut p_yprime = Point::from(&x_k.read().x + eta.scale(xprime));
                p_yprime.evaluate(func, user_data)?;
                if y < p_yprime.get_fx_checked() || n_contract >= max_steps {
                    // if Y < Y' then break
                    break xprime;
                }
                if xprime < 0.0 {
                    // if X' < 0 then L <- X'
                    l = xprime;
                } else {
                    // else R <- X'
                    r = xprime;
                }
                // N₋(t) <- N₋(t) + 1
                n_contract += 1;
            };
            // Xₖ(t+1) <- X'ηₖ + Xₖ(t)
            let mut proposal = Point::from(&x_k.read().x + eta.scale(xprime));
            proposal.evaluate(func, user_data)?;
            positions.push(Arc::new(RwLock::new(proposal)))
        }
        // μ(t+1) <- TuneLengthScale(t, μ(t), N₊(t), N₋(t), M[adapt])
        if step <= n_adaptive {
            *mu *= 2.0 * (n_expand as Float) / (n_expand + n_contract) as Float
        }
        ensemble.push(positions);
        Ok(())
    }
}

/// The Ensemble Slice Sampler
///
/// This sampler follows Algorithm 5 in Karamanis & Beutler.[^1].
///
/// [^1]: Karamanis, M., & Beutler, F. (2020). Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions. arXiv Preprint arXiv: 2002.06212.
#[derive(Clone)]
pub struct ESS {
    rng: Rng,
    moves: Vec<WeightedESSMove>,
    n_adaptive: usize,
    max_steps: usize,
    mu: Float,
}

/// A [`ESSMove`] coupled with a weight
pub type WeightedESSMove = (ESSMove, Float);

impl ESS {
    /// Create a new Ensemble Slice Sampler from a list of weighted [`ESSMove`]s
    pub fn new<T: AsRef<[WeightedESSMove]>>(moves: T, rng: Rng) -> Self {
        Self {
            rng,
            moves: moves.as_ref().to_vec(),
            n_adaptive: 0,
            max_steps: 10000,
            mu: 1.0,
        }
    }
    /// Set the number of adaptive moves to perform at the start of sampling (default: `0`)
    pub const fn with_n_adaptive(mut self, n_adaptive: usize) -> Self {
        self.n_adaptive = n_adaptive;
        self
    }
    /// Set the maximum number of expansion/contractions to perform at each step (default: `10000`)
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
    /// Set the adaptive scaling parameter, $`\mu`$ (default: `1.0`)
    pub const fn with_mu(mut self, mu: Float) -> Self {
        self.mu = mu;
        self
    }
}

impl<U, E> MCMCAlgorithm<U, E> for ESS {
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        ensemble: &mut Ensemble,
    ) -> Result<(), E> {
        ensemble.evaluate_latest(func, user_data)?;
        Ok(())
    }
    fn step(
        &mut self,
        i_step: usize,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        ensemble: &mut Ensemble,
    ) -> Result<(), E> {
        let step_type_index = self
            .rng
            .choice_weighted(&self.moves.iter().map(|s| s.1).collect::<Vec<Float>>())
            .unwrap_or(0);
        let step_type = self.moves[step_type_index].0;
        step_type.step(
            i_step,
            self.n_adaptive,
            self.max_steps,
            &mut self.mu,
            func,
            bounds,
            user_data,
            ensemble,
            &mut self.rng,
        )?;
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        chains: &mut Ensemble,
    ) -> Result<bool, E> {
        Ok(false)
    }
}
