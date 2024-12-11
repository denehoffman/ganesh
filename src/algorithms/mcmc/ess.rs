// https://arxiv.org/abs/2002.06212
// Ensemble Slice Sampling (ESS)
//
// TuneLengthScale(t, μ(t), N₊(t), N₋(t), M[adapt]):
// if t <= M[adapt] then
//     μ(t+1) = 2μ(t)N₊(t)/(N₊(t) + N₋(t))
// else
//     μ(t+1) = μ(t)
// endif
//
// This tells us the length scale at step t given the number of expansions N₊(t) and contractions N₋(t) at step t
//
// Differential Move:
//
// Given a walker Xₖ and complementary set of walkers S, pick two walkers Xₗ and Xₘ from S (without
// replacement) and compute direction vector ηₖ = μ(Xₗ - Xₘ)
//
// Gaussian Move:
//
// Cₛ = 1/|S|   ⅀ (Xₗ - X̅ₛ)(Xₗ - X̅ₛ)†
//            Xₗ∈S
// sample ηₖ/(2μ) ∝ Norm(0, Cₛ)
//
// Global Move (skip for now)
//
// ESS algorithm:
//
// Given t, f, μ(t), S[0], S[1], and M[adapt]:
// N₊(t) <- 0
// N₋(t) <- 0
// for i = 0,1 do
//     for k = 1,...,N/2 do
//         k <- k + i N/2 (assuming walker indices are just split down the middle, this selects the right ensemble)
//         ηₖ <- Differential Move
//         Y ~ U(0, f(Xₖ(t)))
//         U ~ U(0, 1)
//         L <- -U
//         R <- L + 1
//         while Y < f(L) do
//             L <- L - 1
//             N₊(t) <- N₊(t) + 1
//         end while
//         while Y < f(R) do
//             R <- R + 1
//             N₊(t) <- N₊(t) + 1
//         end while
//         while True do
//             X' ~ U(L, R)
//             Y' <- f(X'ηₖ + Xₖ(t))
//             if Y < Y' then
//                 break
//             end if
//             if X' < 0 then
//                 L <- X'
//                 N₋(t) <- N₋(t) + 1
//             else
//                 R <- X'
//                 N₋(t) <- N₋(t) + 1
//             end if
//         end while
//         Xₖ(t+1) <- X'ηₖ + Xₖ(t)
//     end for
// end for
// μ(t+1) <- TuneLengthScale(t, μ(t), N₊(t), N₋(t), M[adapt])
//
// Integrated Autocorrelation Time (IAT)
//             ∞
// IAT = 1 + 2 ⅀ ρ(k)
//            k=1
//
// This needs to be approximated (see Sokal A (1997) Monte carlo methods in statistical mechanics: foundations and new algorithms. In: Functional integration, Springer, pp 131–192):
//
// ρ̂(k) = ĉ(k) / ĉ(0)
//
// where
//                   n-k
// ĉ(k) = 1 / (n - k) ⅀ [X(k+m) - X̅][X(m) - X̅]
//                   m=1
//
// where X̅ is the mean of the samples.
//
// We can then calculate the number of effectively independent samples as n/IAT.
use std::sync::Arc;

use fastrand::Rng;
use nalgebra::{DMatrix, DVector};
use parking_lot::RwLock;

use crate::{algorithms::Point, Bound, Float, Function, RandChoice, SampleFloat};

use super::{Ensemble, MCMCAlgorithm};

#[derive(Copy, Clone)]
pub enum ESStep {
    Differential,
    Gaussian,
}
impl ESStep {
    fn step<U, E>(
        &self,
        i: usize,
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
                    let x_s = ensemble.mean_compliment(i);
                    let n_s = ensemble.len();
                    let c_s = ensemble
                        .iter_compliment(i)
                        .map(|x_l| (&x_l.read().x - &x_s) * (&x_l.read().x - &x_s).transpose())
                        .sum::<DMatrix<Float>>()
                        .unscale(n_s as Float);
                    let l = c_s.cholesky().expect("Error in Cholesky Decomposition").l();
                    let u = DVector::from_fn(x_s.len(), |_, _| rng.normal(0.0, 1.0));
                    (l * u).scale(2.0 * *mu)
                }
            };
            // Y ~ U(0, f(Xₖ(t)))
            let y = x_k.read().fx + rng.float().ln();
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
            while y < p_l.fx && n_expand < max_steps {
                // L <- L - 1
                l -= 1.0;
                p_l.set_position(&x_k.read().x + eta.scale(l));
                p_l.evaluate(func, user_data)?;
                // N₊(t) <- N₊(t) + 1
                n_expand += 1;
            }
            // while Y < f(R) do
            while y < p_r.fx && n_expand < max_steps {
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
                if y < p_yprime.fx || n_contract >= max_steps {
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
        if i <= n_adaptive {
            *mu *= 2.0 * (n_expand as Float) / (n_expand + n_contract) as Float
        }
        ensemble.push(positions);
        Ok(())
    }
}

/// The Affine Invariant MCMC Ensemble Sampler
///
/// <http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml>
#[derive(Clone)]
pub struct ESS {
    rng: Rng,
    step_types: Vec<(ESStep, Float)>,
    n_adaptive: usize,
    max_steps: usize,
    mu: Float,
}

impl ESS {
    pub fn new(step_types: &[(ESStep, Float)], rng: Rng) -> Self {
        Self {
            rng,
            step_types: step_types.to_vec(),
            n_adaptive: 0,
            max_steps: 10000,
            mu: 1.0,
        }
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
            .choice_weighted(&self.step_types.iter().map(|s| s.1).collect::<Vec<Float>>())
            .unwrap_or(0);
        let step_type = self.step_types[step_type_index].0;
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
