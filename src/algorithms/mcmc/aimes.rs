use std::sync::Arc;

use fastrand::Rng;
use nalgebra::DVector;
use parking_lot::RwLock;

use crate::{algorithms::Point, Bound, Float, Function, RandChoice, SampleFloat};

use super::{Ensemble, MCMCAlgorithm};

#[derive(Copy, Clone)]
pub enum AIStep {
    Stretch { a: Float },
    Walk,
}
impl AIStep {
    fn step<U, E>(
        &self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        ensemble: &mut Ensemble,
        rng: &mut Rng,
    ) -> Result<(), E> {
        let mut positions = Vec::with_capacity(ensemble.len());
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
                    let r = z.ln() * ((n - 1) as Float) + proposal.fx - x_k.read().fx;
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
                    let r = proposal.fx - x_k.read().fx;
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

/// The Affine Invariant MCMC Ensemble Sampler
///
/// <http://msp.berkeley.edu/camcos/2010/5-1/p04.xhtml>
#[derive(Clone)]
pub struct AIES {
    rng: Rng,
    step_types: Vec<(AIStep, Float)>,
}

impl AIES {
    pub fn new(step_types: &[(AIStep, Float)], rng: Rng) -> Self {
        Self {
            rng,
            step_types: step_types.to_vec(),
        }
    }
}

impl<U, E> MCMCAlgorithm<U, E> for AIES {
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
        step_type.step(func, bounds, user_data, ensemble, &mut self.rng)?;
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
