use fastrand::Rng;
use fastrand_contrib::RngExt;
use nalgebra::{Cholesky, DMatrix, DVector};

use crate::Float;

pub(crate) fn generate_random_vector(
    dimension: usize,
    lb: Float,
    ub: Float,
    rng: &mut Rng,
) -> DVector<Float> {
    DVector::from_vec((0..dimension).map(|_| rng.range(lb, ub)).collect())
}
pub(crate) fn generate_random_vector_in_limits(
    limits: &[(Float, Float)],
    rng: &mut Rng,
) -> DVector<Float> {
    DVector::from_vec(
        (0..limits.len())
            .map(|i| rng.range(limits[i].0, limits[i].1))
            .collect(),
    )
}

/// Computes the covariance matrix using a given hessian matrix.
pub fn hessian_to_covariance(hessian: &DMatrix<Float>) -> Option<DMatrix<Float>> {
    hessian.clone().try_inverse().or_else(|| {
        hessian
            .clone()
            .pseudo_inverse(Float::cbrt(Float::EPSILON))
            .ok()
    })
}

/// A helper trait to provide a weighted random choice method
pub trait RandChoice {
    /// Return an random index sampled with the given weights
    fn choice_weighted(&mut self, weights: &[Float]) -> Option<usize>;
}

impl RandChoice for Rng {
    fn choice_weighted(&mut self, weights: &[Float]) -> Option<usize> {
        let total_weight = weights.iter().sum();
        let u: Float = self.range(0.0, total_weight);
        let mut cumulative_weight = 0.0;
        for (index, &weight) in weights.iter().enumerate() {
            cumulative_weight += weight;
            if u <= cumulative_weight {
                return Some(index);
            }
        }
        None
    }
}

/// A helper trait to get feature-gated floating-point random values
pub trait SampleFloat {
    /// Get a random value in a range
    fn range(&mut self, lower: Float, upper: Float) -> Float;
    /// Get a random value in the range [0, 1]
    fn float(&mut self) -> Float;
    /// Get a random Normal value
    fn normal(&mut self, mu: Float, sigma: Float) -> Float;
    /// Get a random value from a multivariate Normal distribution
    #[allow(clippy::expect_used)]
    fn mv_normal(&mut self, mu: &DVector<Float>, cov: &DMatrix<Float>) -> DVector<Float> {
        let cholesky = Cholesky::new(cov.clone()).expect("Covariance matrix not positive definite");
        let a = cholesky.l();
        let z = DVector::from_iterator(mu.len(), (0..mu.len()).map(|_| self.normal(0.0, 1.0)));
        mu + a * z
    }
}
impl SampleFloat for Rng {
    #[cfg(not(feature = "f32"))]
    fn range(&mut self, lower: Float, upper: Float) -> Float {
        self.f64_range(lower..upper)
    }
    #[cfg(feature = "f32")]
    fn range(&mut self, lower: Float, upper: Float) -> Float {
        self.f32_range(lower..upper)
    }
    #[cfg(not(feature = "f32"))]
    fn float(&mut self) -> Float {
        self.f64()
    }
    #[cfg(feature = "f32")]
    fn float(&mut self) -> Float {
        self.f32()
    }
    #[cfg(not(feature = "f32"))]
    fn normal(&mut self, mu: Float, sigma: Float) -> Float {
        self.f64_normal(mu, sigma)
    }
    #[cfg(feature = "f32")]
    fn normal(&mut self, mu: Float, sigma: Float) -> Float {
        self.f32_normal(mu, sigma)
    }
}
