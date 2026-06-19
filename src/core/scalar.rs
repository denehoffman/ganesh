//! Scalar support shared by current and future generic optimizer APIs.

use fastrand::Rng;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;

/// Floating-point scalar contract supported by `ganesh` algorithms.
///
/// The generic scalar migration is intentionally limited to Rust's native floating-point types.
/// The scalar contract is intentionally independent from any linear algebra backend. Backend
/// capabilities live in [`crate::core::linalg`], while this trait covers numeric operations and
/// fixed algorithm constants.
pub trait RealScalar: Float + FromPrimitive + ToPrimitive + Debug + Send + Sync + 'static {
    /// Convert an `f64` algorithm constant into this scalar type.
    fn literal(value: f64) -> Self;

    /// Sample a scalar value in `[0, 1)` from the given random number generator.
    fn random_unit(rng: &mut Rng) -> Self;
}

impl RealScalar for f64 {
    fn literal(value: f64) -> Self {
        value
    }

    fn random_unit(rng: &mut Rng) -> Self {
        rng.f64()
    }
}

impl RealScalar for f32 {
    fn literal(value: f64) -> Self {
        value as Self
    }

    fn random_unit(rng: &mut Rng) -> Self {
        rng.f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_and_random_samples_follow_scalar_type() {
        let value32 = f32::literal(1.25);
        let value64 = f64::literal(1.25);
        assert_eq!(value32, 1.25_f32);
        assert_eq!(value64, 1.25_f64);

        let mut rng = Rng::with_seed(7);
        let sample32 = f32::random_unit(&mut rng);
        let sample64 = f64::random_unit(&mut rng);
        assert!((0.0..1.0).contains(&sample32));
        assert!((0.0..1.0).contains(&sample64));
    }
}
