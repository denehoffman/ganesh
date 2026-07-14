//! Scalar support shared by generic optimizer APIs.

use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Real floating-point operations required by `ganesh` algorithms.
///
/// This crate-owned contract is intentionally independent of any linear-algebra library and is
/// open to downstream newtypes and alternative floating-point representations. Algorithms add
/// narrower capability bounds, such as [`RandomScalar`], only when needed.
pub trait RealScalar:
    Copy
    + Clone
    + Debug
    + Display
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Send
    + Sync
    + 'static
{
    /// Additive identity.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;
    /// Positive infinity.
    fn infinity() -> Self;
    /// Machine epsilon for the representation.
    fn epsilon() -> Self;
    /// Convert an algorithm constant into this scalar representation.
    fn literal(value: f64) -> Self;
    /// Convert to the stable f64 reporting/diagnostic boundary.
    fn to_f64(self) -> Option<f64>;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// Cube root.
    fn cbrt(self) -> Self;
    /// Integer power.
    fn powi(self, exponent: i32) -> Self;
    /// Exponential.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Cosine.
    fn cos(self) -> Self;
    /// Euclidean remainder, with a nonnegative result for a positive divisor.
    fn rem_euclid(self, rhs: Self) -> Self;
    /// Fused multiply-add when supported by the representation.
    fn mul_add(self, multiplier: Self, addend: Self) -> Self;
    /// Whether this value is finite.
    fn is_finite(self) -> bool;
    /// Whether this value is NaN.
    fn is_nan(self) -> bool;
    /// Compare values using a deterministic total order, including NaNs and signed zeroes.
    fn total_cmp(&self, other: &Self) -> std::cmp::Ordering;
}

/// Scalar capability required by algorithms that draw random floating-point values.
pub trait RandomScalar: RealScalar {
    /// Sample a scalar value in `[0, 1)` from the given random number generator.
    fn random_unit(rng: &mut fastrand::Rng) -> Self;
}

macro_rules! impl_native_scalar {
    ($type:ty, $sample:ident) => {
        impl RealScalar for $type {
            #[inline]
            fn zero() -> Self {
                0.0
            }
            #[inline]
            fn one() -> Self {
                1.0
            }
            #[inline]
            fn infinity() -> Self {
                Self::INFINITY
            }
            #[inline]
            fn epsilon() -> Self {
                Self::EPSILON
            }
            #[inline]
            fn literal(value: f64) -> Self {
                value as Self
            }
            #[inline]
            fn to_f64(self) -> Option<f64> {
                Some(self as f64)
            }
            #[inline]
            fn abs(self) -> Self {
                self.abs()
            }
            #[inline]
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            #[inline]
            fn cbrt(self) -> Self {
                self.cbrt()
            }
            #[inline]
            fn powi(self, exponent: i32) -> Self {
                self.powi(exponent)
            }
            #[inline]
            fn exp(self) -> Self {
                self.exp()
            }
            #[inline]
            fn ln(self) -> Self {
                self.ln()
            }
            #[inline]
            fn cos(self) -> Self {
                self.cos()
            }
            #[inline]
            fn rem_euclid(self, rhs: Self) -> Self {
                self.rem_euclid(rhs)
            }
            #[inline]
            fn mul_add(self, multiplier: Self, addend: Self) -> Self {
                self.mul_add(multiplier, addend)
            }
            #[inline]
            fn is_finite(self) -> bool {
                self.is_finite()
            }
            #[inline]
            fn is_nan(self) -> bool {
                self.is_nan()
            }
            #[inline]
            fn total_cmp(&self, other: &Self) -> std::cmp::Ordering {
                <$type>::total_cmp(self, other)
            }
        }

        impl RandomScalar for $type {
            #[inline]
            fn random_unit(rng: &mut fastrand::Rng) -> Self {
                rng.$sample()
            }
        }
    };
}

impl_native_scalar!(f64, f64);
impl_native_scalar!(f32, f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_and_random_samples_follow_scalar_type() {
        assert_eq!(f32::literal(1.25), 1.25_f32);
        assert_eq!(f64::literal(1.25), 1.25_f64);
        let mut rng = fastrand::Rng::with_seed(7);
        assert!((0.0..1.0).contains(&f32::random_unit(&mut rng)));
        assert!((0.0..1.0).contains(&f64::random_unit(&mut rng)));
    }
}
