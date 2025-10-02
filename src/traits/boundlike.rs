use std::fmt::{Debug, Display};

use dyn_clone::DynClone;
use fastrand::Rng;
use serde::{Deserialize, Serialize};

use crate::{core::utils::SampleFloat, Float};

/// An enum that describes a bound/limit on a parameter in a minimization.
///
/// [`Bound`]s take a generic `T` which represents some scalar numeric value. They can be used by
/// bounded algorithms directly, or by some unbounded algorithms using parameter space
/// transformations.
#[derive(Default, Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Bound {
    #[default]
    /// `(-inf, +inf)`
    NoBound,
    /// `(min, +inf)`
    LowerBound(Float),
    /// `(-inf, max)`
    UpperBound(Float),
    /// `(min, max)`
    LowerAndUpperBound(Float, Float),
}
impl Display for Bound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.lower(), self.upper())
    }
}

impl Bound {
    /// Get the bounds as a tuple of options.
    pub const fn as_options(&self) -> (Option<Float>, Option<Float>) {
        match self {
            Self::NoBound => (None, None),
            Self::LowerBound(l) => (Some(*l), None),
            Self::UpperBound(u) => (None, Some(*u)),
            Self::LowerAndUpperBound(l, u) => (Some(*l), Some(*u)),
        }
    }
    /// Get the bounds as a tuple of floats (infinite if unbounded).
    pub const fn as_floats(&self) -> (Float, Float) {
        (self.lower(), self.upper())
    }
    /// Check whether the interval has a finite upper bound.
    pub const fn has_upper(&self) -> bool {
        match self {
            Self::NoBound | Self::LowerBound(_) => false,
            Self::UpperBound(_) | Self::LowerAndUpperBound(_, _) => true,
        }
    }
    /// Check whether the interval has a finite lower bound.
    pub const fn has_lower(&self) -> bool {
        match self {
            Self::NoBound | Self::UpperBound(_) => false,
            Self::LowerBound(_) | Self::LowerAndUpperBound(_, _) => true,
        }
    }
    /// Get the lower bound (negative infinity if unbounded).
    pub const fn lower(&self) -> Float {
        match self {
            Self::NoBound | Self::UpperBound(_) => Float::NEG_INFINITY,
            Self::LowerBound(l) => *l,
            Self::LowerAndUpperBound(l, _) => *l,
        }
    }
    /// Get the upper bound (positive infinity if unbounded).
    pub const fn upper(&self) -> Float {
        match self {
            Self::NoBound | Self::LowerBound(_) => Float::INFINITY,
            Self::UpperBound(u) => *u,
            Self::LowerAndUpperBound(_, u) => *u,
        }
    }
    /// Get a random value in the bound.
    ///
    /// This uses the maximum/minimum representable value as
    /// limits in cases where bounds are infinite.
    pub fn random(&self, rng: &mut Rng) -> Float {
        rng.range(self.lower(), self.upper()) as Float
    }
    /// Check to see if the given value is contained in the bound.
    pub fn contains(&self, value: Float) -> bool {
        !((self.has_upper() && value >= self.upper())
            || (self.has_lower() && value <= self.lower()))
    }
    /// Get the signed excess from the bound.
    ///
    /// This will yield a negative value if the value is below the lower bound and a positive
    /// value if the value is above the upper bound. It will return zero if the value is within the bounds.
    pub fn get_excess(&self, value: Float) -> Float {
        match self.as_options() {
            (None, None) => 0.0,
            (None, Some(u)) => Float::max(value - u, 0.0),
            (Some(l), None) => Float::min(value - l, 0.0),
            (Some(l), Some(u)) => Float::max(value - u, 0.0) + Float::min(value - l, 0.0),
        }
    }
    /// Check if the value is near the bounds with a given tolerance.
    pub fn at_bound(&self, value: Float, tol: Float) -> bool {
        (value - self.upper()).abs() < tol || (value - self.lower()).abs() < tol
    }
    /// Clip a value to be within the bounds.
    pub fn clip_value(&self, value: Float) -> Float {
        match self.as_options() {
            (None, None) => value,
            (None, Some(u)) => Float::min(value, u),
            (Some(l), None) => Float::max(value, l),
            (Some(l), Some(u)) => value.clamp(l, u),
        }
    }
}
impl From<(Float, Float)> for Bound {
    fn from(value: (Float, Float)) -> Self {
        let (a, b) = value;
        let (l, u) = if a < b { (a, b) } else { (b, a) };
        match (l.is_finite(), u.is_finite()) {
            (true, true) => Self::LowerAndUpperBound(l, u),
            (true, false) => Self::LowerBound(l),
            (false, true) => Self::UpperBound(u),
            (false, false) => Self::NoBound,
        }
    }
}
impl From<(Option<Float>, Option<Float>)> for Bound {
    fn from(value: (Option<Float>, Option<Float>)) -> Self {
        match value {
            (Some(a), Some(b)) => (a, b),
            (Some(l), None) => (l, Float::INFINITY),
            (None, Some(u)) => (Float::NEG_INFINITY, u),
            (None, None) => (Float::NEG_INFINITY, Float::INFINITY),
        }
        .into()
    }
}

/// A trait representing a transform specifically involving a parameter bound.
#[typetag::serde]
pub trait BoundLike: DynClone + Debug + Send + Sync {
    /// The mapping to internal (unbounded) coordinates.
    fn to_internal_impl(&self, bound: Bound, x: Float) -> Float;
    /// The first derivative of the mapping to internal (unbounded) coordinates.
    fn d_to_internal_impl(&self, bound: Bound, x: Float) -> Float;
    /// The second derivative of the mapping to internal (unbounded) coordinates.
    fn dd_to_internal_impl(&self, bound: Bound, x: Float) -> Float;
    /// The mapping to external (bounded) coordinates
    fn to_external_impl(&self, bound: Bound, z: Float) -> Float;
    /// The first derivative of the mapping to external (bounded) coordinates
    fn d_to_external_impl(&self, bound: Bound, z: Float) -> Float;
    /// The second derivative of the mapping to external (bounded) coordinates
    fn dd_to_external_impl(&self, bound: Bound, z: Float) -> Float;
}

dyn_clone::clone_trait_object!(BoundLike);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bounds_creation() {
        let bound = Bound::from((None, None));
        assert_eq!(bound.lower(), Float::NEG_INFINITY);
        assert_eq!(bound.upper(), Float::INFINITY);
        let bound = Bound::from((None, Some(1.2)));
        assert_eq!(bound.lower(), Float::NEG_INFINITY);
        assert_eq!(bound.upper(), 1.2);
        let bound = Bound::from((Some(-3.4), None));
        assert_eq!(bound.lower(), -3.4);
        assert_eq!(bound.upper(), Float::INFINITY);
        let bound = Bound::from((Some(-3.4), Some(1.2)));
        assert_eq!(bound.lower(), -3.4);
        assert_eq!(bound.upper(), 1.2);
        let bound = Bound::from((Some(1.2), Some(-3.4)));
        assert_eq!(bound.lower(), -3.4);
        assert_eq!(bound.upper(), 1.2);
        let bound = Bound::from((Float::NEG_INFINITY, Float::INFINITY));
        assert_eq!(bound.lower(), Float::NEG_INFINITY);
        assert_eq!(bound.upper(), Float::INFINITY);
        let bound = Bound::from((Float::INFINITY, Float::NEG_INFINITY));
        assert_eq!(bound.lower(), Float::NEG_INFINITY);
        assert_eq!(bound.upper(), Float::INFINITY);
        let bound = Bound::from((Float::NEG_INFINITY, 1.2));
        assert_eq!(bound.lower(), Float::NEG_INFINITY);
        assert_eq!(bound.upper(), 1.2);
        let bound = Bound::from((-3.4, Float::INFINITY));
        assert_eq!(bound.lower(), -3.4);
        assert_eq!(bound.upper(), Float::INFINITY);
        let bound = Bound::from((-3.4, 1.2));
        assert_eq!(bound.lower(), -3.4);
        assert_eq!(bound.upper(), 1.2);
        let bound = Bound::from((1.2, -3.4));
        assert_eq!(bound.lower(), -3.4);
        assert_eq!(bound.upper(), 1.2);
    }
}
