use crate::{core::utils::SampleFloat, Float};
use fastrand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    ops::{Deref, DerefMut},
};

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
impl From<(Float, Float)> for Bound {
    fn from(value: (Float, Float)) -> Self {
        assert!(value.0 < value.1);
        match (value.0.is_finite(), value.1.is_finite()) {
            (true, true) => Self::LowerAndUpperBound(value.0, value.1),
            (true, false) => Self::LowerBound(value.0),
            (false, true) => Self::UpperBound(value.1),
            (false, false) => Self::NoBound,
        }
    }
}
impl From<(Option<Float>, Option<Float>)> for Bound {
    fn from(value: (Option<Float>, Option<Float>)) -> Self {
        assert!(value.0 < value.1);
        match (value.0, value.1) {
            (Some(lb), Some(ub)) => Self::LowerAndUpperBound(lb, ub),
            (Some(lb), None) => Self::LowerBound(lb),
            (None, Some(ub)) => Self::UpperBound(ub),
            (None, None) => Self::NoBound,
        }
    }
}
impl From<&Self> for Bound {
    fn from(value: &Self) -> Self {
        *value
    }
}

impl Bound {
    /// Get a value in the uniform distribution between `lower` and `upper`.
    #[cfg(not(feature = "f32"))]
    pub fn get_uniform(&self, rng: &mut Rng) -> Float {
        rng.range(self.lower(), self.upper()) as Float
    }
    /// Get a value in the uniform distribution between `lower` and `upper`.
    #[cfg(feature = "f32")]
    pub fn get_uniform(&self, rng: &mut Rng) -> Float {
        rng.f32_range(self.lower()..self.upper()) as Float
    }
    /// Checks whether the given `value` is compatible with the bounds.
    pub fn contains(&self, value: Float) -> bool {
        match self {
            Self::NoBound => true,
            Self::LowerBound(lb) => value >= *lb,
            Self::UpperBound(ub) => value <= *ub,
            Self::LowerAndUpperBound(lb, ub) => value >= *lb && value <= *ub,
        }
    }
    /// Checks whether the given `value` is compatible with the bound and returns `0.0` if it is,
    /// and the distance to the bound otherwise signed by whether the bound is a lower (`-`) or
    /// upper (`+`) bound.
    pub fn bound_excess(&self, value: Float) -> Float {
        match self {
            Self::NoBound => 0.0,
            Self::LowerBound(lb) => {
                if value >= *lb {
                    0.0
                } else {
                    value - lb
                }
            }
            Self::UpperBound(ub) => {
                if value <= *ub {
                    0.0
                } else {
                    value - ub
                }
            }
            Self::LowerAndUpperBound(lb, ub) => {
                if value < *lb {
                    value - lb
                } else if value > *ub {
                    value - ub
                } else {
                    0.0
                }
            }
        }
    }
    /// Returns the lower bound or `-inf` if there is none.
    pub const fn lower(&self) -> Float {
        match self {
            Self::NoBound => Float::NEG_INFINITY,
            Self::LowerBound(lb) => *lb,
            Self::UpperBound(_) => Float::NEG_INFINITY,
            Self::LowerAndUpperBound(lb, _) => *lb,
        }
    }
    /// Returns the upper bound or `+inf` if there is none.
    pub const fn upper(&self) -> Float {
        match self {
            Self::NoBound => Float::INFINITY,
            Self::LowerBound(_) => Float::INFINITY,
            Self::UpperBound(ub) => *ub,
            Self::LowerAndUpperBound(_, ub) => *ub,
        }
    }
    /// Checks if the given value is equal to one of the bounds.
    ///
    /// TODO: his just does equality comparison right now, which probably needs to be improved
    /// to something with an epsilon (significant but not critical to most fits right now).
    pub fn at_bound(&self, value: Float) -> bool {
        match self {
            Self::NoBound => false,
            Self::LowerBound(lb) => value == *lb,
            Self::UpperBound(ub) => value == *ub,
            Self::LowerAndUpperBound(lb, ub) => value == *lb || value == *ub,
        }
    }
    /// Converts an unbounded "external" parameter into a bounded "internal" one via the transform:
    ///
    /// Upper and lower bounds:
    /// ```math
    /// x_\text{int} = \arcsin\left(2\frac{x_\text{ext} - x_\text{min}}{x_\text{max} - x_\text{min}} - 1\right)
    /// ```
    /// Upper bound only:
    /// ```math
    /// x_\text{int} = \sqrt{(x_\text{max} - x_\text{ext} + 1)^2 - 1}
    /// ```
    /// Lower bound only:
    /// ```math
    /// x_\text{int} = \sqrt{(x_\text{ext} - x_\text{min} + 1)^2 - 1}
    /// ```
    pub fn to_bounded(&self, val: Float) -> Float {
        match *self {
            Self::LowerBound(lb) => lb - 1.0 + Float::sqrt(Float::powi(val, 2) + 1.0),
            Self::UpperBound(ub) => ub + 1.0 - Float::sqrt(Float::powi(val, 2) + 1.0),
            Self::LowerAndUpperBound(lb, ub) => lb + (Float::sin(val) + 1.0) * (ub - lb) / 2.0,
            Self::NoBound => val,
        }
    }
    /// Converts a bounded "internal" parameter into an unbounded "external" one via the transform:
    ///
    /// Upper and lower bounds:
    /// ```math
    /// x_\text{ext} = x_\text{min} + \left(\sin(x_\text{int}) + 1\right)\frac{x_\text{max} - x_\text{min}}{2}
    /// ```
    /// Upper bound only:
    /// ```math
    /// x_\text{ext} = x_\text{max} + 1 - \sqrt{x_\text{int}^2 + 1}
    /// ```
    /// Lower bound only:
    /// ```math
    /// x_\text{ext} = x_\text{min} - 1 + \sqrt{x_\text{int}^2 + 1}
    /// ```
    pub fn to_unbounded(&self, val: Float) -> Float {
        match *self {
            Self::LowerBound(lb) => Float::sqrt(Float::powi(val - lb + 1.0, 2) - 1.0),
            Self::UpperBound(ub) => Float::sqrt(Float::powi(ub - val + 1.0, 2) - 1.0),
            Self::LowerAndUpperBound(lb, ub) => Float::asin(2.0 * (val - lb) / (ub - lb) - 1.0),
            Self::NoBound => val,
        }
    }
}

/// A struct that contains a list of [`Bound`]s.
#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Bounds(Vec<Bound>);

impl Bounds {
    /// Returns the inner Vector of bounds.
    pub fn into_inner(self) -> Vec<Bound> {
        self.0
    }
}

impl From<Vec<Bound>> for Bounds {
    fn from(value: Vec<Bound>) -> Self {
        Self(value)
    }
}

impl Deref for Bounds {
    type Target = Vec<Bound>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Bounds {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{traits::Boundable, DVector};

    fn sample_bounds() -> Bounds {
        vec![
            Bound::LowerBound(0.0),
            Bound::UpperBound(10.0),
            Bound::LowerAndUpperBound(-1.0, 1.0),
            Bound::NoBound,
        ]
        .into()
    }

    #[test]
    fn test_bound_contains_and_excess() {
        let b1 = Bound::LowerBound(0.0);
        assert!(b1.contains(1.0));
        assert!(!b1.contains(-1.0));
        assert_eq!(b1.bound_excess(-1.0), -1.0);

        let b2 = Bound::UpperBound(5.0);
        assert!(b2.contains(4.0));
        assert!(!b2.contains(6.0));
        assert_eq!(b2.bound_excess(6.0), 1.0);

        let b3 = Bound::LowerAndUpperBound(-1.0, 1.0);
        assert!(b3.contains(0.0));
        assert!(!b3.contains(2.0));
    }

    #[test]
    fn test_bound_lower_upper_at_bound() {
        let b = Bound::LowerAndUpperBound(-2.0, 3.0);
        assert_eq!(b.lower(), -2.0);
        assert_eq!(b.upper(), 3.0);
        assert!(b.at_bound(-2.0));
        assert!(b.at_bound(3.0));
        assert!(!b.at_bound(0.0));
    }

    #[test]
    fn test_bound_transformations() {
        let b = Bound::LowerAndUpperBound(0.0, 2.0);
        let val = 1.0;
        let bounded = b.to_bounded(val);
        let unbounded = b.to_unbounded(bounded);
        assert!((val - unbounded).abs() < 1e-6);
    }

    #[test]
    fn test_boundable_random_and_is_in() {
        let mut rng = Rng::with_seed(0);
        let bounds: Bounds = vec![
            Bound::LowerAndUpperBound(-1.0, 1.0),
            Bound::LowerAndUpperBound(0.0, 5.0),
            Bound::LowerAndUpperBound(10.0, 20.0),
        ]
        .into();

        let v: Vec<Float> = Boundable::random_vector_in(&bounds, &mut rng);
        let d: DVector<Float> = Boundable::random_vector_in(&bounds, &mut rng);

        assert_eq!(v.len(), bounds.len());
        assert_eq!(d.len(), bounds.len());

        assert!(v.is_in(&bounds));
        assert!(d.is_in(&bounds));
    }

    #[test]
    fn test_boundable_excess_constrain_unconstrain() {
        let bounds = sample_bounds();
        let v: Vec<Float> = vec![-1.0, 11.0, 0.0, 5.0];
        let d: DVector<Float> = v.clone().into();

        let v_excess = v.excess_from(&bounds);
        let d_excess = d.excess_from(&bounds);
        assert!(v_excess.iter().any(|x| *x != 0.0));
        assert!(d_excess.iter().any(|x| *x != 0.0));

        let v_constrained = v.constrain_to(Some(&bounds));
        let d_constrained = d.constrain_to(Some(&bounds));
        assert!(v_constrained.is_in(&bounds));
        assert!(d_constrained.is_in(&bounds));

        let v_unconstrained = v_constrained.unconstrain_from(Some(&bounds));
        let d_unconstrained = d_constrained.unconstrain_from(Some(&bounds));
        assert_eq!(v_unconstrained.len(), v.len());
        assert_eq!(d_unconstrained.len(), d.len());
    }

    #[test]
    fn test_bounds_container() {
        let b = Bound::LowerBound(0.0);
        let bounds: Bounds = vec![b].into();
        assert_eq!(bounds.into_inner(), vec![b]);
    }
}
