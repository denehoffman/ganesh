use fastrand::Rng;
use fastrand_contrib::RngExt;
use nalgebra::DVector;
use std::{
    fmt::Display,
    ops::{Deref, DerefMut},
};

use serde::{Deserialize, Serialize};

use crate::{utils::SampleFloat, Float};

/// An enum that describes a bound/limit on a parameter in a minimization.
///
/// [`Bound`]s take a generic `T` which represents some scalar numeric value. They can be used by
/// bounded algorithms directly, or by some unbounded algorithms using parameter space
/// transformations.
#[derive(Default, Copy, Clone, Debug, Serialize, Deserialize)]
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
impl Bound {
    /// Get a value in the uniform distribution between `lower` and `upper`.
    #[cfg(not(feature = "f32"))]
    pub fn get_uniform(&self, rng: &mut Rng) -> Float {
        rng.f64_range(self.lower()..self.upper()) as Float
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
    /// Checks whether the given [`DVector`] is compatible with the list of bounds in each
    /// coordinate.
    pub fn contains_vec(bounds: &[Self], vec: &DVector<Float>) -> bool {
        for (bound, value) in bounds.iter().zip(vec) {
            if !bound.contains(*value) {
                return false;
            }
        }
        true
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
    /// Checks whether each of the given [`DVector`]'s coordinates are compatible with the bounds
    /// and returns a [`DVector`] containing the result of [`Bound::bound_excess`] at each
    /// coordinate.
    pub fn bounds_excess(bounds: &[Self], vec: &DVector<Float>) -> DVector<Float> {
        bounds
            .iter()
            .zip(vec)
            .map(|(b, v)| b.bound_excess(*v))
            .collect::<Vec<Float>>()
            .into()
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
    pub fn to_bounded(values: &[Float], bounds: Option<&Bounds>) -> DVector<Float> {
        bounds
            .map_or_else(
                || values.to_vec(),
                |bounds| {
                    values
                        .iter()
                        .zip(bounds.iter())
                        .map(|(val, bound)| bound._to_bounded(*val))
                        .collect()
                },
            )
            .into()
    }
    fn _to_bounded(&self, val: Float) -> Float {
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
    pub fn to_unbounded(values: &[Float], bounds: Option<&Bounds>) -> DVector<Float> {
        bounds
            .map_or_else(
                || values.to_vec(),
                |bounds| {
                    values
                        .iter()
                        .zip(bounds.iter())
                        .map(|(val, bound)| bound._to_unbounded(*val))
                        .collect()
                },
            )
            .into()
    }
    fn _to_unbounded(&self, val: Float) -> Float {
        match *self {
            Self::LowerBound(lb) => Float::sqrt(Float::powi(val - lb + 1.0, 2) - 1.0),
            Self::UpperBound(ub) => Float::sqrt(Float::powi(ub - val + 1.0, 2) - 1.0),
            Self::LowerAndUpperBound(lb, ub) => Float::asin(2.0 * (val - lb) / (ub - lb) - 1.0),
            Self::NoBound => val,
        }
    }
}

/// A struct that contains a list of [`Bound`]s.
#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Bounds(Vec<Bound>);

impl Bounds {
    /// Creates a random vector of values in the bounds.
    pub fn random_vector(&self, rng: &mut Rng) -> DVector<Float> {
        self.iter()
            .map(|b| rng.range(b.lower(), b.upper()))
            .collect::<Vec<Float>>()
            .into()
    }

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
