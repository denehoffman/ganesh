use std::borrow::Cow;

use crate::{
    core::{Bound, Bounds},
    DVector, Float,
};
use fastrand::Rng;

/// A trait for types which can be constrained to a set of [`Bounds`].
pub trait Boundable {
    /// Creates a random vector of values in the bounds.
    fn random_vector_in(bounds: &Bounds, rng: &mut Rng) -> Self;
    /// Checks whether [`self`] is contained in the given [`Bounds`] .
    fn is_in(&self, bounds: &Bounds) -> bool;
    /// Returns the signed amount that [`self`] exceeds the given [`Bounds`].
    fn excess_from(&self, bounds: &Bounds) -> Self;
    /// Transform [`self`] to an instance constrained inside the given [`Bounds`].
    fn constrain_to(&self, bounds: Option<&Bounds>) -> Cow<Self>
    where
        Self: Clone;
    /// Transform [`self`] from an instance constrained inside the given [`Bounds`] to an
    /// unconstrained instance.
    fn unconstrain_from(&self, bounds: Option<&Bounds>) -> Cow<Self>
    where
        Self: Clone;
    /// Transform a pair of [`Boundable`]s into a set of [`Bounds`].
    fn pack(lower: &Self, upper: &Self) -> Bounds;
    /// Unpack a set of [`Bounds`] into a pair of [`Boundable`]s (lower, upper).
    fn unpack(bounds: &Bounds) -> (Self, Self)
    where
        Self: Sized;
    /// Restrict a [`Boundable`] by clipping it within a set of [`Bounds`].
    fn clip_to(&self, bounds: Option<&Bounds>) -> Cow<Self>
    where
        Self: Clone;
}
impl Boundable for Vec<Float> {
    fn random_vector_in(bounds: &Bounds, rng: &mut Rng) -> Self {
        bounds.iter().map(|b| b.get_uniform(rng)).collect()
    }
    fn is_in(&self, bounds: &Bounds) -> bool {
        for (value, bound) in self.iter().zip(bounds.iter()) {
            if !bound.contains(*value) {
                return false;
            }
        }
        true
    }

    fn excess_from(&self, bounds: &Bounds) -> Self {
        self.iter()
            .zip(bounds.iter())
            .map(|(v, b)| b.bound_excess(*v))
            .collect()
    }

    fn constrain_to(&self, bounds: Option<&Bounds>) -> Cow<Self> {
        bounds.map_or_else(
            || Cow::Borrowed(self),
            |bounds| {
                Cow::Owned(
                    self.iter()
                        .zip(bounds.iter())
                        .map(|(val, bound)| bound.to_bounded(*val))
                        .collect(),
                )
            },
        )
    }

    fn unconstrain_from(&self, bounds: Option<&Bounds>) -> Cow<Self> {
        bounds.map_or_else(
            || Cow::Borrowed(self),
            |bounds| {
                Cow::Owned(
                    self.iter()
                        .zip(bounds.iter())
                        .map(|(val, bound)| bound.to_unbounded(*val))
                        .collect(),
                )
            },
        )
    }

    fn pack(lower: &Self, upper: &Self) -> Bounds {
        lower
            .into_iter()
            .zip(upper.into_iter())
            .map(|(l, u)| Bound::from((l, u)))
            .collect::<Vec<Bound>>()
            .into()
    }

    fn unpack(bounds: &Bounds) -> (Self, Self) {
        (
            bounds.iter().map(|b| b.lower()).collect(),
            bounds.iter().map(|b| b.upper()).collect(),
        )
    }

    fn clip_to(&self, bounds: Option<&Bounds>) -> Cow<Self> {
        bounds.map_or_else(
            || Cow::Borrowed(self),
            |bounds| {
                Cow::Owned(
                    self.iter()
                        .zip(bounds.iter())
                        .map(|(val, bound)| bound.clip_value(*val))
                        .collect(),
                )
            },
        )
    }
}

impl Boundable for DVector<Float> {
    fn random_vector_in(bounds: &Bounds, rng: &mut Rng) -> Self {
        bounds
            .iter()
            .map(|b| b.get_uniform(rng))
            .collect::<Vec<_>>()
            .into()
    }
    fn is_in(&self, bounds: &Bounds) -> bool {
        for (value, bound) in self.iter().zip(bounds.iter()) {
            if !bound.contains(*value) {
                return false;
            }
        }
        true
    }

    fn excess_from(&self, bounds: &Bounds) -> Self {
        self.iter()
            .zip(bounds.iter())
            .map(|(v, b)| b.bound_excess(*v))
            .collect::<Vec<_>>()
            .into()
    }

    fn constrain_to(&self, bounds: Option<&Bounds>) -> Cow<Self> {
        bounds.map_or_else(
            || Cow::Borrowed(self),
            |bounds| {
                Cow::Owned(
                    self.iter()
                        .zip(bounds.iter())
                        .map(|(val, bound)| bound.to_bounded(*val))
                        .collect::<Vec<_>>()
                        .into(),
                )
            },
        )
    }

    fn unconstrain_from(&self, bounds: Option<&Bounds>) -> Cow<Self> {
        bounds.map_or_else(
            || Cow::Borrowed(self),
            |bounds| {
                Cow::Owned(
                    self.iter()
                        .zip(bounds.iter())
                        .map(|(val, bound)| bound.to_unbounded(*val))
                        .collect::<Vec<_>>()
                        .into(),
                )
            },
        )
    }

    fn pack(lower: &Self, upper: &Self) -> Bounds {
        lower
            .into_iter()
            .zip(upper.into_iter())
            .map(|(l, u)| Bound::from((*l, *u)))
            .collect::<Vec<Bound>>()
            .into()
    }

    fn unpack(bounds: &Bounds) -> (Self, Self) {
        (
            DVector::from_vec(bounds.iter().map(|b| b.lower()).collect()),
            DVector::from_vec(bounds.iter().map(|b| b.upper()).collect()),
        )
    }

    fn clip_to(&self, bounds: Option<&Bounds>) -> Cow<Self> {
        bounds.map_or_else(
            || Cow::Borrowed(self),
            |bounds| {
                Cow::Owned(
                    self.iter()
                        .zip(bounds.iter())
                        .map(|(val, bound)| bound.clip_value(*val))
                        .collect::<Vec<_>>()
                        .into(),
                )
            },
        )
    }
}
