use crate::{core::Bounds, DVector, Float};
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
    fn constrain_to(&self, bounds: Option<&Bounds>) -> Self;
    /// Transform [`self`] from an instance constrained inside the given [`Bounds`] to an
    /// unconstrained instance.
    fn unconstrain_from(&self, bounds: Option<&Bounds>) -> Self;
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

    fn constrain_to(&self, bounds: Option<&Bounds>) -> Self {
        bounds.map_or_else(
            || self.clone(),
            |bounds| {
                self.iter()
                    .zip(bounds.iter())
                    .map(|(val, bound)| bound.to_bounded(*val))
                    .collect()
            },
        )
    }

    fn unconstrain_from(&self, bounds: Option<&Bounds>) -> Self {
        bounds.map_or_else(
            || self.clone(),
            |bounds| {
                self.iter()
                    .zip(bounds.iter())
                    .map(|(val, bound)| bound.to_unbounded(*val))
                    .collect()
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

    fn constrain_to(&self, bounds: Option<&Bounds>) -> Self {
        bounds.map_or_else(
            || self.clone(),
            |bounds| {
                self.iter()
                    .zip(bounds.iter())
                    .map(|(val, bound)| bound.to_bounded(*val))
                    .collect::<Vec<_>>()
                    .into()
            },
        )
    }

    fn unconstrain_from(&self, bounds: Option<&Bounds>) -> Self {
        bounds.map_or_else(
            || self.clone(),
            |bounds| {
                self.iter()
                    .zip(bounds.iter())
                    .map(|(val, bound)| bound.to_unbounded(*val))
                    .collect::<Vec<_>>()
                    .into()
            },
        )
    }
}
