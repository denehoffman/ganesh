/// Module containing the Nelder-Mead minimization algorithm.
pub mod nelder_mead;
pub use nelder_mead::NelderMead;

/// Module containing various line-search methods.
pub mod line_search;

/// Module containing the BFGS method.
pub mod bfgs;
pub use bfgs::BFGS;

/// Module containing the L-BFGS method.
pub mod lbfgs;
pub use lbfgs::LBFGS;

/// Module containing the L-BFGS-B method.
pub mod lbfgsb;
pub use lbfgsb::LBFGSB;

use nalgebra::DVector;
use std::{cmp::Ordering, fmt::Debug};

use crate::{Bound, Float, Function};

/// Describes a point in parameter space that can be used in [`Algorithm`](`crate::Algorithm`)s.
#[derive(PartialEq, Clone, Default, Debug)]
pub struct Point {
    x: DVector<Float>,
    fx: Float,
}
impl Point {
    /// Get the dimension of the underlying space.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.x.len()
    }
    /// Convert the [`Point`] into a [`Vec`]-`Float` tuple.
    pub fn into_vec_val(self) -> (Vec<Float>, Float) {
        (self.x.data.into(), self.fx)
    }
}
impl Point {
    /// Evaluate the given function at the point's coordinate and set the `fx` value to the result.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn evaluate<U, E>(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.fx = func.evaluate(self.x.as_slice(), user_data)?;
        Ok(())
    }
    /// Evaluate the given function at the point's coordinate and set the `fx` value to the result.
    /// This function assumes `x` is an internal, unbounded vector, but performs a coordinate transform
    /// to bound `x` when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn evaluate_bounded<U, E>(
        &mut self,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.fx = func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?;
        Ok(())
    }
    /// Compare two points by their `fx` value.
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        self.fx.total_cmp(&other.fx)
    }
}

impl From<DVector<Float>> for Point {
    fn from(value: DVector<Float>) -> Self {
        Self {
            x: value,
            fx: Float::NAN,
        }
    }
}
impl From<Vec<Float>> for Point {
    fn from(value: Vec<Float>) -> Self {
        Self {
            x: DVector::from_vec(value),
            fx: Float::NAN,
        }
    }
}
impl<'a> From<&'a Point> for &'a Vec<Float> {
    fn from(value: &'a Point) -> Self {
        value.x.data.as_vec()
    }
}
impl From<&[Float]> for Point {
    fn from(value: &[Float]) -> Self {
        Self {
            x: DVector::from_column_slice(value),
            fx: Float::NAN,
        }
    }
}
impl<'a> From<&'a Point> for &'a [Float] {
    fn from(value: &'a Point) -> Self {
        value.x.data.as_slice()
    }
}
impl PartialOrd for Point {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.fx.partial_cmp(&other.fx)
    }
}
