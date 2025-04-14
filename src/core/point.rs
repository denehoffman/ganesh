use std::fmt::Display;

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::{traits::CostFunction, Float};

use super::Bound;

/// Describes a point in parameter space that can be used in [`Solver`](`crate::traits::Solver`)s.
#[derive(PartialEq, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Point {
    /// the point's position
    pub x: DVector<Float>,
    /// the point's evaluation
    pub fx: Float,
}
impl Point {
    /// Get the dimension of the underlying space.
    #[allow(clippy::len_without_is_empty)]
    pub fn dimension(&self) -> usize {
        self.x.len()
    }
    /// Convert the [`Point`] into a [`Vec`]-`Float` tuple.
    pub fn into_vec_val(self) -> (Vec<Float>, Float) {
        let fx = self.fx_checked();
        (self.x.data.into(), fx)
    }
    /// Evaluate the given function at the point's coordinate and set the `fx` value to the result.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn evaluate<U, E>(
        &mut self,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
    ) -> Result<(), E> {
        if self.fx.is_nan() {
            self.fx = func.evaluate(self.x.as_slice(), user_data)?;
        }
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
    pub fn evaluate_bounded<UD, E>(
        &mut self,
        func: &dyn CostFunction<UD, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut UD,
    ) -> Result<(), E> {
        if self.fx.is_nan() {
            self.fx = func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?;
        }
        Ok(())
    }
    /// Compare two points by their `fx` value.
    pub fn total_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fx.total_cmp(&other.fx)
    }
    /// Move the point to a new position, resetting the evaluation of the point
    pub fn set_position(&mut self, x: DVector<Float>) {
        self.x = x;
        self.fx = Float::NAN;
    }
    /// Get the current evaluation of the point, if it has been evaluated
    ///
    /// # Panics
    ///
    /// This method will panic if the point is unevaluated.
    pub fn fx_checked(&self) -> Float {
        assert!(!self.fx.is_nan(), "Point value requested before evaluation");
        self.fx
    }
    /// Converts the point's `x` from an unbounded space to a bounded one.
    pub fn to_bounded(&self, bounds: Option<&Vec<Bound>>) -> Self {
        Self {
            x: Bound::to_bounded(self.x.as_slice(), bounds),
            fx: self.fx,
        }
    }
}

impl Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "fx: {}", self.fx)?;
        writeln!(f, "{}", self.x)
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
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.fx.partial_cmp(&other.fx)
    }
}
