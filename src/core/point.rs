use std::fmt::{Debug, Display};

use serde::{Deserialize, Serialize};

use crate::{
    core::bound::Boundable,
    traits::{CostFunction, LogDensity},
    DVector, Float,
};

use super::Bounds;

/// Describes a point in parameter space that can be used in [`Algorithm`](`crate::traits::Algorithm`)s.
#[derive(PartialEq, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Point<I> {
    /// the point's position
    pub x: I,
    /// the point's evaluation
    pub fx: Float,
}
impl<I> Point<I> {
    /// Convert the [`Point`] into a [`Vec`]-`Float` tuple.
    pub fn destructure(self) -> (I, Float) {
        let fx = self.fx_checked();
        (self.x, fx)
    }
    /// Evaluate the given function at the point's coordinate and set the `fx` value to the result.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn evaluate<U, E>(
        &mut self,
        func: &dyn CostFunction<U, E, Input = I>,
        user_data: &mut U,
    ) -> Result<(), E> {
        if self.fx.is_nan() {
            self.fx = func.evaluate(&self.x, user_data)?;
        }
        Ok(())
    }
    /// Evaluate the given function at the point's coordinate and set the `fx` value to the result.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn log_density<U, E>(
        &mut self,
        func: &dyn LogDensity<U, E, Input = I>,
        user_data: &mut U,
    ) -> Result<(), E> {
        if self.fx.is_nan() {
            self.fx = func.log_density(&self.x, user_data)?;
        }
        Ok(())
    }
    /// Compare two points by their `fx` value.
    pub fn total_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fx.total_cmp(&other.fx)
    }
    /// Move the point to a new position, resetting the evaluation of the point
    pub fn set_position(&mut self, x: I) {
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
}

impl<I> Point<I>
where
    I: Boundable,
{
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
        func: &dyn CostFunction<U, E, Input = I>,
        bounds: Option<&Bounds>,
        user_data: &mut U,
    ) -> Result<(), E> {
        if self.fx.is_nan() {
            self.fx = func.evaluate(&self.x.constrain_to(bounds), user_data)?;
        }
        Ok(())
    }
    /// Converts the point's `x` from an unbounded space to a bounded one.
    pub fn constrain_to(&self, bounds: Option<&Bounds>) -> Self {
        Self {
            x: self.x.constrain_to(bounds),
            fx: self.fx,
        }
    }
}

impl<I: Debug> Display for Point<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "x: {:?}, f(x): {}", self.x, self.fx)
    }
}

impl<I> From<I> for Point<I> {
    fn from(value: I) -> Self {
        Self {
            x: value,
            fx: Float::NAN,
        }
    }
}
// impl<'a> From<&'a Point> for &'a Vec<Float> {
//     fn from(value: &'a Point) -> Self {
//         value.x.data.as_vec()
//     }
// }
impl From<&[Float]> for Point<DVector<Float>> {
    fn from(value: &[Float]) -> Self {
        Self {
            x: DVector::from_column_slice(value),
            fx: Float::NAN,
        }
    }
}
impl From<Vec<Float>> for Point<DVector<Float>> {
    fn from(value: Vec<Float>) -> Self {
        Self {
            x: DVector::from_vec(value),
            fx: Float::NAN,
        }
    }
}
// impl<'a> From<&'a Point> for &'a [Float] {
//     fn from(value: &'a Point) -> Self {
//         value.x.data.as_slice()
//     }
// }
impl<I> PartialOrd for Point<I>
where
    I: PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.fx.partial_cmp(&other.fx)
    }
}
