use crate::{
    core::bound::Bounds,
    traits::{Boundable, CostFunction, LogDensity},
    DVector, Float,
};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

/// Describes a point in parameter space that can be used in [`Algorithm`](`crate::traits::Algorithm`)s.
#[derive(PartialEq, Clone, Default, Debug, Serialize, Deserialize)]
pub struct Point<I> {
    /// the point's position
    pub x: I,
    /// the point's evaluation
    pub fx: Float,
}
impl<I> Point<I> {
    /// Convert the [`Point`] into a `I`-`Float` tuple.
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
        args: &U,
    ) -> Result<(), E> {
        if self.fx.is_nan() {
            self.fx = func.evaluate(&self.x, args)?;
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
        args: &U,
    ) -> Result<(), E> {
        if self.fx.is_nan() {
            self.fx = func.log_density(&self.x, args)?;
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
        args: &U,
    ) -> Result<(), E> {
        if self.fx.is_nan() {
            self.fx = func.evaluate(&self.x.constrain_to(bounds), args)?;
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
impl<I> PartialOrd for Point<I>
where
    I: PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.fx.partial_cmp(&other.fx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::Bound, test_functions::Rosenbrock};
    use std::cmp::Ordering;

    #[test]
    fn test_destructure_and_fx_checked() {
        let p = Point {
            x: vec![1.0, 2.0],
            fx: 5.0,
        };
        let (x, fx) = p.clone().destructure();
        assert_eq!(x, vec![1.0, 2.0]);
        assert_eq!(fx, 5.0);
        assert_eq!(p.fx_checked(), 5.0);
    }

    #[test]
    #[should_panic(expected = "Point value requested before evaluation")]
    fn test_fx_checked_panics_if_nan() {
        let p: Point<Vec<Float>> = Point {
            x: vec![1.0],
            fx: Float::NAN,
        };
        let _ = p.fx_checked();
    }

    #[test]
    fn test_evaluate_sets_fx_once() {
        let f = Rosenbrock { n: 2 };
        let mut p: Point<DVector<Float>> = Point::from(vec![1.0, 1.0]);
        assert!(p.fx.is_nan());
        p.evaluate(&f, &()).unwrap();
        assert_eq!(p.fx, 0.0);
        p.evaluate(&f, &()).unwrap();
        assert_eq!(p.fx, 0.0);
    }

    #[test]
    fn test_log_density_sets_fx_once() {
        let f = Rosenbrock { n: 2 };
        let mut p: Point<DVector<Float>> = Point::from(vec![0.0, 0.0]);
        assert!(p.fx.is_nan());
        p.log_density(&f, &()).unwrap();
        assert_eq!(p.fx, 0.0);
        p.log_density(&f, &()).unwrap();
        assert_eq!(p.fx, 0.0);
    }

    #[test]
    fn test_total_cmp_and_partial_cmp() {
        let p1 = Point {
            x: vec![1.0],
            fx: 1.0,
        };
        let p2 = Point {
            x: vec![2.0],
            fx: 2.0,
        };
        assert_eq!(p1.total_cmp(&p2), Ordering::Less);
        assert_eq!(p1.partial_cmp(&p2), Some(Ordering::Less));
    }

    #[test]
    fn test_set_position_resets_fx() {
        let mut p = Point {
            x: vec![1.0],
            fx: 5.0,
        };
        p.set_position(vec![2.0]);
        assert_eq!(p.x, vec![2.0]);
        assert!(p.fx.is_nan());
    }

    #[test]
    fn test_evaluate_bounded_and_constrain_to() {
        let f = Rosenbrock { n: 2 };
        let bounds: Bounds = vec![
            Bound::LowerAndUpperBound(-2.0, 2.0),
            Bound::LowerAndUpperBound(-2.0, 2.0),
        ]
        .into();
        let mut p: Point<DVector<Float>> = Point::from(vec![0.0, 0.0]);
        p.evaluate_bounded(&f, Some(&bounds), &()).unwrap();
        assert_eq!(p.fx, 1.0);

        let constrained = p.constrain_to(Some(&bounds));
        assert_eq!(constrained.fx, p.fx);
        assert!(constrained.x.len() == p.x.len());
    }

    #[test]
    fn test_from_and_display() {
        let p: Point<DVector<Float>> = Point::from(vec![1.0, 2.0]);
        let s = format!("{}", p);
        assert!(s.contains("x:"));
        assert!(s.contains("f(x):"));
    }
}
