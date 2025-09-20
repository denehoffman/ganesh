use crate::{
    traits::{CostFunction, LogDensity, Transform},
    DVector, Float,
};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

/// Describes a point in parameter space that can be used in [`Algorithm`](`crate::traits::Algorithm`)s.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Point<I> {
    /// the point's position
    pub x: I,
    /// the point's evaluation (`None` if the point has not yet been evaluated)
    pub fx: Option<Float>,
}
impl<I> Point<I> {
    /// Convert the [`Point`] into a `I`-`Float` tuple.
    pub fn destructure(self) -> (I, Float) {
        let fx = self.fx_checked();
        (self.x, fx)
    }
    /// Compare two points by their `fx` value.
    pub fn total_cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (&self.fx, &other.fx) {
            (None, None) => std::cmp::Ordering::Equal,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (Some(_), None) => std::cmp::Ordering::Less,
            (Some(s), Some(o)) => s.total_cmp(o),
        }
    }
    /// Move the point to a new position, resetting the evaluation of the point
    pub fn set_position(&mut self, x: I) {
        self.x = x;
        self.fx = None;
    }
    /// Get the current evaluation of the point, if it has been evaluated
    ///
    /// # Panics
    ///
    /// This method will panic if the point is unevaluated.
    pub fn fx_checked(&self) -> Float {
        #[allow(clippy::expect_used)]
        self.fx.expect("Point value requested before evaluation")
    }
}
impl Point<DVector<Float>> {
    /// Evaluate the given function at the point's coordinate and set the `fx` value to the result.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn evaluate<U, E>(&mut self, func: &dyn CostFunction<U, E>, args: &U) -> Result<(), E> {
        if self.fx.is_none() {
            self.fx = Some(func.evaluate(&self.x, args)?);
        }
        Ok(())
    }
    /// Evaluate the given function at the point's coordinate and set the `fx` value to the result.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn log_density<U, E>(&mut self, func: &dyn LogDensity<U, E>, args: &U) -> Result<(), E> {
        if self.fx.is_none() {
            self.fx = Some(func.log_density(&self.x, args)?);
        }
        Ok(())
    }
    /// Evaluate the given function at the point's coordinate and set the `fx` value to the result.
    /// This function assumes `x` is an internal vector, but performs a coordinate transform
    /// when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn evaluate_transformed<T, U, E>(
        &mut self,
        func: &dyn CostFunction<U, E>,
        transform: &Option<T>,
        args: &U,
    ) -> Result<(), E>
    where
        T: Transform + Clone,
    {
        if self.fx.is_none() {
            self.fx = Some(func.evaluate(&transform.to_external(&self.x), args)?);
        }
        Ok(())
    }
    /// Evaluate the log density function at the point's coordinate and set the `fx` value to the result.
    /// This function assumes `x` is an internal vector, but performs a coordinate transform
    /// when evaluating the function.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn log_density_transformed<T, U, E>(
        &mut self,
        func: &dyn LogDensity<U, E>,
        transform: &Option<T>,
        args: &U,
    ) -> Result<(), E>
    where
        T: Transform + Clone,
    {
        if self.fx.is_none() {
            self.fx = Some(func.log_density(&transform.to_external(&self.x), args)?);
        }
        Ok(())
    }
    /// Converts the point's `x` from an internal space to a external one.
    pub fn to_external<T>(&self, transform: &Option<T>) -> Self
    where
        T: Transform + Clone,
    {
        Self {
            x: transform.to_external(&self.x).into_owned(),
            fx: self.fx,
        }
    }
    /// Converts the point's `x` from a external space to an internal one.
    pub fn to_internal<T>(&self, transform: &Option<T>) -> Self
    where
        T: Transform + Clone,
    {
        Self {
            x: transform.to_internal(&self.x).into_owned(),
            fx: self.fx,
        }
    }
}

impl<I: Debug> Display for Point<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "x: {:?}, f(x): {:?}", self.x, self.fx)
    }
}

// impl<I> From<I> for Point<I> {
//     fn from(value: I) -> Self {
//         Self { x: value, fx: None }
//     }
// }
impl From<&[Float]> for Point<DVector<Float>> {
    fn from(value: &[Float]) -> Self {
        Self {
            x: DVector::from_column_slice(value),
            fx: None,
        }
    }
}
impl From<Vec<Float>> for Point<DVector<Float>> {
    fn from(value: Vec<Float>) -> Self {
        Self {
            x: DVector::from_vec(value),
            fx: None,
        }
    }
}
impl From<DVector<Float>> for Point<DVector<Float>> {
    fn from(value: DVector<Float>) -> Self {
        Self { x: value, fx: None }
    }
}
impl<I> PartialEq for Point<I> {
    fn eq(&self, other: &Self) -> bool {
        self.fx == other.fx
    }
}
impl<I> PartialOrd for Point<I> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.fx.partial_cmp(&other.fx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::Bounds, test_functions::Rosenbrock, traits::Bound};
    use nalgebra::dvector;
    use std::cmp::Ordering;

    #[test]
    fn test_destructure_and_fx_checked() {
        let p = Point {
            x: dvector![1.0, 2.0],
            fx: Some(5.0),
        };
        let (x, fx) = p.clone().destructure();
        assert_eq!(x, dvector![1.0, 2.0]);
        assert_eq!(fx, 5.0);
        assert_eq!(p.fx_checked(), 5.0);
    }

    #[test]
    #[should_panic(expected = "Point value requested before evaluation")]
    fn test_fx_checked_panics_if_nan() {
        let p = Point {
            x: dvector![1.0],
            fx: None,
        };
        let _ = p.fx_checked();
    }

    #[test]
    fn test_evaluate_sets_fx_once() {
        let f = Rosenbrock { n: 2 };
        let mut p = Point::from(vec![1.0, 1.0]);
        assert!(p.fx.is_none());
        p.evaluate(&f, &()).unwrap();
        assert_eq!(p.fx, Some(0.0));
        p.evaluate(&f, &()).unwrap();
        assert_eq!(p.fx, Some(0.0));
    }

    #[test]
    fn test_log_density_sets_fx_once() {
        let f = Rosenbrock { n: 2 };
        let mut p = Point::from(vec![0.0, 0.0]);
        assert!(p.fx.is_none());
        p.log_density(&f, &()).unwrap();
        assert_eq!(p.fx, Some(0.0));
        p.log_density(&f, &()).unwrap();
        assert_eq!(p.fx, Some(0.0));
    }

    #[test]
    fn test_total_cmp_and_partial_cmp() {
        let p1 = Point {
            x: dvector![1.0],
            fx: Some(1.0),
        };
        let p2 = Point {
            x: dvector![2.0],
            fx: Some(2.0),
        };
        assert_eq!(p1.total_cmp(&p2), Ordering::Less);
        assert_eq!(p1.partial_cmp(&p2), Some(Ordering::Less));
    }

    #[test]
    fn test_set_position_resets_fx() {
        let mut p = Point {
            x: dvector![1.0],
            fx: Some(5.0),
        };
        p.set_position(dvector![2.0]);
        assert_eq!(p.x, dvector![2.0]);
        assert!(p.fx.is_none());
    }

    #[test]
    fn test_evaluate_bounded_and_constrain_to() {
        let f = Rosenbrock { n: 2 };
        let bounds: Bounds = vec![
            Bound::LowerAndUpperBound(-2.0, 2.0),
            Bound::LowerAndUpperBound(-2.0, 2.0),
        ]
        .into();
        let mut p = Point::from(vec![0.0, 0.0]);
        p.evaluate_transformed(&f, &Some(bounds.clone()), &())
            .unwrap();
        assert_eq!(p.fx, Some(1.0));

        let constrained = p.to_external(&Some(bounds));
        assert_eq!(constrained.fx, p.fx);
        assert!(constrained.x.len() == p.x.len());
    }

    #[test]
    fn test_from_and_display() {
        let p = Point::from(vec![1.0, 2.0]);
        let s = format!("{}", p);
        assert!(s.contains("x:"));
        assert!(s.contains("f(x):"));
    }
}
