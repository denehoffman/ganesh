use crate::{
    core::{LinearAlgebra, RealScalar, Vector},
    traits::{CostFunction, LogDensity},
};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};

/// Coordinates in parameter space that have not been evaluated.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct Point<I> {
    /// The point's position.
    pub x: I,
}

impl<I> Point<I> {
    /// Create a new unevaluated point from coordinates.
    pub const fn new(x: I) -> Self {
        Self { x }
    }

    /// Return a shared reference to the point coordinates.
    pub const fn x(&self) -> &I {
        &self.x
    }

    /// Convert the point into its coordinates.
    pub fn into_inner(self) -> I {
        self.x
    }
}

impl<T, B> Point<Vector<T, B>>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Evaluate a scalar/linear-algebra-generic objective and retain the scalar value with the point.
    ///
    /// # Errors
    /// Returns an error when objective evaluation fails.
    pub fn evaluate<P, U, E>(self, func: &P, args: &U) -> Result<EvaluatedPoint<Vector<T, B>, T>, E>
    where
        P: CostFunction<T, B, U, E>,
    {
        let value = func.evaluate(&self.x, args)?;
        Ok(EvaluatedPoint::new(self.x, value))
    }

    /// Evaluate a scalar/linear-algebra-generic log density.
    ///
    /// # Errors
    /// Returns an error when density evaluation fails.
    pub fn log_density<P, U, E>(
        self,
        func: &P,
        args: &U,
    ) -> Result<EvaluatedPoint<Vector<T, B>, T>, E>
    where
        P: LogDensity<T, B, U, E>,
    {
        let value = func.log_density(&self.x, args)?;
        Ok(EvaluatedPoint::new(self.x, value))
    }
}

impl<I: Debug> Display for Point<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "x: {:?}", self.x)
    }
}

/// Coordinates in parameter space together with a computed scalar value.
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct EvaluatedPoint<I, T = f64> {
    /// The point's position.
    pub x: I,
    /// The point's computed value.
    pub fx: T,
}

impl<I, T> EvaluatedPoint<I, T> {
    /// Create a new evaluated point from coordinates and value.
    pub const fn new(x: I, fx: T) -> Self {
        Self { x, fx }
    }

    /// Return the computed value.
    pub const fn value(&self) -> &T {
        &self.fx
    }

    /// Convert the [`EvaluatedPoint`] into an `I`-`Float` tuple.
    pub fn into_parts(self) -> (I, T) {
        (self.x, self.fx)
    }
}

impl<I, T: RealScalar> EvaluatedPoint<I, T> {
    /// Compare two points by their `fx` value.
    pub fn total_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fx.total_cmp(&other.fx)
    }
}

impl<I: Debug, T: Debug> Display for EvaluatedPoint<I, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "x: {:?}, f(x): {:?}", self.x, self.fx)
    }
}

impl<I, T: PartialEq> PartialEq for EvaluatedPoint<I, T> {
    fn eq(&self, other: &Self) -> bool {
        self.fx == other.fx
    }
}

impl<I, T: PartialOrd> PartialOrd for EvaluatedPoint<I, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.fx.partial_cmp(&other.fx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::Vector, test_functions::Rosenbrock};
    use nalgebra::dvector;
    use std::cmp::Ordering;

    #[test]
    fn evaluated_point_splits_coordinates_and_value() {
        let p = EvaluatedPoint::new(dvector![1.0, 2.0], 5.0);
        let (x, fx) = p.into_parts();
        assert_eq!(x, dvector![1.0, 2.0]);
        assert_eq!(fx, 5.0);
    }

    #[test]
    fn point_evaluation_returns_evaluated_point() {
        let f = Rosenbrock { n: 2 };
        let p = Point::new(Vector::<f64>::from_vec(vec![1.0, 1.0]));
        let evaluated = p.evaluate(&f, &()).unwrap();
        assert_eq!(evaluated.fx, 0.0);
    }

    #[test]
    fn log_density_returns_evaluated_point() {
        let f = Rosenbrock { n: 2 };
        let p = Point::new(Vector::<f64>::from_vec(vec![0.0, 0.0]));
        let evaluated = p.log_density(&f, &()).unwrap();
        assert_eq!(evaluated.fx, 0.0);
    }

    #[test]
    fn evaluated_point_total_cmp_and_partial_cmp() {
        let p1 = EvaluatedPoint::new(dvector![1.0], 1.0);
        let p2 = EvaluatedPoint::new(dvector![2.0], 2.0);
        assert_eq!(p1.total_cmp(&p2), Ordering::Less);
        assert_eq!(p1.partial_cmp(&p2), Some(Ordering::Less));
    }

    #[test]
    fn from_and_display() {
        let p = Point::new(vec![1.0, 2.0]);
        let s = format!("{}", p);
        assert!(s.contains("x:"));
    }
}
