use crate::{
    core::{LinearAlgebra, RealScalar, Vector},
    traits::{CostFunction, LegacyCostFunction, LegacyLogDensity, LogDensity, Transform},
    DVector, Float,
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
    /// Evaluate a scalar/backend-generic objective and retain the scalar value with the point.
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

    /// Evaluate a scalar/backend-generic log density.
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

impl Point<DVector<Float>> {
    /// Evaluate the given function at the point's coordinate.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn evaluate<U, E>(
        self,
        func: &dyn LegacyCostFunction<U, E>,
        args: &U,
    ) -> Result<EvaluatedPoint<DVector<Float>>, E> {
        let fx = func.evaluate(&self.x, args)?;
        Ok(EvaluatedPoint::new(self.x, fx))
    }

    /// Evaluate the given log-density function at the point's coordinate.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn log_density<U, E>(
        self,
        func: &dyn LegacyLogDensity<U, E>,
        args: &U,
    ) -> Result<EvaluatedPoint<DVector<Float>>, E> {
        let fx = func.log_density(&self.x, args)?;
        Ok(EvaluatedPoint::new(self.x, fx))
    }

    /// Evaluate the given function after converting internal coordinates to external coordinates.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn evaluate_transformed<T, U, E>(
        self,
        func: &dyn LegacyCostFunction<U, E>,
        transform: &Option<T>,
        args: &U,
    ) -> Result<EvaluatedPoint<DVector<Float>>, E>
    where
        T: Transform + Clone,
    {
        let fx = func.evaluate(&transform.to_external(&self.x), args)?;
        Ok(EvaluatedPoint::new(self.x, fx))
    }

    /// Evaluate the log-density function after converting internal coordinates to external
    /// coordinates.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. Users should implement this trait to return a
    /// `std::convert::Infallible` if the function evaluation never fails.
    pub fn log_density_transformed<T, U, E>(
        self,
        func: &dyn LegacyLogDensity<U, E>,
        transform: &Option<T>,
        args: &U,
    ) -> Result<EvaluatedPoint<DVector<Float>>, E>
    where
        T: Transform + Clone,
    {
        let fx = func.log_density(&transform.to_external(&self.x), args)?;
        Ok(EvaluatedPoint::new(self.x, fx))
    }

    /// Converts the point's `x` from internal coordinates to external coordinates.
    pub fn to_external<T>(&self, transform: &Option<T>) -> Self
    where
        T: Transform + Clone,
    {
        Self::new(transform.to_external(&self.x).into_owned())
    }

    /// Converts the point's `x` from external coordinates to internal coordinates.
    pub fn to_internal<T>(&self, transform: &Option<T>) -> Self
    where
        T: Transform + Clone,
    {
        Self::new(transform.to_internal(&self.x).into_owned())
    }
}

impl<I: Debug> Display for Point<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "x: {:?}", self.x)
    }
}

impl From<&[Float]> for Point<DVector<Float>> {
    fn from(value: &[Float]) -> Self {
        Self::new(DVector::from_column_slice(value))
    }
}

impl From<Vec<Float>> for Point<DVector<Float>> {
    fn from(value: Vec<Float>) -> Self {
        Self::new(DVector::from_vec(value))
    }
}

impl From<DVector<Float>> for Point<DVector<Float>> {
    fn from(value: DVector<Float>) -> Self {
        Self::new(value)
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

impl EvaluatedPoint<DVector<Float>, Float> {
    /// Converts the point's `x` from internal coordinates to external coordinates.
    pub fn to_external<T>(&self, transform: &Option<T>) -> Self
    where
        T: Transform + Clone,
    {
        Self::new(transform.to_external(&self.x).into_owned(), self.fx)
    }

    /// Converts the point's `x` from external coordinates to internal coordinates.
    pub fn to_internal<T>(&self, transform: &Option<T>) -> Self
    where
        T: Transform + Clone,
    {
        Self::new(transform.to_internal(&self.x).into_owned(), self.fx)
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
    use crate::{core::Bounds, test_functions::Rosenbrock, traits::Bound};
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
        let p = Point::from(vec![1.0, 1.0]);
        let evaluated = p.evaluate(&f, &()).unwrap();
        assert_eq!(evaluated.fx, 0.0);
    }

    #[test]
    fn log_density_returns_evaluated_point() {
        let f = Rosenbrock { n: 2 };
        let p = Point::from(vec![0.0, 0.0]);
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
    fn evaluate_bounded_and_constrain_to() {
        let f = Rosenbrock { n: 2 };
        let bounds: Bounds = vec![
            Bound::LowerAndUpperBound(-2.0, 2.0),
            Bound::LowerAndUpperBound(-2.0, 2.0),
        ]
        .into();
        let p = Point::from(vec![0.0, 0.0]);
        let evaluated = p
            .evaluate_transformed(&f, &Some(bounds.clone()), &())
            .unwrap();
        assert_eq!(evaluated.fx, 1.0);

        let constrained = evaluated.to_external(&Some(bounds));
        assert_eq!(constrained.fx, evaluated.fx);
        assert!(constrained.x.len() == evaluated.x.len());
    }

    #[test]
    fn from_and_display() {
        let p = Point::from(vec![1.0, 2.0]);
        let s = format!("{}", p);
        assert!(s.contains("x:"));
    }
}
