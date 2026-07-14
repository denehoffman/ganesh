//! Scalar- and linear-algebra-generic parameter transforms.

use crate::core::{LinearAlgebra, Matrix, NalgebraProvider, RealScalar, Vector};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{CostFunction, Gradient, LogDensity};
use serde::Serialize;
use std::marker::PhantomData;

/// A differentiable coordinate transform used by linear-algebra-generic algorithms.
pub trait Transform<T, B>: Send + Sync
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Apply this transform first and `outer` second.
    fn then<X>(self, outer: X) -> TransformChain<Self, X>
    where
        Self: Sized,
        X: Transform<T, B>,
    {
        TransformChain::new(self, outer)
    }

    /// Return user-facing parameter bounds when this transform represents bounds.
    fn parameter_bounds(&self) -> Option<&[ScalarBound<T>]> {
        None
    }

    /// Convert internal coordinates to user-facing external coordinates.
    fn to_external(&self, internal: &Vector<T, B>) -> Vector<T, B>;

    /// Convert external coordinates to internal coordinates.
    fn to_internal(&self, external: &Vector<T, B>) -> Vector<T, B>;

    /// Jacobian of `to_external` at `internal`.
    fn to_external_jacobian(&self, internal: &Vector<T, B>) -> Matrix<T, B>;

    /// Hessian of external component `component` at `internal`.
    fn to_external_component_hessian(
        &self,
        component: usize,
        internal: &Vector<T, B>,
    ) -> Matrix<T, B>;

    /// Pull an external-space gradient back to internal coordinates.
    fn pullback_gradient(
        &self,
        internal: &Vector<T, B>,
        external_gradient: &Vector<T, B>,
    ) -> Vector<T, B> {
        self.to_external_jacobian(internal)
            .transpose()
            .mul_vec(external_gradient)
    }

    /// Pull an external-space Hessian back to internal coordinates.
    fn pullback_hessian(
        &self,
        internal: &Vector<T, B>,
        external_gradient: &Vector<T, B>,
        external_hessian: &Matrix<T, B>,
    ) -> Matrix<T, B> {
        let jacobian = self.to_external_jacobian(internal);
        let mut hessian = jacobian
            .transpose()
            .mul_mat(external_hessian)
            .mul_mat(&jacobian);
        for component in 0..external_gradient.len() {
            let contribution = self
                .to_external_component_hessian(component, internal)
                .scale(external_gradient.get(component));
            hessian = &hessian + &contribution;
        }
        hessian
    }
}

/// Composition of two same-dimensional coordinate transforms.
#[derive(Clone, Debug)]
pub struct TransformChain<Inner, Outer> {
    inner: Inner,
    outer: Outer,
}

impl<Inner, Outer> TransformChain<Inner, Outer> {
    /// Construct a chain that applies `inner` first and `outer` second.
    pub const fn new(inner: Inner, outer: Outer) -> Self {
        Self { inner, outer }
    }

    /// Return the transform applied first.
    pub const fn inner(&self) -> &Inner {
        &self.inner
    }

    /// Return the transform applied second.
    pub const fn outer(&self) -> &Outer {
        &self.outer
    }
}

impl<T, B, Inner, Outer> Transform<T, B> for TransformChain<Inner, Outer>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    Inner: Transform<T, B>,
    Outer: Transform<T, B>,
{
    fn parameter_bounds(&self) -> Option<&[ScalarBound<T>]> {
        self.outer.parameter_bounds()
    }

    fn to_external(&self, internal: &Vector<T, B>) -> Vector<T, B> {
        self.outer.to_external(&self.inner.to_external(internal))
    }

    fn to_internal(&self, external: &Vector<T, B>) -> Vector<T, B> {
        self.inner.to_internal(&self.outer.to_internal(external))
    }

    fn to_external_jacobian(&self, internal: &Vector<T, B>) -> Matrix<T, B> {
        let intermediate = self.inner.to_external(internal);
        self.outer
            .to_external_jacobian(&intermediate)
            .mul_mat(&self.inner.to_external_jacobian(internal))
    }

    fn to_external_component_hessian(
        &self,
        component: usize,
        internal: &Vector<T, B>,
    ) -> Matrix<T, B> {
        let intermediate = self.inner.to_external(internal);
        let inner_jacobian = self.inner.to_external_jacobian(internal);
        let outer_jacobian = self.outer.to_external_jacobian(&intermediate);
        let mut hessian = inner_jacobian
            .transpose()
            .mul_mat(
                &self
                    .outer
                    .to_external_component_hessian(component, &intermediate),
            )
            .mul_mat(&inner_jacobian);
        for intermediate_component in 0..internal.len() {
            hessian = &hessian
                + &self
                    .inner
                    .to_external_component_hessian(intermediate_component, internal)
                    .scale(outer_jacobian.get(component, intermediate_component));
        }
        hessian
    }
}

/// Identity coordinate transform.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityTransform;

impl<T, B> Transform<T, B> for IdentityTransform
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn to_external(&self, internal: &Vector<T, B>) -> Vector<T, B> {
        internal.clone()
    }

    fn to_internal(&self, external: &Vector<T, B>) -> Vector<T, B> {
        external.clone()
    }

    fn to_external_jacobian(&self, internal: &Vector<T, B>) -> Matrix<T, B> {
        Matrix::identity(internal.len())
    }

    fn to_external_component_hessian(
        &self,
        _component: usize,
        internal: &Vector<T, B>,
    ) -> Matrix<T, B> {
        Matrix::zeros(internal.len(), internal.len())
    }
}

/// Component-wise periodic coordinate transform for optimization.
///
/// Periodic coordinates are represented internally on the real line and canonically wrapped into
/// half-open external intervals. This repeated lift is not a proper target for MCMC sampling.
#[derive(Clone, Debug)]
pub struct PeriodicTransform<T = f64, B = NalgebraProvider>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    intervals: Vec<Option<(T, T)>>,
    _provider: PhantomData<B>,
}

impl<T, B> PeriodicTransform<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Construct from one optional periodic interval per coordinate.
    ///
    /// `None` leaves a coordinate unchanged. Periodic intervals are canonicalized to
    /// `[lower, upper)`.
    ///
    /// # Errors
    /// Returns a configuration error for an empty list, no periodic coordinates, non-finite
    /// endpoints, or intervals whose endpoints are not ordered.
    pub fn new<I>(intervals: I) -> GaneshResult<Self>
    where
        I: IntoIterator<Item = Option<(T, T)>>,
    {
        let intervals = intervals.into_iter().collect::<Vec<_>>();
        if intervals.is_empty() {
            return Err(GaneshError::ConfigError(
                "periodic intervals must contain at least one parameter".to_string(),
            ));
        }
        if intervals.iter().all(Option::is_none) {
            return Err(GaneshError::ConfigError(
                "periodic intervals must contain at least one periodic parameter".to_string(),
            ));
        }
        if intervals
            .iter()
            .flatten()
            .any(|(lower, upper)| !lower.is_finite() || !upper.is_finite() || lower >= upper)
        {
            return Err(GaneshError::ConfigError(
                "periodic interval endpoints must be finite and ordered".to_string(),
            ));
        }
        Ok(Self {
            intervals,
            _provider: PhantomData,
        })
    }

    /// Return the optional periodic interval for each coordinate.
    pub fn intervals(&self) -> &[Option<(T, T)>] {
        &self.intervals
    }

    fn canonicalize(&self, values: &Vector<T, B>) -> Vector<T, B> {
        assert_eq!(
            values.len(),
            self.intervals.len(),
            "periodic transform dimension mismatch"
        );
        Vector::from_vec(
            self.intervals
                .iter()
                .copied()
                .enumerate()
                .map(|(index, interval)| {
                    let value = values.get(index);
                    interval.map_or(value, |(lower, upper)| {
                        lower + (value - lower).rem_euclid(upper - lower)
                    })
                })
                .collect(),
        )
    }
}

impl<T, B> Transform<T, B> for PeriodicTransform<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn to_external(&self, internal: &Vector<T, B>) -> Vector<T, B> {
        self.canonicalize(internal)
    }

    fn to_internal(&self, external: &Vector<T, B>) -> Vector<T, B> {
        self.canonicalize(external)
    }

    fn to_external_jacobian(&self, internal: &Vector<T, B>) -> Matrix<T, B> {
        assert_eq!(
            internal.len(),
            self.intervals.len(),
            "periodic transform dimension mismatch"
        );
        Matrix::identity(internal.len())
    }

    fn to_external_component_hessian(
        &self,
        _component: usize,
        internal: &Vector<T, B>,
    ) -> Matrix<T, B> {
        assert_eq!(
            internal.len(),
            self.intervals.len(),
            "periodic transform dimension mismatch"
        );
        Matrix::zeros(internal.len(), internal.len())
    }
}

/// Generic diagonal scaling transform.
#[derive(Clone, Debug)]
pub struct ScaleTransform<T = f64, B = NalgebraProvider>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    multipliers: Vector<T, B>,
    _provider: PhantomData<B>,
}

/// One scalar-valued parameter bound.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum ScalarBound<T: RealScalar> {
    /// No finite limits.
    Unbounded,
    /// Finite lower limit.
    Lower(T),
    /// Finite upper limit.
    Upper(T),
    /// Finite lower and upper limits.
    Both(T, T),
}

impl<T: RealScalar> ScalarBound<T> {
    /// Construct a bound from possibly-infinite endpoints.
    ///
    /// # Errors
    /// Returns a configuration error for NaNs or reversed finite bounds.
    pub fn new(lower: T, upper: T) -> GaneshResult<Self> {
        if lower.is_nan() || upper.is_nan() || lower >= upper {
            return Err(GaneshError::ConfigError(
                "bounds must be ordered and must not contain NaN".to_string(),
            ));
        }
        Ok(match (lower.is_finite(), upper.is_finite()) {
            (false, false) => Self::Unbounded,
            (true, false) => Self::Lower(lower),
            (false, true) => Self::Upper(upper),
            (true, true) => Self::Both(lower, upper),
        })
    }
}

/// Smooth linear-algebra-generic box-bounds transform.
#[derive(Clone, Debug)]
pub struct Bounds<T = f64, B = NalgebraProvider>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    bounds: Vec<ScalarBound<T>>,
    _provider: PhantomData<B>,
}

impl<T, B> Bounds<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Construct bounds from lower/upper endpoint pairs.
    ///
    /// # Errors
    /// Returns a configuration error when a bound is invalid or the collection is empty.
    pub fn new<I>(bounds: I) -> GaneshResult<Self>
    where
        I: IntoIterator<Item = (T, T)>,
    {
        let bounds = bounds
            .into_iter()
            .map(|(lower, upper)| ScalarBound::new(lower, upper))
            .collect::<GaneshResult<Vec<_>>>()?;
        if bounds.is_empty() {
            return Err(GaneshError::ConfigError(
                "bounds must contain at least one parameter".to_string(),
            ));
        }
        Ok(Self {
            bounds,
            _provider: PhantomData,
        })
    }

    /// Return the parameter bounds.
    pub fn bounds(&self) -> &[ScalarBound<T>] {
        &self.bounds
    }

    fn external_component(bound: ScalarBound<T>, internal: T) -> T {
        let root = (internal * internal + T::one()).sqrt();
        match bound {
            ScalarBound::Unbounded => internal,
            ScalarBound::Lower(lower) => lower + root + internal,
            ScalarBound::Upper(upper) => upper - root + internal,
            ScalarBound::Both(lower, upper) => {
                let two = T::literal(2.0);
                let center = (lower + upper) / two;
                let width = (upper - lower) / two;
                center + width * internal / root
            }
        }
    }

    fn internal_component(bound: ScalarBound<T>, external: T) -> T {
        let two = T::literal(2.0);
        match bound {
            ScalarBound::Unbounded => external,
            ScalarBound::Lower(lower) => {
                let distance = external - lower;
                (distance - T::one() / distance) / two
            }
            ScalarBound::Upper(upper) => {
                let distance = upper - external;
                (T::one() / distance - distance) / two
            }
            ScalarBound::Both(lower, upper) => {
                let center = (lower + upper) / two;
                let width = (upper - lower) / two;
                let mut unit = (external - center) / width;
                let margin = T::epsilon().sqrt();
                let limit = T::one() - margin;
                if unit > limit {
                    unit = limit;
                } else if unit < -limit {
                    unit = -limit;
                }
                unit / (T::one() - unit * unit).sqrt()
            }
        }
    }

    fn first_derivative(bound: ScalarBound<T>, internal: T) -> T {
        let base = internal * internal + T::one();
        let root = base.sqrt();
        match bound {
            ScalarBound::Unbounded => T::one(),
            ScalarBound::Lower(_) => T::one() + internal / root,
            ScalarBound::Upper(_) => T::one() - internal / root,
            ScalarBound::Both(lower, upper) => {
                let width = (upper - lower) / T::literal(2.0);
                width / (base * root)
            }
        }
    }

    fn second_derivative(bound: ScalarBound<T>, internal: T) -> T {
        let base = internal * internal + T::one();
        let denominator = base * base.sqrt();
        match bound {
            ScalarBound::Unbounded => T::zero(),
            ScalarBound::Lower(_) => T::one() / denominator,
            ScalarBound::Upper(_) => -T::one() / denominator,
            ScalarBound::Both(lower, upper) => {
                let width = (upper - lower) / T::literal(2.0);
                -T::literal(3.0) * width * internal / (denominator * base)
            }
        }
    }
}

impl<T, B> Transform<T, B> for Bounds<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn parameter_bounds(&self) -> Option<&[ScalarBound<T>]> {
        Some(&self.bounds)
    }

    fn to_external(&self, internal: &Vector<T, B>) -> Vector<T, B> {
        assert_eq!(
            internal.len(),
            self.bounds.len(),
            "bounds dimension mismatch"
        );
        Vector::from_vec(
            self.bounds
                .iter()
                .copied()
                .enumerate()
                .map(|(index, bound)| Self::external_component(bound, internal.get(index)))
                .collect(),
        )
    }

    fn to_internal(&self, external: &Vector<T, B>) -> Vector<T, B> {
        assert_eq!(
            external.len(),
            self.bounds.len(),
            "bounds dimension mismatch"
        );
        Vector::from_vec(
            self.bounds
                .iter()
                .copied()
                .enumerate()
                .map(|(index, bound)| Self::internal_component(bound, external.get(index)))
                .collect(),
        )
    }

    fn to_external_jacobian(&self, internal: &Vector<T, B>) -> Matrix<T, B> {
        let mut jacobian = Matrix::zeros(internal.len(), internal.len());
        for (index, bound) in self.bounds.iter().copied().enumerate() {
            jacobian.set(
                index,
                index,
                Self::first_derivative(bound, internal.get(index)),
            );
        }
        jacobian
    }

    fn to_external_component_hessian(
        &self,
        component: usize,
        internal: &Vector<T, B>,
    ) -> Matrix<T, B> {
        let mut hessian = Matrix::zeros(internal.len(), internal.len());
        hessian.set(
            component,
            component,
            Self::second_derivative(self.bounds[component], internal.get(component)),
        );
        hessian
    }
}

impl<T, B> ScaleTransform<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Construct from external-to-internal multipliers.
    ///
    /// # Errors
    /// Returns a configuration error for empty, zero, or non-finite multipliers.
    pub fn from_multipliers<I>(multipliers: I) -> GaneshResult<Self>
    where
        I: IntoIterator<Item = T>,
    {
        let multipliers = multipliers.into_iter().collect::<Vec<_>>();
        if multipliers.is_empty()
            || multipliers
                .iter()
                .any(|value| !value.is_finite() || *value == T::zero())
        {
            return Err(GaneshError::ConfigError(
                "scale multipliers must be nonempty, finite, and nonzero".to_string(),
            ));
        }
        Ok(Self {
            multipliers: Vector::from_vec(multipliers),
            _provider: PhantomData,
        })
    }

    /// Construct from characteristic external parameter scales.
    ///
    /// # Errors
    /// Returns a configuration error for invalid scales.
    pub fn from_parameter_scales<I>(scales: I) -> GaneshResult<Self>
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_multipliers(scales.into_iter().map(|scale| T::one() / scale))
    }

    /// Return external-to-internal multipliers.
    pub const fn multipliers(&self) -> &Vector<T, B> {
        &self.multipliers
    }
}

impl<T, B> Transform<T, B> for ScaleTransform<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn to_external(&self, internal: &Vector<T, B>) -> Vector<T, B> {
        Vector::from_vec(
            (0..internal.len())
                .map(|index| internal.get(index) / self.multipliers.get(index))
                .collect(),
        )
    }

    fn to_internal(&self, external: &Vector<T, B>) -> Vector<T, B> {
        Vector::from_vec(
            (0..external.len())
                .map(|index| external.get(index) * self.multipliers.get(index))
                .collect(),
        )
    }

    fn to_external_jacobian(&self, internal: &Vector<T, B>) -> Matrix<T, B> {
        let mut jacobian = Matrix::zeros(internal.len(), internal.len());
        for index in 0..internal.len() {
            jacobian.set(index, index, T::one() / self.multipliers.get(index));
        }
        jacobian
    }

    fn to_external_component_hessian(
        &self,
        _component: usize,
        internal: &Vector<T, B>,
    ) -> Matrix<T, B> {
        Matrix::zeros(internal.len(), internal.len())
    }
}

/// Objective adapter that presents external-space functions in internal coordinates.
pub struct TransformedProblem<'a, P, T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    problem: &'a P,
    transform: Option<&'a dyn Transform<T, B>>,
}

impl<'a, P, T, B> TransformedProblem<'a, P, T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Construct an adapter around a problem and optional transform.
    pub const fn new(problem: &'a P, transform: Option<&'a dyn Transform<T, B>>) -> Self {
        Self { problem, transform }
    }

    /// Convert external coordinates to the algorithm's internal coordinates.
    pub fn to_internal(&self, external: &Vector<T, B>) -> Vector<T, B> {
        self.transform.map_or_else(
            || external.clone(),
            |transform| transform.to_internal(external),
        )
    }

    /// Convert internal coordinates to external coordinates.
    pub fn to_external(&self, internal: &Vector<T, B>) -> Vector<T, B> {
        self.transform.map_or_else(
            || internal.clone(),
            |transform| transform.to_external(internal),
        )
    }
}

impl<P, T, B, U, E> CostFunction<T, B, U, E> for TransformedProblem<'_, P, T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    fn evaluate(&self, internal: &Vector<T, B>, args: &U) -> Result<T, E> {
        self.problem.evaluate(&self.to_external(internal), args)
    }
}

impl<P, T, B, U, E> Gradient<T, B, U, E> for TransformedProblem<'_, P, T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: Gradient<T, B, U, E>,
{
    fn gradient(&self, internal: &Vector<T, B>, args: &U) -> Result<Vector<T, B>, E> {
        let external = self.to_external(internal);
        let gradient = self.problem.gradient(&external, args)?;
        Ok(self.transform.map_or_else(
            || gradient.clone(),
            |transform| transform.pullback_gradient(internal, &gradient),
        ))
    }

    fn hessian(&self, internal: &Vector<T, B>, args: &U) -> Result<Matrix<T, B>, E> {
        let external = self.to_external(internal);
        let (gradient, hessian) = self.problem.gradient_with_hessian(&external, args)?;
        Ok(self.transform.map_or_else(
            || hessian.clone(),
            |transform| transform.pullback_hessian(internal, &gradient, &hessian),
        ))
    }
}

impl<P, T, B, U, E> LogDensity<T, B, U, E> for TransformedProblem<'_, P, T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: LogDensity<T, B, U, E>,
{
    fn log_density(&self, internal: &Vector<T, B>, args: &U) -> Result<T, E> {
        self.problem.log_density(&self.to_external(internal), args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::NalgebraProvider;
    use std::convert::Infallible;

    struct Quadratic;

    impl CostFunction for Quadratic {
        fn evaluate(&self, x: &Vector<f64>, _: &()) -> Result<f64, Infallible> {
            Ok(x.dot(x))
        }
    }

    impl Gradient for Quadratic {
        fn gradient(&self, x: &Vector<f64>, _: &()) -> Result<Vector<f64>, Infallible> {
            Ok(x.scale(2.0))
        }

        fn hessian(&self, x: &Vector<f64>, _: &()) -> Result<Matrix<f64>, Infallible> {
            Ok(Matrix::identity(x.len()).scale(2.0))
        }
    }

    struct CircularObjective {
        target: f64,
    }

    impl CostFunction for CircularObjective {
        fn evaluate(&self, x: &Vector<f64>, _: &()) -> Result<f64, Infallible> {
            Ok(1.0 - (x.get(0) - self.target).cos())
        }
    }

    impl Gradient for CircularObjective {
        fn gradient(&self, x: &Vector<f64>, _: &()) -> Result<Vector<f64>, Infallible> {
            Ok(Vector::from_vec(vec![(x.get(0) - self.target).sin()]))
        }

        fn hessian(&self, x: &Vector<f64>, _: &()) -> Result<Matrix<f64>, Infallible> {
            Ok(Matrix::from_vec(1, 1, vec![(x.get(0) - self.target).cos()]))
        }
    }

    #[test]
    fn identity_transform_preserves_derivatives() {
        let transform = IdentityTransform;
        let problem =
            TransformedProblem::<_, f64, NalgebraProvider>::new(&Quadratic, Some(&transform));
        let x = Vector::from_vec(vec![1.0, 2.0]);
        assert_eq!(problem.evaluate(&x, &()).unwrap(), 5.0);
        assert_eq!(problem.gradient(&x, &()).unwrap().to_vec(), vec![2.0, 4.0]);
        assert_eq!(problem.hessian(&x, &()).unwrap().get(0, 0), 2.0);
    }

    #[test]
    fn scale_transform_round_trips() {
        let transform =
            ScaleTransform::<f32, NalgebraProvider>::from_parameter_scales([2.0, 0.5]).unwrap();
        let external = Vector::from_vec(vec![4.0, 3.0]);
        let internal = transform.to_internal(&external);
        assert_eq!(internal.to_vec(), vec![2.0, 6.0]);
        assert_eq!(transform.to_external(&internal), external);
    }

    #[test]
    fn bounds_transform_round_trips_all_bound_kinds() {
        let bounds = Bounds::<f64, NalgebraProvider>::new([
            (-1.0, 2.0),
            (0.0, f64::INFINITY),
            (f64::NEG_INFINITY, 3.0),
            (f64::NEG_INFINITY, f64::INFINITY),
        ])
        .unwrap();
        let external = Vector::from_vec(vec![0.25, 1.5, 1.0, -2.0]);
        let internal = bounds.to_internal(&external);
        let roundtrip = bounds.to_external(&internal);
        for index in 0..external.len() {
            assert!((roundtrip.get(index) - external.get(index)).abs() < 1e-12);
        }
    }

    #[test]
    fn periodic_transform_wraps_mixed_coordinates() {
        let transform = PeriodicTransform::<f64>::new([
            Some((-std::f64::consts::PI, std::f64::consts::PI)),
            None,
            Some((0.0, 360.0)),
        ])
        .unwrap();
        let values = Vector::from_vec(vec![3.0 * std::f64::consts::PI, -4.5, -1080.0 + 15.0]);
        let wrapped = transform.to_external(&values);
        assert!((wrapped.get(0) + std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(wrapped.get(1), -4.5);
        assert_eq!(wrapped.get(2), 15.0);
        assert_eq!(transform.to_internal(&wrapped), wrapped);
        assert_eq!(transform.to_external_jacobian(&values), Matrix::identity(3));
        assert_eq!(
            transform.to_external_component_hessian(0, &values),
            Matrix::zeros(3, 3)
        );
    }

    #[test]
    fn periodic_transform_uses_half_open_intervals_for_f32() {
        let transform = PeriodicTransform::<f32>::new([Some((-2.0, 3.0))]).unwrap();
        let lower = Vector::from_vec(vec![-2.0]);
        let upper = Vector::from_vec(vec![3.0]);
        assert_eq!(transform.to_external(&lower).get(0), -2.0);
        assert_eq!(transform.to_external(&upper).get(0), -2.0);
        assert_eq!(
            transform
                .to_external(&Vector::from_vec(vec![5_000_001.0]))
                .get(0),
            1.0
        );
    }

    #[test]
    fn periodic_transform_rejects_invalid_configurations() {
        assert!(PeriodicTransform::<f64>::new([]).is_err());
        assert!(PeriodicTransform::<f64>::new([None]).is_err());
        assert!(PeriodicTransform::<f64>::new([Some((0.0, 0.0))]).is_err());
        assert!(PeriodicTransform::<f64>::new([Some((1.0, -1.0))]).is_err());
        assert!(PeriodicTransform::<f64>::new([Some((0.0, f64::INFINITY))]).is_err());
    }

    #[test]
    fn periodic_objective_preserves_gradient_and_hessian_across_seam() {
        let transform =
            PeriodicTransform::<f64>::new([Some((-std::f64::consts::PI, std::f64::consts::PI))])
                .unwrap();
        let objective = CircularObjective {
            target: -std::f64::consts::PI + 0.2,
        };
        let transformed =
            TransformedProblem::<_, f64, NalgebraProvider>::new(&objective, Some(&transform));
        let seam = Vector::from_vec(vec![std::f64::consts::PI]);
        let epsilon = 1e-4;
        let plus = Vector::from_vec(vec![std::f64::consts::PI + epsilon]);
        let minus = Vector::from_vec(vec![std::f64::consts::PI - epsilon]);
        let center_value = transformed.evaluate(&seam, &()).unwrap();
        let plus_value = transformed.evaluate(&plus, &()).unwrap();
        let minus_value = transformed.evaluate(&minus, &()).unwrap();
        let numerical_gradient = (plus_value - minus_value) / (2.0 * epsilon);
        let numerical_hessian =
            (2.0f64.mul_add(-center_value, plus_value) + minus_value) / (epsilon * epsilon);
        assert!(
            (transformed.gradient(&seam, &()).unwrap().get(0) - numerical_gradient).abs() < 1e-8
        );
        assert!(
            (transformed.hessian(&seam, &()).unwrap().get(0, 0) - numerical_hessian).abs() < 1e-7
        );
    }

    #[test]
    fn transform_chain_matches_numerical_derivatives() {
        let inner = ScaleTransform::<f64>::from_parameter_scales([2.0, 0.5]).unwrap();
        let outer = Bounds::<f64>::new([(-1.0, 3.0), (f64::NEG_INFINITY, f64::INFINITY)]).unwrap();
        let chain = inner.then(outer);
        let x = Vector::from_vec(vec![0.35, -0.8]);
        let external = chain.to_external(&x);
        let roundtrip = chain.to_internal(&external);
        for index in 0..x.len() {
            assert!((roundtrip.get(index) - x.get(index)).abs() < 1e-12);
        }
        assert_eq!(chain.parameter_bounds(), chain.outer().parameter_bounds());

        let epsilon = 1e-5;
        let jacobian = chain.to_external_jacobian(&x);
        for row in 0..2 {
            for col in 0..2 {
                let mut plus = x.clone();
                let mut minus = x.clone();
                plus.set(col, plus.get(col) + epsilon);
                minus.set(col, minus.get(col) - epsilon);
                let numerical = (chain.to_external(&plus).get(row)
                    - chain.to_external(&minus).get(row))
                    / (2.0 * epsilon);
                assert!((jacobian.get(row, col) - numerical).abs() < 1e-7);
            }
        }

        for component in 0..2 {
            let hessian = chain.to_external_component_hessian(component, &x);
            for row in 0..2 {
                for col in 0..2 {
                    let mut plus = x.clone();
                    let mut minus = x.clone();
                    plus.set(col, plus.get(col) + epsilon);
                    minus.set(col, minus.get(col) - epsilon);
                    let numerical = (chain.to_external_jacobian(&plus).get(component, row)
                        - chain.to_external_jacobian(&minus).get(component, row))
                        / (2.0 * epsilon);
                    assert!((hessian.get(row, col) - numerical).abs() < 1e-6);
                }
            }
        }

        let transformed =
            TransformedProblem::<_, f64, NalgebraProvider>::new(&Quadratic, Some(&chain));
        let gradient = transformed.gradient(&x, &()).unwrap();
        let hessian = transformed.hessian(&x, &()).unwrap();
        for row in 0..2 {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus.set(row, plus.get(row) + epsilon);
            minus.set(row, minus.get(row) - epsilon);
            let numerical_gradient = (transformed.evaluate(&plus, &()).unwrap()
                - transformed.evaluate(&minus, &()).unwrap())
                / (2.0 * epsilon);
            assert!((gradient.get(row) - numerical_gradient).abs() < 1e-7);

            let plus_gradient = transformed.gradient(&plus, &()).unwrap();
            let minus_gradient = transformed.gradient(&minus, &()).unwrap();
            for col in 0..2 {
                let numerical_hessian =
                    (plus_gradient.get(col) - minus_gradient.get(col)) / (2.0 * epsilon);
                assert!((hessian.get(col, row) - numerical_hessian).abs() < 1e-6);
            }
        }
    }
}
