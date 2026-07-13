//! Scalar- and backend-generic parameter transforms.

use crate::core::{LinearAlgebra, Matrix, RealScalar, Vector};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{CostFunction, Gradient, LogDensity};
use std::marker::PhantomData;

/// A differentiable coordinate transform used by backend-generic algorithms.
pub trait BackendTransform<T, B>: Send + Sync
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
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

/// Identity coordinate transform.
#[derive(Clone, Copy, Debug, Default)]
pub struct IdentityTransform;

impl<T, B> BackendTransform<T, B> for IdentityTransform
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

/// Backend-generic diagonal scaling transform.
#[derive(Clone, Debug)]
pub struct BackendScaleTransform<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    multipliers: Vector<T, B>,
    _backend: PhantomData<B>,
}

/// One scalar-valued parameter bound.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
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

/// Smooth backend-generic box-bounds transform.
#[derive(Clone, Debug)]
pub struct BackendBounds<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    bounds: Vec<ScalarBound<T>>,
    _backend: PhantomData<B>,
}

impl<T, B> BackendBounds<T, B>
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
            _backend: PhantomData,
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

impl<T, B> BackendTransform<T, B> for BackendBounds<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
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

impl<T, B> BackendScaleTransform<T, B>
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
            _backend: PhantomData,
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

impl<T, B> BackendTransform<T, B> for BackendScaleTransform<T, B>
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
pub struct BackendTransformedProblem<'a, P, T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    problem: &'a P,
    transform: Option<&'a dyn BackendTransform<T, B>>,
}

impl<'a, P, T, B> BackendTransformedProblem<'a, P, T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Construct an adapter around a problem and optional transform.
    pub const fn new(problem: &'a P, transform: Option<&'a dyn BackendTransform<T, B>>) -> Self {
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

impl<P, T, B, U, E> CostFunction<T, B, U, E> for BackendTransformedProblem<'_, P, T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    fn evaluate(&self, internal: &Vector<T, B>, args: &U) -> Result<T, E> {
        self.problem.evaluate(&self.to_external(internal), args)
    }
}

impl<P, T, B, U, E> Gradient<T, B, U, E> for BackendTransformedProblem<'_, P, T, B>
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

impl<P, T, B, U, E> LogDensity<T, B, U, E> for BackendTransformedProblem<'_, P, T, B>
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
    use crate::core::NalgebraBackend;
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

    #[test]
    fn identity_transform_preserves_derivatives() {
        let transform = IdentityTransform;
        let problem =
            BackendTransformedProblem::<_, f64, NalgebraBackend>::new(&Quadratic, Some(&transform));
        let x = Vector::from_vec(vec![1.0, 2.0]);
        assert_eq!(problem.evaluate(&x, &()).unwrap(), 5.0);
        assert_eq!(problem.gradient(&x, &()).unwrap().to_vec(), vec![2.0, 4.0]);
        assert_eq!(problem.hessian(&x, &()).unwrap().get(0, 0), 2.0);
    }

    #[test]
    fn scale_transform_round_trips() {
        let transform =
            BackendScaleTransform::<f32, NalgebraBackend>::from_parameter_scales([2.0, 0.5])
                .unwrap();
        let external = Vector::from_vec(vec![4.0, 3.0]);
        let internal = transform.to_internal(&external);
        assert_eq!(internal.to_vec(), vec![2.0, 6.0]);
        assert_eq!(transform.to_external(&internal), external);
    }

    #[test]
    fn bounds_transform_round_trips_all_bound_kinds() {
        let bounds = BackendBounds::<f64, NalgebraBackend>::new([
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
}
