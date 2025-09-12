use crate::{
    core::utils::SampleFloat,
    traits::{transform::Transform, Boundable},
    DMatrix, DVector, Float,
};
use fastrand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    fmt::Display,
    ops::{Deref, DerefMut},
};

/// An enum that describes a bound/limit on a parameter in a minimization.
///
/// [`Bound`]s take a generic `T` which represents some scalar numeric value. They can be used by
/// bounded algorithms directly, or by some unbounded algorithms using parameter space
/// transformations.
#[derive(Default, Copy, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Bound {
    #[default]
    /// `(-inf, +inf)`
    NoBound,
    /// `(min, +inf)`
    LowerBound(Float),
    /// `(-inf, max)`
    UpperBound(Float),
    /// `(min, max)`
    LowerAndUpperBound(Float, Float),
}
impl Display for Bound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.lower(), self.upper())
    }
}
impl From<(Float, Float)> for Bound {
    fn from(value: (Float, Float)) -> Self {
        let (l, u) = if value.0 < value.1 {
            value
        } else {
            (value.1, value.0)
        };
        match (l.is_finite(), u.is_finite()) {
            (true, true) => Self::LowerAndUpperBound(l, u),
            (true, false) => Self::LowerBound(l),
            (false, true) => Self::UpperBound(u),
            (false, false) => Self::NoBound,
        }
    }
}
impl From<(&Float, &Float)> for Bound {
    fn from(value: (&Float, &Float)) -> Self {
        let (l, u) = if value.0 < value.1 {
            value
        } else {
            (value.1, value.0)
        };
        match (l.is_finite(), u.is_finite()) {
            (true, true) => Self::LowerAndUpperBound(*l, *u),
            (true, false) => Self::LowerBound(*l),
            (false, true) => Self::UpperBound(*u),
            (false, false) => Self::NoBound,
        }
    }
}
impl From<(Option<Float>, Option<Float>)> for Bound {
    fn from(value: (Option<Float>, Option<Float>)) -> Self {
        match (value.0, value.1) {
            (Some(a), Some(b)) => {
                if a < b {
                    Self::LowerAndUpperBound(a, b)
                } else {
                    Self::LowerAndUpperBound(b, a)
                }
            }
            (Some(lb), None) => Self::LowerBound(lb),
            (None, Some(ub)) => Self::UpperBound(ub),
            (None, None) => Self::NoBound,
        }
    }
}
impl<B> From<&B> for Bound
where
    B: Into<Bound>,
{
    fn from(value: &B) -> Self {
        value.into()
    }
}

impl Bound {
    /// Get a value in the uniform distribution between `lower` and `upper`.
    #[cfg(not(feature = "f32"))]
    pub fn get_uniform(&self, rng: &mut Rng) -> Float {
        rng.range(self.lower(), self.upper()) as Float
    }
    /// Get a value in the uniform distribution between `lower` and `upper`.
    #[cfg(feature = "f32")]
    pub fn get_uniform(&self, rng: &mut Rng) -> Float {
        rng.f32_range(self.lower()..self.upper()) as Float
    }
    /// Checks whether the given `value` is compatible with the bounds.
    pub fn contains(&self, value: Float) -> bool {
        match self {
            Self::NoBound => true,
            Self::LowerBound(lb) => value >= *lb,
            Self::UpperBound(ub) => value <= *ub,
            Self::LowerAndUpperBound(lb, ub) => value >= *lb && value <= *ub,
        }
    }
    /// Checks whether the given `value` is compatible with the bound and returns `0.0` if it is,
    /// and the distance to the bound otherwise signed by whether the bound is a lower (`-`) or
    /// upper (`+`) bound.
    pub fn bound_excess(&self, value: Float) -> Float {
        match self {
            Self::NoBound => 0.0,
            Self::LowerBound(lb) => {
                if value >= *lb {
                    0.0
                } else {
                    value - lb
                }
            }
            Self::UpperBound(ub) => {
                if value <= *ub {
                    0.0
                } else {
                    value - ub
                }
            }
            Self::LowerAndUpperBound(lb, ub) => {
                if value < *lb {
                    value - lb
                } else if value > *ub {
                    value - ub
                } else {
                    0.0
                }
            }
        }
    }
    /// Returns the lower bound or `-inf` if there is none.
    pub const fn lower(&self) -> Float {
        match self {
            Self::NoBound => Float::NEG_INFINITY,
            Self::LowerBound(lb) => *lb,
            Self::UpperBound(_) => Float::NEG_INFINITY,
            Self::LowerAndUpperBound(lb, _) => *lb,
        }
    }
    /// Returns the upper bound or `+inf` if there is none.
    pub const fn upper(&self) -> Float {
        match self {
            Self::NoBound => Float::INFINITY,
            Self::LowerBound(_) => Float::INFINITY,
            Self::UpperBound(ub) => *ub,
            Self::LowerAndUpperBound(_, ub) => *ub,
        }
    }
    /// Checks if the given value is equal to one of the bounds.
    ///
    /// TODO: his just does equality comparison right now, which probably needs to be improved
    /// to something with an epsilon (significant but not critical to most fits right now).
    pub fn at_bound(&self, value: Float) -> bool {
        match self {
            Self::NoBound => false,
            Self::LowerBound(lb) => value == *lb,
            Self::UpperBound(ub) => value == *ub,
            Self::LowerAndUpperBound(lb, ub) => value == *lb || value == *ub,
        }
    }
    /// Converts an unbounded "external" parameter into a bounded "internal" one via the transform:
    ///
    /// Upper and lower bounds:
    /// ```math
    /// x_\text{int} = \arcsin\left(2\frac{x_\text{ext} - x_\text{min}}{x_\text{max} - x_\text{min}} - 1\right)
    /// ```
    /// Upper bound only:
    /// ```math
    /// x_\text{int} = \sqrt{(x_\text{max} - x_\text{ext} + 1)^2 - 1}
    /// ```
    /// Lower bound only:
    /// ```math
    /// x_\text{int} = \sqrt{(x_\text{ext} - x_\text{min} + 1)^2 - 1}
    /// ```
    pub fn to_bounded(&self, val: Float) -> Float {
        match *self {
            Self::LowerBound(lb) => lb - 1.0 + Float::sqrt(Float::powi(val, 2) + 1.0),
            Self::UpperBound(ub) => ub + 1.0 - Float::sqrt(Float::powi(val, 2) + 1.0),
            Self::LowerAndUpperBound(lb, ub) => lb + (Float::sin(val) + 1.0) * (ub - lb) / 2.0,
            Self::NoBound => val,
        }
    }
    /// Converts a bounded "internal" parameter into an unbounded "external" one via the transform:
    ///
    /// Upper and lower bounds:
    /// ```math
    /// x_\text{ext} = x_\text{min} + \left(\sin(x_\text{int}) + 1\right)\frac{x_\text{max} - x_\text{min}}{2}
    /// ```
    /// Upper bound only:
    /// ```math
    /// x_\text{ext} = x_\text{max} + 1 - \sqrt{x_\text{int}^2 + 1}
    /// ```
    /// Lower bound only:
    /// ```math
    /// x_\text{ext} = x_\text{min} - 1 + \sqrt{x_\text{int}^2 + 1}
    /// ```
    pub fn to_unbounded(&self, val: Float) -> Float {
        match *self {
            Self::LowerBound(lb) => Float::sqrt(Float::powi(val - lb + 1.0, 2) - 1.0),
            Self::UpperBound(ub) => Float::sqrt(Float::powi(ub - val + 1.0, 2) - 1.0),
            Self::LowerAndUpperBound(lb, ub) => Float::asin(2.0 * (val - lb) / (ub - lb) - 1.0),
            Self::NoBound => val,
        }
    }

    /// Clips a value to be within the given bounds.
    pub fn clip_value(&self, val: Float) -> Float {
        match *self {
            Bound::NoBound => val,
            Bound::LowerBound(lb) => val.clamp(lb, val),
            Bound::UpperBound(ub) => val.clamp(val, ub),
            Bound::LowerAndUpperBound(lb, ub) => val.clamp(lb, ub),
        }
    }
}

/// A struct that contains a list of [`Bound`]s.
#[derive(Default, Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Bounds(Vec<Bound>);

impl Bounds {
    /// Returns the inner Vector of bounds.
    pub fn into_inner(self) -> Vec<Bound> {
        self.0
    }
    /// Applies a coordinate transform to the bounds.
    pub fn apply<T, I>(&self, transform: T) -> Self
    where
        T: Transform<I>,
        I: Boundable + Clone,
    {
        let (l, u) = I::unpack(self);
        let l_int = transform.exterior_to_interior(&l);
        let u_int = transform.exterior_to_interior(&u);
        I::pack(l_int.as_ref(), u_int.as_ref())
    }
}

impl<B> From<Vec<B>> for Bounds
where
    B: Into<Bound>,
{
    fn from(value: Vec<B>) -> Self {
        Self(value.into_iter().map(Into::into).collect())
    }
}
impl<B> From<&[B]> for Bounds
where
    B: Into<Bound>,
{
    fn from(value: &[B]) -> Self {
        Self(value.into_iter().map(Into::into).collect())
    }
}
impl<const N: usize, B> From<[B; N]> for Bounds
where
    B: Into<Bound>,
{
    fn from(value: [B; N]) -> Self {
        Self(value.into_iter().map(Into::into).collect())
    }
}

impl Deref for Bounds {
    type Target = Vec<Bound>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Bounds {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Transform<T> for Bounds
where
    T: Boundable + Clone,
{
    fn exterior_to_interior<'a>(&'a self, x: &'a T) -> Cow<'a, T> {
        x.unconstrain_from(Some(self))
    }

    fn interior_to_exterior<'a>(&'a self, x: &'a T) -> Cow<'a, T> {
        x.constrain_to(Some(self))
    }
}

/// This transform may be used to ensure algorithms operate on a space of positive-semidefinite matrices.
///
/// Some functions take parameters which form matrices and require that these matrices be
/// positive-semidefinite. However, most algorithms are agnostic of the structure of such a matrix,
/// and many do not allow for complex constraints required to ensure this. One solution, given
/// here, is to provide a set of internal parameters which span the entire real line over which
/// most algorithms can easily operate along with a transformation which maps these internal
/// parameters to a space of positive-semidefinite matrices.
///
/// The particular mapping requires us to form a lower-triangular matrix $`L`$ such that
/// $`L_{ii} > 0`$ (the off-diagonals may have any values). Then the matrix $`LL^{\intercal}`$ is
/// guaranteed to be positive-semidefinite. To ensure the diagonal entries are strictly positive
/// and non-zero, we apply the softplus transformation to them: $`\ln(1 + e^x)`$.
#[derive(Clone)]
pub struct SymmetricPositiveSemidefiniteTransform {
    dim: usize,
    indices: Vec<usize>,
}
impl SymmetricPositiveSemidefiniteTransform {
    /// Construct a new [`SymmetricPositiveSemidefiniteTransform`] given the indices representing the
    /// parameters of the matrix.
    ///
    /// Note that since we specify symmetric matrices, we expect a total of $`n = d(d+1)/2`$
    /// indices where $`d`$ is the dimension of the matrix. These indices traverse the
    /// lower-triangular part of the matrix in row-major order (left to right, top to bottom).
    pub fn new(indices: &[usize]) -> Self {
        let n = indices.len();
        let dim = (Float::sqrt((8 * n + 1) as Float) as usize - 1) / 2;
        assert_eq!(dim * (dim + 1) / 2, n, "Invalid number of indices: {n}! There should be n = d(d+1)/2 indices for a d-dimensional symmetric matrix.");
        Self {
            dim,
            indices: indices.to_vec(),
        }
    }
    fn pack(&self, input: &DMatrix<Float>, output: &mut DVector<Float>) {
        let mut k = 0;
        for i in 0..self.dim {
            for j in 0..=i {
                output[self.indices[k]] = input[(i, j)];
                k += 1;
            }
        }
    }
    fn unpack(&self, input: &DVector<Float>) -> DMatrix<Float> {
        let mut output = DMatrix::zeros(self.dim, self.dim);
        let mut k = 0;
        for i in 0..self.dim {
            for j in 0..=i {
                let val = input[self.indices[k]];
                output[(i, j)] = val;
                output[(j, i)] = val;
                k += 1;
            }
        }
        output
    }
}
impl Transform<DVector<Float>> for SymmetricPositiveSemidefiniteTransform {
    fn exterior_to_interior(&self, x: &DVector<Float>) -> Cow<DVector<Float>> {
        let cov = self.unpack(&x);
        #[allow(clippy::expect_used)]
        let chol = cov
            .cholesky()
            .expect("Covariance matrix not positive definite");
        let l = chol.l();
        let mut output = x.clone();
        let mut k = 0;
        for i in 0..self.dim {
            for j in 0..=i {
                let exterior = l[(i, j)];
                let interior = if i == j {
                    (exterior.exp_m1()).ln()
                } else {
                    exterior
                };
                output[self.indices[k]] = interior;
                k += 1;
            }
        }
        Cow::Owned(output)
    }

    fn interior_to_exterior(&self, x: &DVector<Float>) -> Cow<DVector<Float>> {
        let mut l = DMatrix::zeros(self.dim, self.dim);
        let mut k = 0;
        for i in 0..self.dim {
            for j in 0..=i {
                let interior = x[self.indices[k]];
                l[(i, j)] = if i == j {
                    interior.exp().ln_1p()
                } else {
                    interior
                };
                k += 1;
            }
        }
        let cov = &l * l.transpose();
        let mut out = x.clone();
        self.pack(&cov, &mut out);
        Cow::Owned(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{traits::Boundable, DVector};

    fn sample_bounds() -> Bounds {
        vec![
            Bound::LowerBound(0.0),
            Bound::UpperBound(10.0),
            Bound::LowerAndUpperBound(-1.0, 1.0),
            Bound::NoBound,
        ]
        .into()
    }

    #[test]
    fn test_bound_contains_and_excess() {
        let b1 = Bound::LowerBound(0.0);
        assert!(b1.contains(1.0));
        assert!(!b1.contains(-1.0));
        assert_eq!(b1.bound_excess(-1.0), -1.0);

        let b2 = Bound::UpperBound(5.0);
        assert!(b2.contains(4.0));
        assert!(!b2.contains(6.0));
        assert_eq!(b2.bound_excess(6.0), 1.0);

        let b3 = Bound::LowerAndUpperBound(-1.0, 1.0);
        assert!(b3.contains(0.0));
        assert!(!b3.contains(2.0));
    }

    #[test]
    fn test_bound_lower_upper_at_bound() {
        let b = Bound::LowerAndUpperBound(-2.0, 3.0);
        assert_eq!(b.lower(), -2.0);
        assert_eq!(b.upper(), 3.0);
        assert!(b.at_bound(-2.0));
        assert!(b.at_bound(3.0));
        assert!(!b.at_bound(0.0));
    }

    #[test]
    fn test_bound_transformations() {
        let b = Bound::LowerAndUpperBound(0.0, 2.0);
        let val = 1.0;
        let bounded = b.to_bounded(val);
        let unbounded = b.to_unbounded(bounded);
        assert!((val - unbounded).abs() < 1e-6);
    }

    #[test]
    fn test_boundable_random_and_is_in() {
        let mut rng = Rng::with_seed(0);
        let bounds: Bounds = vec![
            Bound::LowerAndUpperBound(-1.0, 1.0),
            Bound::LowerAndUpperBound(0.0, 5.0),
            Bound::LowerAndUpperBound(10.0, 20.0),
        ]
        .into();

        let v: Vec<Float> = Boundable::random_vector_in(&bounds, &mut rng);
        let d: DVector<Float> = Boundable::random_vector_in(&bounds, &mut rng);

        assert_eq!(v.len(), bounds.len());
        assert_eq!(d.len(), bounds.len());

        assert!(v.is_in(&bounds));
        assert!(d.is_in(&bounds));
    }

    #[test]
    fn test_boundable_excess_constrain_unconstrain() {
        let bounds = sample_bounds();
        let v: Vec<Float> = vec![-1.0, 11.0, 0.0, 5.0];
        let d: DVector<Float> = v.clone().into();

        let v_excess = v.excess_from(&bounds);
        let d_excess = d.excess_from(&bounds);
        assert!(v_excess.iter().any(|x| *x != 0.0));
        assert!(d_excess.iter().any(|x| *x != 0.0));

        let v_constrained = v.constrain_to(Some(&bounds));
        let d_constrained = d.constrain_to(Some(&bounds));
        assert!(v_constrained.is_in(&bounds));
        assert!(d_constrained.is_in(&bounds));

        let v_unconstrained = v_constrained.unconstrain_from(Some(&bounds));
        let d_unconstrained = d_constrained.unconstrain_from(Some(&bounds));
        assert_eq!(v_unconstrained.len(), v.len());
        assert_eq!(d_unconstrained.len(), d.len());
    }

    #[test]
    fn test_bounds_container() {
        let b = Bound::LowerBound(0.0);
        let bounds: Bounds = vec![b].into();
        assert_eq!(bounds.into_inner(), vec![b]);
    }

    /// A simple offset transform: adds +1.0 on `exterior_to_interior`, subtracts 1.0 on `interior_to_exterior`.
    #[derive(Clone)]
    struct Offset;
    impl Transform<DVector<Float>> for Offset {
        fn exterior_to_interior<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x.add_scalar(1.0))
        }
        fn interior_to_exterior<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x.add_scalar(-1.0))
        }
    }

    /// Identity transform for testing borrowed vs owned behavior.
    #[derive(Clone)]
    struct Identity;
    impl Transform<DVector<Float>> for Identity {
        fn exterior_to_interior<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Borrowed(x)
        }
        fn interior_to_exterior<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Borrowed(x)
        }
    }

    /// A scaling transform: multiplies by `factor` on `exterior_to_interior`, divides on `interior_to_exterior`.
    #[derive(Clone)]
    struct Scale(Float);
    impl Transform<DVector<Float>> for Scale {
        fn exterior_to_interior<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x * self.0)
        }
        fn interior_to_exterior<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x / self.0)
        }
    }

    #[test]
    fn identity_roundtrip_borrowed() {
        let id = Identity;
        let val = DVector::from(vec![10.0]);
        let res_interior = id.exterior_to_interior(&val);
        assert!(matches!(res_interior, Cow::Borrowed(_)));
        assert_eq!(res_interior[0], 10.0);

        let res_exterior = id.interior_to_exterior(&val);
        assert!(matches!(res_exterior, Cow::Borrowed(_)));
        assert_eq!(res_exterior[0], 10.0);
    }

    #[test]
    fn offset_roundtrip_owned() {
        let off = Offset;
        let val = DVector::from(vec![10.0]);
        let interior = off.exterior_to_interior(&val);
        assert_eq!(interior[0], 11.0);

        let exterior = off.interior_to_exterior(&interior);
        assert_eq!(exterior[0], 10.0);
    }

    #[test]
    fn scale_roundtrip_owned() {
        let sc = Scale(2.0);
        let val = DVector::from(vec![4.0]);
        let interior = sc.exterior_to_interior(&val);
        assert_eq!(interior[0], 8.0);

        let exterior = sc.interior_to_exterior(&interior);
        assert_eq!(exterior[0], 4.0);
    }

    #[test]
    fn chain_applies_in_sequence_offset_then_scale() {
        let off = Offset;
        let sc = Scale(2.0);
        let chain = off.chain(&sc);

        let val = DVector::from(vec![3.0]);
        let res = chain.exterior_to_interior(&val);
        assert_eq!(res[0], (3.0 + 1.0) * 2.0);

        let back = chain.interior_to_exterior(&res);
        assert_eq!(back[0], 3.0);
    }

    #[test]
    fn chain_applies_in_sequence_scale_then_offset() {
        let sc = Scale(2.0);
        let off = Offset;
        let chain = sc.chain(&off);

        let val = DVector::from(vec![3.0]);
        let res = chain.exterior_to_interior(&val);
        assert_eq!(res[0], (3.0 * 2.0) + 1.0);

        let back = chain.interior_to_exterior(&res);
        assert_eq!(back[0], 3.0);
    }

    #[test]
    fn chain_with_borrow() {
        let id = Identity;
        let chain = id.chain(&id);

        let val = DVector::from(vec![7.0]);
        let res = chain.exterior_to_interior(&val);
        assert!(matches!(res, Cow::Borrowed(_)));
        assert_eq!(res[0], 7.0);
    }

    #[test]
    fn apply_scale_flips_bounds() {
        let b = Bound::LowerAndUpperBound(1.0, 5.0);
        let bounds = Bounds::from(vec![b]);

        let scaled = bounds.apply(Scale(-1.0));
        assert_eq!(scaled.len(), 1);

        match &scaled[0] {
            Bound::LowerAndUpperBound(l, u) => {
                assert_eq!((*l, *u), (-5.0, -1.0));
            }
            _ => panic!("Expected LowerAndUpperBound"),
        }
    }
}
