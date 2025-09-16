use crate::{
    core::utils::SampleFloat,
    traits::{Boundable, Transform},
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
impl<'a, B> From<&'a B> for Bound
where
    B: ?Sized + ToOwned,
    Self: From<B::Owned>,
{
    fn from(value: &'a B) -> Self {
        Self::from(value.to_owned())
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
            Self::NoBound => val,
            Self::LowerBound(lb) => val.clamp(lb, val),
            Self::UpperBound(ub) => val.clamp(val, ub),
            Self::LowerAndUpperBound(lb, ub) => val.clamp(lb, ub),
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
    pub fn apply<T>(&self, transform: &T) -> Self
    where
        T: Transform,
    {
        let (l, u) = DVector::unpack(self);
        let l_int = transform.to_internal(&l);
        let u_int = transform.to_internal(&u);
        DVector::pack(l_int.as_ref(), u_int.as_ref())
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
    [B]: ToOwned<Owned = Vec<B>>,
    B: Into<Bound> + Clone,
{
    fn from(value: &[B]) -> Self {
        Self(value.iter().cloned().map(Into::into).collect())
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

impl Transform for Bounds {
    fn to_external<'a>(&'a self, z: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        z.constrain_to(Some(self))
    }

    fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        x.unconstrain_from(Some(self))
    }

    fn jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        let mut jac = DMatrix::zeros(self.0.len(), self.0.len());
        for (i, zi) in z.iter().enumerate() {
            jac[(i, i)] = match self.0[i] {
                Bound::NoBound => 1.0,
                Bound::LowerBound(_) => zi / Float::sqrt(zi.powi(2) + 1.0),
                Bound::UpperBound(_) => -zi / Float::sqrt(zi.powi(2) + 1.0),
                Bound::LowerAndUpperBound(lb, ub) => Float::cos(*zi) * (ub - lb) / 2.0,
            }
        }
        jac
    }

    fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        let mut hess = DMatrix::zeros(self.0.len(), self.0.len());
        hess[(a, a)] = match self.0[a] {
            Bound::NoBound => 0.0,
            Bound::LowerBound(_) => Float::powf(z.index(a).powi(2) + 1.0, 3.0 / 2.0),
            Bound::UpperBound(_) => -Float::powf(z.index(a).powi(2) + 1.0, 3.0 / 2.0),
            Bound::LowerAndUpperBound(lb, ub) => -Float::sin(*z.index(a)) * (ub - lb) / 2.0,
        };
        hess
    }
}

/// This transform may be used to ensure algorithms operate on a space of positive-definite matrices.
///
/// Some functions take parameters which form matrices and require that these matrices be
/// positive-definite. However, most algorithms are agnostic of the structure of such a matrix,
/// and many do not allow for complex constraints required to ensure this. One solution, given
/// here, is to provide a set of internal parameters which span the entire real line over which
/// most algorithms can easily operate along with a transformation which maps these internal
/// parameters to a space of positive-definite matrices.
///
/// The particular mapping requires us to form a lower-triangular matrix $`L`$ such that
/// $`L_{ii} > 0`$ (the off-diagonals may have any values). Then the matrix $`LL^{\intercal}`$ is
/// guaranteed to be positive-definite. To ensure the diagonal entries are strictly positive
/// and non-zero, we apply the Squareplus[^1] transformation to them parameterized by $`\beta`$
/// along with the addition of a small positive constant $`\delta`$, whose square is the minimum
/// allowed value of the diagonal entries of the resulting positive-definite matrix.
///
/// [^1]: [J. T. Barron, "Squareplus: A Softplus-Like Algebraic Rectifier," 2021, arXiv. doi: 10.48550/ARXIV.2112.11687.](https://doi.org/10.48550/ARXIV.2112.11687)
#[derive(Clone)]
pub struct SymmetricPositiveDefiniteTransform {
    dim: usize,
    indices: Vec<usize>,
    beta: Float,
    delta: Float,
}
impl SymmetricPositiveDefiniteTransform {
    /// Construct a new [`SymmetricPositiveDefiniteTransform`] given the indices representing the
    /// parameters of the matrix.
    ///
    /// Note that since we specify symmetric matrices, we expect a total of $`n = d(d+1)/2`$
    /// indices where $`d`$ is the dimension of the matrix. These indices traverse the
    /// lower-triangular part of the matrix in row-major order (left to right, top to bottom).
    ///
    /// # Panics
    ///
    /// The constructor will panic if the number of indices does not correspond to a whole-number
    /// dimension.
    pub fn new(indices: &[usize]) -> Self {
        let n = indices.len();
        let dim = (Float::sqrt((8 * n + 1) as Float) as usize - 1) / 2;
        assert_eq!(dim * (dim + 1) / 2, n, "Invalid number of indices: {n}! There should be n = d(d+1)/2 indices for a d-dimensional symmetric matrix.");
        Self {
            dim,
            indices: indices.to_vec(),
            beta: 8.0,
            delta: 1e-4,
        }
    }
    /// Set the $`\beta`$ parameter in the Squareplus[^1] transform (default = `8.0`).
    ///
    /// This parameter may need to be carefully chosen for some problems to ensure that the
    /// resulting matrices are not singular. The original paper suggests $`\beta = 4`$ or
    /// $`\beta = 4 \log(2)`$, but $`\beta = 8`$ seems to work well for some test cases.
    pub fn with_beta(mut self, beta: Float) -> Self {
        assert!(beta > 0.0);
        self.beta = beta;
        self
    }
    /// Set the $`delta`$ parameter whose square is added to the diagonal entries of the resulting
    /// positive-definite matrix (default = `1e-4`).
    pub fn with_delta(mut self, delta: Float) -> Self {
        assert!(delta > 0.0);
        self.delta = delta;
        self
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
    #[inline]
    fn softplus(&self, x_int: Float) -> Float {
        (x_int + Float::sqrt(x_int.mul_add(x_int, self.beta))) * 0.5
    }
    #[inline]
    fn dsoftplus(&self, x_int: Float) -> Float {
        0.5 * (1.0 + x_int / Float::sqrt(x_int.mul_add(x_int, self.beta)))
    }
    #[inline]
    fn ddsoftplus(&self, x_int: Float) -> Float {
        self.beta / (2.0 * Float::powf(x_int.mul_add(x_int, self.beta), 3.0 / 2.0))
    }
    #[inline]
    fn k_to_ij(&self, mut k: usize) -> (usize, usize) {
        for i in 0..self.dim {
            let row_len = i + 1;
            if k < row_len {
                return (i, k);
            }
            k -= row_len;
        }
        unreachable!("k_to_ij: invalid index {} for dimension {}", k, self.dim);
    }
    #[inline]
    fn build_l(&self, z: &DVector<Float>) -> DMatrix<Float> {
        let mut l = DMatrix::zeros(self.dim, self.dim);
        for k in 0..self.indices.len() {
            let (i, j) = self.k_to_ij(k);
            let p = self.indices[k];
            let v = z[p];
            l[(i, j)] = if i == j {
                self.delta + self.softplus(v)
            } else {
                v
            };
        }
        l
    }
}
impl Transform for SymmetricPositiveDefiniteTransform {
    fn to_external(&self, z: &DVector<Float>) -> Cow<DVector<Float>> {
        let l = self.build_l(z);
        let cov = &l * l.transpose();
        let mut out = z.clone();
        self.pack(&cov, &mut out);
        Cow::Owned(out)
    }

    fn to_internal(&self, x: &DVector<Float>) -> Cow<DVector<Float>> {
        let cov = self.unpack(x);
        #[allow(clippy::expect_used)]
        let chol = cov
            .cholesky()
            .expect("Covariance matrix not positive definite");
        let l = chol.l();
        let mut output = x.clone();
        let mut k = 0;
        for i in 0..self.dim {
            for j in 0..=i {
                let external = l[(i, j)];
                let internal = if i == j {
                    let y = (external - self.delta).max(Float::MIN_POSITIVE);
                    y - self.beta / (4.0 * y)
                } else {
                    external
                };
                output[self.indices[k]] = internal;
                k += 1;
            }
        }
        Cow::Owned(output)
    }

    fn jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        let m = z.len();
        let n = self.indices.len();
        let l = self.build_l(z);
        let mut jac = DMatrix::zeros(m, m);

        let mut pos_to_k = vec![None; m];
        for k in 0..n {
            let p = self.indices[k];
            if p < m {
                pos_to_k[p] = Some(k);
            }
        }
        for p in 0..m {
            if let Some(kp) = pos_to_k[p] {
                let (ri, cj) = self.k_to_ij(kp);
                let df = if ri == cj { self.dsoftplus(z[p]) } else { 1.0 };
                let mut dcov = DMatrix::zeros(self.dim, self.dim);
                for j in 0..self.dim {
                    dcov[(ri, j)] += df * l[(j, cj)];
                }
                for i in 0..self.dim {
                    dcov[(i, ri)] += df * l[(i, cj)];
                }
                let mut dvec = DVector::zeros(n);
                self.pack(&dcov, &mut dvec);
                for k in 0..n {
                    let row_pos = self.indices[k];
                    jac[(row_pos, p)] = dvec[k];
                }
            } else {
                jac[(p, p)] = 1.0;
            }
        }
        jac
    }

    fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        let m = z.len();
        let n = self.indices.len();
        let mut pos_to_k = vec![None; m];
        for k in 0..n {
            let p = self.indices[k];
            if p < m {
                pos_to_k[p] = Some(k);
            }
        }
        let Some(k_out) = (if a < m { pos_to_k[a] } else { None }) else {
            return DMatrix::zeros(m, m); // return 0 if not an index used in the transform
        };
        let (i_out, j_out) = self.k_to_ij(k_out);
        let l = self.build_l(z);

        let mut row = vec![0; n];
        let mut col = vec![0; n];
        let mut df = vec![0.0; n];
        let mut ddf = vec![0.0; n];
        for k in 0..n {
            let (r, c) = self.k_to_ij(k);
            row[k] = r;
            col[k] = c;
            let p = self.indices[k];
            if r == c {
                df[k] = self.dsoftplus(z[p]);
                ddf[k] = self.ddsoftplus(z[p]);
            } else {
                df[k] = 1.0;
                ddf[k] = 0.0;
            }
        }

        let mut hess_block = DMatrix::zeros(n, n);
        for p in 0..n {
            let rp = row[p];
            let cp = col[p];
            let df_p = df[p];
            for q in p..n {
                let rq = row[q];
                let cq = col[q];
                let df_q = df[q];
                let mut val = 0.0;
                if cp == cq {
                    if i_out == rp && j_out == rq {
                        val += df_p * df_q;
                    }
                    if i_out == rq && j_out == rp {
                        val += df_p * df_q;
                    }
                }
                if p == q && rp == cp {
                    let ddf_p = ddf[p];
                    if i_out == rp {
                        val += ddf_p * l[(j_out, rp)];
                    }
                    if j_out == rp {
                        val += ddf_p * l[(i_out, rp)];
                    }
                }
                hess_block[(p, q)] = val;
                if p != q {
                    hess_block[(q, p)] = val;
                }
            }
        }
        let mut hess = DMatrix::zeros(m, m);
        for p in 0..n {
            let rp = self.indices[p];
            for q in 0..n {
                let cq = self.indices[q];
                hess[(rp, cq)] = hess_block[(p, q)];
            }
        }
        hess
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::dvector;

    use super::*;
    use crate::{
        traits::{transform::TransformExt, Boundable},
        DVector,
    };

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

        let d: DVector<Float> = Boundable::random_vector_in(&bounds, &mut rng);

        assert_eq!(d.len(), bounds.len());

        assert!(d.is_in(&bounds));
    }

    #[test]
    fn test_boundable_excess_constrain_unconstrain() {
        let bounds = sample_bounds();
        let d: DVector<Float> = dvector![-1.0, 11.0, 0.0, 5.0];

        let d_excess = d.excess_from(&bounds);
        assert!(d_excess.iter().any(|x| *x != 0.0));

        let d_constrained = d.constrain_to(Some(&bounds));
        assert!(d_constrained.is_in(&bounds));

        let d_unconstrained = d_constrained.unconstrain_from(Some(&bounds));
        assert_eq!(d_unconstrained.len(), d.len());
    }

    #[test]
    fn test_bounds_container() {
        let b = Bound::LowerBound(0.0);
        let bounds: Bounds = vec![b].into();
        assert_eq!(bounds.into_inner(), vec![b]);
    }

    /// A simple offset transform: adds +1.0 on `external_to_internal`, subtracts 1.0 on `internal_to_external`.
    #[derive(Clone)]
    struct Offset;
    impl Transform for Offset {
        fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x.add_scalar(1.0))
        }
        fn to_external<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x.add_scalar(-1.0))
        }
    }

    /// Identity transform for testing borrowed vs owned behavior.
    #[derive(Clone)]
    struct Identity;
    impl Transform for Identity {
        fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Borrowed(x)
        }
        fn to_external<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Borrowed(x)
        }
    }

    /// A scaling transform: multiplies by `factor` on `external_to_internal`, divides on `internal_to_external`.
    #[derive(Clone)]
    struct Scale(Float);
    impl Transform for Scale {
        fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x * self.0)
        }
        fn to_external<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x / self.0)
        }
    }

    #[test]
    fn identity_roundtrip_borrowed() {
        let id = Identity;
        let val = DVector::from(vec![10.0]);
        let res_internal = id.to_internal(&val);
        assert!(matches!(res_internal, Cow::Borrowed(_)));
        assert_eq!(res_internal[0], 10.0);

        let res_external = id.to_external(&val);
        assert!(matches!(res_external, Cow::Borrowed(_)));
        assert_eq!(res_external[0], 10.0);
    }

    #[test]
    fn offset_roundtrip_owned() {
        let off = Offset;
        let val = DVector::from(vec![10.0]);
        let internal = off.to_internal(&val);
        assert_eq!(internal[0], 11.0);

        let external = off.to_external(&internal);
        assert_eq!(external[0], 10.0);
    }

    #[test]
    fn scale_roundtrip_owned() {
        let sc = Scale(2.0);
        let val = DVector::from(vec![4.0]);
        let internal = sc.to_internal(&val);
        assert_eq!(internal[0], 8.0);

        let external = sc.to_external(&internal);
        assert_eq!(external[0], 4.0);
    }

    #[test]
    fn chain_applies_in_sequence_offset_then_scale() {
        let off = Offset;
        let sc = Scale(2.0);
        let chain = sc.compose(off);

        let val = DVector::from(vec![3.0]);
        let res = chain.to_internal(&val);
        assert_eq!(res[0], (3.0 + 1.0) * 2.0);

        let back = chain.to_external(&res);
        assert_eq!(back[0], 3.0);
    }

    #[test]
    fn chain_applies_in_sequence_scale_then_offset() {
        let sc = Scale(2.0);
        let off = Offset;
        let chain = off.compose(sc);

        let val = DVector::from(vec![3.0]);
        let res = chain.to_internal(&val);
        assert_eq!(res[0], (3.0 * 2.0) + 1.0);

        let back = chain.to_external(&res);
        assert_eq!(back[0], 3.0);
    }

    #[test]
    fn chain_with_borrow() {
        let id = Identity;
        let chain = id.clone().compose(id);

        let val = DVector::from(vec![7.0]);
        let res = chain.to_internal(&val);
        assert!(matches!(res, Cow::Borrowed(_)));
        assert_eq!(res[0], 7.0);
    }

    #[test]
    fn apply_scale_flips_bounds() {
        let b = Bound::LowerAndUpperBound(1.0, 5.0);
        let bounds = Bounds::from(vec![b]);

        let scaled = bounds.apply(&Scale(-1.0));
        assert_eq!(scaled.len(), 1);

        match &scaled[0] {
            Bound::LowerAndUpperBound(l, u) => {
                assert_eq!((*l, *u), (-5.0, -1.0));
            }
            _ => panic!("Expected LowerAndUpperBound"),
        }
    }

    #[test]
    fn test_into_transforms() {
        let id = Identity;
        let val = DVector::from(vec![7.0]);
        let res_int = id.to_owned_internal(&val);
        let res_ext = id.to_owned_external(&val);
        assert_eq!(res_int, DVector::from(vec![7.0]));
        assert_eq!(res_ext, DVector::from(vec![7.0]));
    }
}
