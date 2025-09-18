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
    fn from((a, b): (Float, Float)) -> Self {
        let (l, u) = if a < b { (a, b) } else { (b, a) };
        match (l.is_finite(), u.is_finite()) {
            (true, true) => Self::LowerAndUpperBound(l, u),
            (true, false) => Self::LowerBound(l),
            (false, true) => Self::UpperBound(u),
            (false, false) => Self::NoBound,
        }
    }
}

impl From<(Option<Float>, Option<Float>)> for Bound {
    fn from((lo, hi): (Option<Float>, Option<Float>)) -> Self {
        match (lo, hi) {
            (Some(a), Some(b)) => {
                let (l, u) = if a < b { (a, b) } else { (b, a) };
                Self::LowerAndUpperBound(l, u)
            }
            (Some(l), None) => Self::LowerBound(l),
            (None, Some(u)) => Self::UpperBound(u),
            (None, None) => Self::NoBound,
        }
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
    /// Converts an unbounded "internal" parameter into a bounded "external" one via the transform:
    ///
    /// Upper and lower bounds:
    /// ```math
    /// x_\text{ext} = c + w \frac{x_\text{int}}{\sqrt{x_\text{int}^2 + 1}}
    /// ```
    /// where
    /// ```math
    /// c = \frac{x_\text{min} + x_\text{max}}{2},\ w = \frac{x_\text{max} - x_\text{min}}{2}
    /// ```
    /// Upper bound only:
    /// ```math
    /// x_\text{ext} = x_\text{max} - (\sqrt{x_\text{int}^2 + 1} - x_\text{int})
    /// ```
    /// Lower bound only:
    /// ```math
    /// x_\text{ext} = x_\text{min} + (\sqrt{x_\text{int}^2 + 1} + x_\text{int})
    /// ```
    pub fn to_bounded(&self, z: Float) -> Float {
        match *self {
            Self::LowerBound(lb) => lb + (Float::sqrt(z.mul_add(z, 1.0)) + z),
            Self::UpperBound(ub) => ub - (Float::sqrt(z.mul_add(z, 1.0)) - z),
            Self::LowerAndUpperBound(lb, ub) => {
                let c = 0.5 * (lb + ub);
                let w = 0.5 * (ub - lb);
                c + w * z / Float::sqrt(z.mul_add(z, 1.0))
            }
            Self::NoBound => z,
        }
    }
    /// Converts a bounded "external" parameter into an unbounded "internal" one via the transform:
    ///
    /// Upper and lower bounds:
    /// ```math
    /// x_\text{int} = \frac{u}{\sqrt{1 - u^2}}
    /// ```
    /// where
    /// ```math
    /// u = \frac{x_\text{ext} - c}{w},\ c = \frac{x_\text{min} + x_\text{max}}{2},\ w = \frac{x_\text{max} - x_\text{min}}{2}
    /// ```
    /// Upper bound only:
    /// ```math
    /// x_\text{int} = \frac{1}{2}\left(\frac{1}{(x_\text{max} - x_\text{ext})} - (x_\text{max} - x_\text{ext}) \right)
    /// ```
    /// Lower bound only:
    /// ```math
    /// x_\text{int} = \frac{1}{2}\left((x_\text{ext} - x_\text{min}) - \frac{1}{(x_\text{ext} - x_\text{min})} \right)
    /// ```
    pub fn to_unbounded(&self, x: Float) -> Float {
        match *self {
            Self::LowerBound(lb) => {
                let s = x - lb;
                0.5 * (s - 1.0 / s)
            }
            Self::UpperBound(ub) => {
                let s = ub - x;
                0.5 * (1.0 / s - s)
            }
            Self::LowerAndUpperBound(lb, ub) => {
                let c = 0.5 * (lb + ub);
                let w = 0.5 * (ub - lb);
                let mut u = (x - c) / w;
                if u >= 1.0 {
                    u = 1.0 - Float::EPSILON;
                }
                if u <= -1.0 {
                    u = -1.0 + Float::EPSILON;
                }
                u / Float::sqrt(u.mul_add(-u, 1.0))
            }
            Self::NoBound => x,
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
    B: Into<Bound> + Copy,
{
    fn from(value: &[B]) -> Self {
        Self(value.iter().copied().map(Into::into).collect())
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
impl<const N: usize, B> From<&[B; N]> for Bounds
where
    B: Into<Bound> + Copy,
{
    fn from(value: &[B; N]) -> Self {
        Self(value.as_slice().iter().copied().map(Into::into).collect())
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

    fn to_external_jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        let mut jac = DMatrix::zeros(self.0.len(), self.0.len());
        for (i, zi) in z.iter().enumerate() {
            jac[(i, i)] = match self.0[i] {
                Bound::NoBound => 1.0,
                Bound::LowerBound(_) => 1.0 + zi / Float::sqrt(zi.powi(2) + 1.0),
                Bound::UpperBound(_) => 1.0 - zi / Float::sqrt(zi.powi(2) + 1.0),
                Bound::LowerAndUpperBound(lb, ub) => {
                    let w = 0.5 * (ub - lb);
                    w / Float::powf(1.0 + zi * zi, 1.5)
                }
            }
        }
        jac
    }

    fn to_external_component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        let mut hess = DMatrix::zeros(self.0.len(), self.0.len());
        let za = z[a];
        hess[(a, a)] = match self.0[a] {
            Bound::NoBound => 0.0,
            Bound::LowerBound(_) => 1.0 / Float::powf(za.mul_add(za, 1.0), 1.5),
            Bound::UpperBound(_) => -1.0 / Float::powf(za.mul_add(za, 1.0), 1.5),
            Bound::LowerAndUpperBound(lb, ub) => {
                let w = 0.5 * (ub - lb);
                -3.0 * w * za / Float::powf(za.mul_add(za, 1.0), 2.5)
            }
        };
        hess
    }
}

/// A [`Transform`] that maps internal Cartesian coordinates to external spherical coordinates.
///
/// By specifying the indices of a vector which represent $`x`$, $`y`$, and $`z`$ in the internal
/// algorithm space, this transformation can construct $`r`$, $`\theta`$, and $`\phi`$ (in the
/// respective index slots) in spherical coordinates. This assumes the "physics" convention where
/// $`\theta`$ is the polar angle and $`\phi`$ is the azimuthal angle.
#[derive(Clone, Copy, Debug)]
pub struct SphericalTransform {
    /// The index of the internal $`x`$ value and the external $`r`$ value.
    pub i_x: usize,
    /// The index of the internal $`y`$ value and the external $`\theta`$ value.
    pub i_y: usize,
    /// The index of the internal $`z`$ value and the external $`\phi`$ value.
    pub i_z: usize,
}
impl SphericalTransform {
    /// Construct a new [`SphericalTransform`] from the internal/external indices.
    ///
    /// # Panics
    ///
    /// This method will panic if any of the indices are the same.
    pub fn new(i_x: usize, i_y: usize, i_z: usize) -> Self {
        assert!(i_x != i_y && i_x != i_z && i_y != i_z);
        Self { i_x, i_y, i_z }
    }
    #[inline]
    fn xyz_from_internal(&self, z: &DVector<Float>) -> (Float, Float, Float) {
        (z[self.i_x], z[self.i_y], z[self.i_z])
    }
    #[inline]
    fn rtp_from_internal(&self, z: &DVector<Float>) -> (Float, Float, Float) {
        let (cx, cy, cz) = self.xyz_from_internal(z);
        let r = (cx.mul_add(cx, cy.mul_add(cy, cz * cz))).sqrt();
        let s = cx.hypot(cy);
        let theta = if r > 0.0 { s.atan2(cz) } else { 0.0 };
        let phi = cy.atan2(cx);
        (r, theta, phi)
    }
    // #[inline]
    // fn jacobian_block_from_internal(&self, z: &DVector<Float>) -> [[Float; 3]; 3] {
    //     let (cx, cy, cz) = self.xyz_from_internal(z);
    //     let r2 = cx.mul_add(cx, cy.mul_add(cy, cz * cz));
    //     let r = r2.sqrt();
    //     let s2 = cx.mul_add(cx, cy * cy);
    //     let s = s2.sqrt();
    //     if r <= Float::EPSILON {
    //         return [[0.0; 3]; 3];
    //     }
    //     let dr = [cx / r, cy / r, cz / r];
    //     let mut dt = [0.0; 3];
    //     if s > Float::EPSILON {
    //         dt[0] = cx * cz / r2 / s;
    //         dt[1] = cy * cz / r2 / s;
    //     }
    //     if r2 > 0.0 {
    //         dt[2] = -s / r2;
    //     }
    //     let mut dp = [0.0; 3];
    //     if s2 > Float::EPSILON {
    //         dp[0] = -cy / s2;
    //         dp[1] = cx / s2;
    //     }
    //     [dr, dt, dp]
    // }
    #[inline]
    fn jacobian_block_from_internal(&self, z: &DVector<Float>) -> [[Float; 3]; 3] {
        let (cx, cy, cz) = self.xyz_from_internal(z);
        let r2 = cx * cx + cy * cy + cz * cz;
        let r = r2.sqrt();
        let s2 = cx * cx + cy * cy;
        let s = s2.sqrt();
        if r <= Float::EPSILON {
            return [[0.0; 3]; 3];
        }
        let dr = [cx / r, cy / r, cz / r];
        let mut dt = [0.0; 3];
        if s > Float::EPSILON {
            dt[0] = cx * cz / r2 / s;
            dt[1] = cy * cz / r2 / s;
        }
        if r2 > 0.0 {
            dt[2] = -s / r2;
        }
        let mut dp = [0.0; 3];
        if s2 > Float::EPSILON {
            dp[0] = -cy / s2;
            dp[1] = cx / s2;
        }
        [dr, dt, dp]
    }
    fn hessian_block_from_internal_r(&self, z: &DVector<Float>) -> [Float; 6] {
        let (cx, cy, cz) = self.xyz_from_internal(z);
        let r2 = cx.mul_add(cx, cy.mul_add(cy, cz * cz));
        let r = r2.sqrt();
        let r3 = r * r * r;
        if r <= Float::EPSILON {
            return [0.0; 6];
        }
        let drdxdx = cy.mul_add(cy, cz * cz) / r3;
        let drdxdy = -(cx * cy) / r3;
        let drdxdz = -(cx * cz) / r3;
        let drdydy = cx.mul_add(cx, cz * cz) / r3;
        let drdydz = -(cy * cz) / r3;
        let drdzdz = cx.mul_add(cx, cy * cy) / r3;
        [drdxdx, drdxdy, drdxdz, drdydy, drdydz, drdzdz]
    }
    fn hessian_block_from_internal_t(&self, z: &DVector<Float>) -> [Float; 6] {
        let (cx, cy, cz) = self.xyz_from_internal(z);
        let r2 = cx.mul_add(cx, cy.mul_add(cy, cz * cz));
        let s = cx.hypot(cy);
        let s2 = s * s;
        let r4 = r2 * r2;
        let s3 = s * s * s;
        if r4 <= Float::EPSILON {
            return [0.0; 6];
        }
        let dtdxdx =
            (cz * Float::mul_add(
                2.0,
                -cx.powi(4),
                (cx * cx * cy).mul_add(-cy, (cy * cy * cz).mul_add(cz, cy.powi(4))),
            )) / (s3 * r4);
        let dtdxdy = (-cx * cy * cz * Float::mul_add(3.0, s2, cz * cz)) / (s3 * r4);
        let dtdxdz = (cx * cz.mul_add(-cz, s2)) / (s * r4);
        let dtdydy =
            (cz * Float::mul_add(
                2.0,
                -cy.powi(4),
                (cx * cx * cy).mul_add(-cy, (cx * cx * cz).mul_add(cz, cx.powi(4))),
            )) / (s3 * r4);
        let dtdydz = (cy * cz.mul_add(-cz, s2)) / (s * r4);
        let dtdzdz = (2.0 * s * cz) / r4;
        [dtdxdx, dtdxdy, dtdxdz, dtdydy, dtdydz, dtdzdz]
    }
    fn hessian_block_from_internal_p(&self, z: &DVector<Float>) -> [Float; 6] {
        let (cx, cy, cz) = self.xyz_from_internal(z);
        let r2 = cx.mul_add(cx, cy.mul_add(cy, cz * cz));
        let s2 = cx.mul_add(cx, cy * cy);
        let r4 = r2 * r2;
        let s4 = s2 * s2;
        if r4 <= Float::EPSILON {
            return [0.0; 6];
        }
        let dpdxdx = (2.0 * cx * cy) / s4;
        let dpdxdy = cy.mul_add(cy, -(cx * cx)) / s4;
        let dpdxdz = 0.0;
        let dpdydy = -(2.0 * cx * cy) / s4;
        let dpdydz = 0.0;
        let dpdzdz = 0.0;
        [dpdxdx, dpdxdy, dpdxdz, dpdydy, dpdydz, dpdzdz]
    }
}
impl Transform for SphericalTransform {
    fn to_external<'a>(&'a self, z: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        let (r, theta, phi) = self.rtp_from_internal(z);
        let mut out = z.clone();
        out[self.i_x] = r;
        out[self.i_y] = theta;
        out[self.i_z] = phi;
        Cow::Owned(out)
    }

    fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        let r = x[self.i_x];
        let theta = x[self.i_y];
        let phi = x[self.i_z];
        let (st, ct) = theta.sin_cos();
        let (sp, cp) = phi.sin_cos();
        let cx = r * st * cp;
        let cy = r * st * sp;
        let cz = r * ct;
        let mut out = x.clone();
        out[self.i_x] = cx;
        out[self.i_y] = cy;
        out[self.i_z] = cz;
        Cow::Owned(out)
    }

    fn to_external_jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        let jac = self.jacobian_block_from_internal(z);
        let mut out = DMatrix::identity(z.len(), z.len());
        out[(self.i_x, self.i_x)] = jac[0][0];
        out[(self.i_x, self.i_y)] = jac[0][1];
        out[(self.i_x, self.i_z)] = jac[0][2];
        out[(self.i_y, self.i_x)] = jac[1][0];
        out[(self.i_y, self.i_y)] = jac[1][1];
        out[(self.i_y, self.i_z)] = jac[1][2];
        out[(self.i_z, self.i_x)] = jac[2][0];
        out[(self.i_z, self.i_y)] = jac[2][1];
        out[(self.i_z, self.i_z)] = jac[2][2];
        out
    }

    fn to_external_component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        let mut out = DMatrix::zeros(z.len(), z.len());
        let hess;
        if a == self.i_x {
            hess = self.hessian_block_from_internal_r(z);
        } else if a == self.i_y {
            hess = self.hessian_block_from_internal_t(z);
        } else if a == self.i_z {
            hess = self.hessian_block_from_internal_p(z);
        } else {
            return out;
        }
        out[(self.i_x, self.i_x)] = hess[0];
        out[(self.i_x, self.i_y)] = hess[1];
        out[(self.i_x, self.i_z)] = hess[2];
        out[(self.i_y, self.i_x)] = hess[1];
        out[(self.i_y, self.i_y)] = hess[3];
        out[(self.i_y, self.i_z)] = hess[4];
        out[(self.i_z, self.i_x)] = hess[2];
        out[(self.i_z, self.i_y)] = hess[4];
        out[(self.i_z, self.i_z)] = hess[5];
        out
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use approx::assert_relative_eq;
    use nalgebra::dvector;

    use super::*;
    use crate::{
        traits::{transform::TransformExt, Boundable, CostFunction, Gradient, TransformedProblem},
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

    /// A simple offset transform: adds +1.0 on `to_internal`, subtracts 1.0 on `to_external`.
    #[derive(Clone)]
    struct Offset;
    impl Transform for Offset {
        fn to_external<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x.add_scalar(-1.0))
        }
        fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x.add_scalar(1.0))
        }

        fn to_external_jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
            DMatrix::identity(z.len(), z.len())
        }

        fn to_external_component_hessian(&self, _a: usize, z: &DVector<Float>) -> DMatrix<Float> {
            DMatrix::zeros(z.len(), z.len())
        }
    }

    /// Identity transform for testing borrowed vs owned behavior.
    #[derive(Clone)]
    struct Identity;
    impl Transform for Identity {
        fn to_external<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Borrowed(x)
        }
        fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Borrowed(x)
        }
        fn to_external_jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
            DMatrix::identity(z.len(), z.len())
        }

        fn to_external_component_hessian(&self, _a: usize, z: &DVector<Float>) -> DMatrix<Float> {
            DMatrix::zeros(z.len(), z.len())
        }
    }

    /// A scaling transform: multiplies by `factor` on `external_to_internal`, divides on `internal_to_external`.
    #[derive(Clone)]
    struct Scale(Float);
    impl Transform for Scale {
        fn to_external<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x / self.0)
        }
        fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x * self.0)
        }
        fn to_external_jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
            DMatrix::identity(z.len(), z.len()).unscale(self.0)
        }

        fn to_external_component_hessian(&self, _a: usize, z: &DVector<Float>) -> DMatrix<Float> {
            DMatrix::zeros(z.len(), z.len())
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

    struct Quadratic;
    impl CostFunction for Quadratic {
        fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(x.map(|xi| xi * xi * 0.5).sum())
        }
    }
    impl Gradient for Quadratic {
        fn gradient(&self, x: &DVector<Float>, _args: &()) -> Result<DVector<Float>, Infallible> {
            Ok(x.clone())
        }

        fn hessian(&self, x: &DVector<Float>, _args: &()) -> Result<DMatrix<Float>, Infallible> {
            Ok(DMatrix::identity(x.len(), x.len()))
        }
    }

    #[test]
    fn test_bounds_transform() -> Result<(), Infallible> {
        let f = Quadratic;
        let t = Bounds::from([
            (-1.1, 2.0),
            (Float::NEG_INFINITY, 10.0),
            (0.2, Float::INFINITY),
            (Float::NEG_INFINITY, Float::INFINITY),
        ]);
        let p = TransformedProblem::new(&f, &t);
        let x_unbounded = dvector![1.0, -1.5, 0.8, 0.5];
        let x_bounded = t.to_owned_external(&x_unbounded);
        assert_relative_eq!(t.to_owned_internal(&x_bounded), x_unbounded);

        let f_int = p.evaluate(&x_unbounded, &())?;
        let f_ext = f.evaluate(&x_bounded, &())?;
        assert_relative_eq!(f_int, f_ext);

        let g_int = p.gradient(&x_unbounded, &())?;
        let g_ext = f.gradient(&x_bounded, &())?;
        let g_int_to_ext = p.pushforward_gradient(&x_unbounded, &g_int);
        assert_relative_eq!(g_int_to_ext, g_ext, epsilon = Float::EPSILON.sqrt());

        let h_int = p.hessian(&x_unbounded, &())?;
        let h_ext = f.hessian(&x_bounded, &())?;
        let h_int_to_ext = p.pushforward_hessian(&x_unbounded, &g_int, &h_int);
        assert_relative_eq!(h_int_to_ext, h_ext, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }

    #[test]
    fn test_spherical_transform() -> Result<(), Infallible> {
        let f = Quadratic;
        let t = SphericalTransform::new(0, 1, 2);
        let p = TransformedProblem::new(&f, &t);
        let xyz = dvector![1.0, -1.5, 0.8];
        let rtp = t.to_owned_external(&xyz);
        assert_relative_eq!(t.to_owned_internal(&rtp), xyz);

        let f_int = p.evaluate(&xyz, &())?;
        let f_ext = f.evaluate(&rtp, &())?;
        assert_relative_eq!(f_int, f_ext);

        let g_int = p.gradient(&xyz, &())?;
        let g_ext = f.gradient(&rtp, &())?;
        let g_int_to_ext = p.pushforward_gradient(&xyz, &g_int);
        assert_relative_eq!(g_int_to_ext, g_ext, epsilon = Float::EPSILON.sqrt());

        let h_int = p.hessian(&xyz, &())?;
        let h_ext = f.hessian(&rtp, &())?;
        let h_int_to_ext = p.pushforward_hessian(&xyz, &g_int, &h_int);
        assert_relative_eq!(h_int_to_ext, h_ext, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }
}
