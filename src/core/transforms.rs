use crate::{
    traits::{Bound, BoundLike, Transform},
    DMatrix, DVector, Float,
};
use fastrand::Rng;
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    ops::{Deref, DerefMut},
};

/// The default bounds transformation.
///
/// This will allow [`Bounds`] objects to transform a [`Bound`] according to the following
/// formulae:
///
/// Upper and lower bounds:
/// ```math
/// x_\text{ext} = c + w \frac{x_\text{int}}{\sqrt{x_\text{int}^2 + 1}}
/// ```
/// ```math
/// x_\text{int} = \frac{u}{\sqrt{1 - u^2}}
/// ```
/// where
/// ```math
/// u = \frac{x_\text{ext} - c}{w},\ c = \frac{x_\text{min} + x_\text{max}}{2},\ w = \frac{x_\text{max} - x_\text{min}}{2}
/// ```
/// Upper bound only:
/// ```math
/// x_\text{ext} = x_\text{max} - (\sqrt{x_\text{int}^2 + 1} - x_\text{int})
/// ```
/// ```math
/// x_\text{int} = \frac{1}{2}\left(\frac{1}{(x_\text{max} - x_\text{ext})} - (x_\text{max} - x_\text{ext}) \right)
/// ```
/// Lower bound only:
/// ```math
/// x_\text{ext} = x_\text{min} + (\sqrt{x_\text{int}^2 + 1} + x_\text{int})
/// ```
/// ```math
/// x_\text{int} = \frac{1}{2}\left((x_\text{ext} - x_\text{min}) - \frac{1}{(x_\text{ext} - x_\text{min})} \right)
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DefaultBoundsTransform;

#[typetag::serde]
impl BoundLike for DefaultBoundsTransform {
    fn to_internal_impl(&self, bound: Bound, x: Float) -> Float {
        match bound {
            Bound::NoBound => x,
            Bound::LowerBound(l) => {
                let s = x - l;
                0.5 * (s - 1.0 / s)
            }
            Bound::UpperBound(u) => {
                let s = u - x;
                0.5 * (1.0 / s - s)
            }
            Bound::LowerAndUpperBound(l, u) => {
                let c = 0.5 * (l + u);
                let w = 0.5 * (u - l);
                let mut u = (x - c) / w;
                if u >= 1.0 {
                    u = 1.0 - Float::EPSILON;
                }
                if u <= -1.0 {
                    u = -1.0 + Float::EPSILON;
                }
                u / u.mul_add(-u, 1.0).sqrt()
            }
        }
    }

    fn d_to_internal_impl(&self, bound: Bound, x: Float) -> Float {
        match bound {
            Bound::NoBound => 1.0,
            Bound::LowerBound(l) => (1.0 + 1.0 / (l - x).powi(2)) / 2.0,
            Bound::UpperBound(u) => (1.0 + 1.0 / (x - u).powi(2)) / 2.0,
            Bound::LowerAndUpperBound(l, u) => {
                (l - u).powi(2) / ((l - x) * (x - u)).powf(1.5) / 4.0
            }
        }
    }

    fn dd_to_internal_impl(&self, bound: Bound, x: Float) -> Float {
        match bound {
            Bound::NoBound => 0.0,
            Bound::LowerBound(l) => (l - x).powi(-3),
            Bound::UpperBound(u) => (u - x).powi(-3),
            Bound::LowerAndUpperBound(l, u) => {
                -3.0 * (l - u).powi(2) * (x.mul_add(-2.0, l) + u)
                    / 8.0
                    / ((l - x) * (x - u)).powf(2.5)
            }
        }
    }

    fn to_external_impl(&self, bound: Bound, z: Float) -> Float {
        match bound {
            Bound::NoBound => z,
            Bound::LowerBound(l) => l + (z.mul_add(z, 1.0).sqrt() + z),
            Bound::UpperBound(u) => u - (z.mul_add(z, 1.0).sqrt() - z),
            Bound::LowerAndUpperBound(l, u) => {
                let c = 0.5 * (l + u);
                let w = 0.5 * (u - l);
                c + w * z / (z.mul_add(z, 1.0)).sqrt()
            }
        }
    }

    fn d_to_external_impl(&self, bound: Bound, z: Float) -> Float {
        match bound {
            Bound::NoBound => 1.0,
            Bound::LowerBound(_) => 1.0 + z / (z.mul_add(z, 1.0)).sqrt(),
            Bound::UpperBound(_) => 1.0 - z / (z.mul_add(z, 1.0)).sqrt(),
            Bound::LowerAndUpperBound(l, u) => {
                let w = 0.5 * (u - l);
                w / z.mul_add(z, 1.0).powf(1.5)
            }
        }
    }

    fn dd_to_external_impl(&self, bound: Bound, z: Float) -> Float {
        match bound {
            Bound::NoBound => 0.0,
            Bound::LowerBound(_) => 1.0 / z.mul_add(z, 1.0).powf(1.5),
            Bound::UpperBound(_) => -1.0 / z.mul_add(z, 1.0).powf(1.5),
            Bound::LowerAndUpperBound(l, u) => {
                let w = 0.5 * (u - l);
                -3.0 * w * z / z.mul_add(z, 1.0).powf(2.5)
            }
        }
    }
}
/// A Minuit/LMFIT-style bounds transform.
///
/// This will allow [`Bounds`] objects to transform a [`Bound`] according to the following
/// formulae:
///
/// Upper and lower bounds:
/// ```math
/// x_\text{ext} = x_\text{min} + \left(\sin(x_\text{int}) + 1\right)\frac{x_\text{max} - x_\text{min}}{2}
/// ```
/// ```math
/// x_\text{int} = \arcsin\left(2\frac{x_\text{ext} - x_\text{min}}{x_\text{max} - x_\text{min}} - 1\right)
/// ```
///
/// Upper bound only:
/// ```math
/// x_\text{ext} = x_\text{max} + 1 - \sqrt{x_\text{int}^2 + 1}
/// ```
/// ```math
/// x_\text{int} = \sqrt{(x_\text{max} - x_\text{ext} + 1)^2 - 1}
/// ```
///
/// Lower bound only:
/// ```math
/// x_\text{ext} = x_\text{min} - 1 + \sqrt{x_\text{int}^2 + 1}
/// ```
/// ```math
/// x_\text{int} = \sqrt{(x_\text{ext} - x_\text{min} + 1)^2 - 1}
/// ```
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct MinuitBoundsTransform;

#[typetag::serde]
impl BoundLike for MinuitBoundsTransform {
    fn to_internal_impl(&self, bound: Bound, x: Float) -> Float {
        match bound {
            Bound::NoBound => x,
            Bound::LowerBound(l) => (x - l + 1.0).mul_add(x - l + 1.0, -1.0).sqrt(),
            Bound::UpperBound(u) => (u - x + 1.0).mul_add(u - x + 1.0, -1.0).sqrt(),
            Bound::LowerAndUpperBound(l, u) => (2.0 * (x - l) / (u - l) - 1.0).asin(),
        }
    }

    fn d_to_internal_impl(&self, bound: Bound, x: Float) -> Float {
        match bound {
            Bound::NoBound => 1.0,
            Bound::LowerBound(l) => (1.0 - l + x) / (1.0 - l + x).mul_add(1.0 - l + x, -1.0).sqrt(),
            Bound::UpperBound(u) => (x - u - 1.0) / (1.0 - x + u).mul_add(1.0 - x + u, -1.0).sqrt(),
            Bound::LowerAndUpperBound(l, u) => ((l - x) * (x - u)).powf(-0.5),
        }
    }

    fn dd_to_internal_impl(&self, bound: Bound, x: Float) -> Float {
        match bound {
            Bound::NoBound => 0.0,
            Bound::LowerBound(l) => -(1.0 - l + x).mul_add(1.0 - l + x, -1.0).powf(-1.5),
            Bound::UpperBound(u) => -(1.0 - x + u).mul_add(1.0 - x + u, -1.0).powf(-1.5),
            Bound::LowerAndUpperBound(l, u) => {
                x.mul_add(2.0, -u - l) / (2.0 * ((l - x) * (x - u)).powf(1.5))
            }
        }
    }

    fn to_external_impl(&self, bound: Bound, z: Float) -> Float {
        match bound {
            Bound::NoBound => z,
            Bound::LowerBound(l) => l - 1.0 + z.mul_add(z, 1.0).sqrt(),
            Bound::UpperBound(u) => u + 1.0 - z.mul_add(z, 1.0).sqrt(),
            Bound::LowerAndUpperBound(l, u) => l + (z.sin() + 1.0) * (u - l) / 2.0,
        }
    }

    fn d_to_external_impl(&self, bound: Bound, z: Float) -> Float {
        match bound {
            Bound::NoBound => 1.0,
            Bound::LowerBound(_) => z / z.mul_add(z, 1.0).sqrt(),
            Bound::UpperBound(_) => -z / z.mul_add(z, 1.0).sqrt(),
            Bound::LowerAndUpperBound(l, u) => (u - l) * z.cos() / 2.0,
        }
    }

    fn dd_to_external_impl(&self, bound: Bound, z: Float) -> Float {
        match bound {
            Bound::NoBound => 0.0,
            Bound::LowerBound(_) => z.mul_add(z, 1.0).powf(-1.5),
            Bound::UpperBound(_) => -z.mul_add(z, 1.0).powf(-1.5),
            Bound::LowerAndUpperBound(l, u) => (l - u) * z.sin() / 2.0,
        }
    }
}

/// A struct that contains a list of [`Bound`]s.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bounds(Vec<(Bound, Box<dyn BoundLike>)>);

impl Bounds {
    /// Create a new [`Bounds`] object from a list of [`Bound`]s-compatible structs and a list of [`BoundLike`] transforms.
    pub fn new<B>(
        bounds: impl IntoIterator<Item = B>,
        transforms: &[&(dyn BoundLike + 'static)],
    ) -> Self
    where
        B: Into<Bound> + Copy,
    {
        Self(
            bounds
                .into_iter()
                .map(Into::into)
                .zip(transforms.iter().map(|t| dyn_clone::clone_box(*t)))
                .collect(),
        )
    }
    /// Create a new [`Bounds`] object from a list of [`Bound`]s-compatible structs with the [`DefaultBoundsTransform`] transform.
    pub fn new_default<B, I>(bounds: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: Into<Bound>,
    {
        let bvec: Vec<Bound> = bounds.into_iter().map(Into::into).collect();
        let n = bvec.len();
        let tvec =
            std::iter::repeat_with(|| Box::new(DefaultBoundsTransform) as Box<dyn BoundLike>)
                .take(n);

        Self(bvec.into_iter().zip(tvec).collect())
    }
    /// Generate a random vector in the bounds.
    ///
    /// This uses the maximum/minimum representable value as
    /// limits in cases where bounds are infinite.
    pub fn random(&self, rng: &mut Rng) -> DVector<Float> {
        DVector::from_iterator(self.0.len(), self.0.iter().map(|(b, _)| b.random(rng)))
    }
    /// Check to see if the given vector is contained in a box with the given bounds.
    pub fn contains(&self, vec: &DVector<Float>) -> bool {
        self.0
            .iter()
            .zip(vec.iter())
            .all(|((bi, _), vi)| bi.contains(*vi))
    }
    /// Get the signed excess from each bound.
    ///
    /// This will return negative values for coordinates which are less than a lower bound and
    /// positive values for coordinates which are greater than an upper bound. It will return zero
    /// for coordinates which are within the bounds.
    pub fn get_excess(&self, vec: &DVector<Float>) -> DVector<Float> {
        vec.map_with_location(|i, _, v| self.0[i].0.get_excess(v))
    }
    /// Get a new vector which is clipped to be within the bounds.
    pub fn clip_values(&self, values: &DVector<Float>) -> DVector<Float> {
        values.map_with_location(|i, _, v| self.0[i].0.clip_value(v))
    }
    /// Applies a coordinate transform to the bounds.
    pub fn apply<T>(&self, transform: &T) -> Self
    where
        T: Transform,
    {
        let (l, u) = self
            .0
            .iter()
            .map(|(b, _)| b.as_floats())
            .collect::<(DVector<Float>, DVector<Float>)>();
        let l_int = transform.to_internal(&l);
        let u_int = transform.to_internal(&u);
        Self(
            l_int
                .iter()
                .copied()
                .zip(u_int.iter().copied())
                .zip(self.0.iter())
                .map(|(b, bt)| (Bound::from(b), bt.1.clone()))
                .collect(),
        )
    }
}

impl Deref for Bounds {
    type Target = Vec<(Bound, Box<dyn BoundLike>)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Bounds {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<B> From<Vec<B>> for Bounds
where
    B: Into<Bound> + Copy,
{
    fn from(value: Vec<B>) -> Self {
        Self::new_default(value)
    }
}
impl<B> From<&[B]> for Bounds
where
    B: Into<Bound> + Copy,
{
    fn from(value: &[B]) -> Self {
        Self::new_default(value.iter().copied())
    }
}
impl<const N: usize, B> From<[B; N]> for Bounds
where
    B: Into<Bound>,
{
    fn from(value: [B; N]) -> Self {
        Self::new_default(value)
    }
}
impl<const N: usize, B> From<&[B; N]> for Bounds
where
    B: Into<Bound> + Copy,
{
    fn from(value: &[B; N]) -> Self {
        Self::new_default(value.iter().copied())
    }
}

impl Transform for Bounds {
    fn to_external<'a>(&'a self, z: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        Cow::Owned(DVector::from_iterator(
            z.len(),
            self.iter()
                .zip(z.iter())
                .map(|((bi, ti), zi)| ti.to_external_impl(*bi, *zi)),
        ))
    }

    fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        Cow::Owned(DVector::from_iterator(
            x.len(),
            self.iter()
                .zip(x.iter())
                .map(|((bi, ti), xi)| ti.to_internal_impl(*bi, *xi)),
        ))
    }

    fn to_external_jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        DMatrix::from_diagonal(&DVector::from_iterator(
            z.len(),
            self.iter()
                .zip(z.iter())
                .map(|((bi, ti), zi)| ti.d_to_external_impl(*bi, *zi)),
        ))
    }

    fn to_external_component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        let mut h = DMatrix::zeros(z.len(), z.len());
        let za = z[a];
        let (ba, ta) = &self.0[a];
        h[(a, a)] = ta.dd_to_external_impl(*ba, za);
        h
    }

    fn to_internal_jacobian(&self, x: &DVector<Float>) -> DMatrix<Float> {
        DMatrix::from_diagonal(&DVector::from_iterator(
            x.len(),
            self.iter()
                .zip(x.iter())
                .map(|((bi, ti), xi)| ti.d_to_internal_impl(*bi, *xi)),
        ))
    }

    fn to_internal_component_hessian(&self, b: usize, x: &DVector<Float>) -> DMatrix<Float> {
        let mut g = DMatrix::zeros(x.len(), x.len());
        let xb = x[b];
        let (bb, tb) = &self.0[b];
        g[(b, b)] = tb.dd_to_internal_impl(*bb, xb);
        g
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
    #[inline]
    fn jacobian_block_from_internal(&self, z: &DVector<Float>) -> [[Float; 3]; 3] {
        let (cx, cy, cz) = self.xyz_from_internal(z);
        let r2 = cx.mul_add(cx, cy.mul_add(cy, cz * cz));
        let r = r2.sqrt();
        let s2 = cx.mul_add(cx, cy * cy);
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
        traits::{transform::TransformExt, CostFunction, Gradient, TransformedProblem},
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
        assert_eq!(b1.get_excess(-1.0), -1.0);

        let b2 = Bound::UpperBound(5.0);
        assert!(b2.contains(4.0));
        assert!(!b2.contains(6.0));
        assert_eq!(b2.get_excess(6.0), 1.0);

        let b3 = Bound::LowerAndUpperBound(-1.0, 1.0);
        assert!(b3.contains(0.0));
        assert!(!b3.contains(2.0));
    }

    #[test]
    fn test_bound_lower_upper_at_bound() {
        let b = Bound::LowerAndUpperBound(-2.0, 3.0);
        assert_eq!(b.lower(), -2.0);
        assert_eq!(b.upper(), 3.0);
        assert!(b.at_bound(-2.0, Float::EPSILON));
        assert!(b.at_bound(3.0, Float::EPSILON));
        assert!(!b.at_bound(0.0, Float::EPSILON));
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

        let d: DVector<Float> = bounds.random(&mut rng);

        assert_eq!(d.len(), bounds.len());
        assert!(bounds.contains(&d));
    }

    #[test]
    fn test_boundable_excess_constrain_unconstrain() {
        let bounds = sample_bounds();
        let d: DVector<Float> = dvector![-1.0, 11.0, 0.0, 5.0];

        let d_excess = bounds.get_excess(&d);
        assert_eq!(d_excess, dvector![-1.0, 1.0, 0.0, 0.0]);
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
    #[allow(clippy::suboptimal_flops)]
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

        match &scaled[0].0 {
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
    fn test_default_bounds_transform() -> Result<(), Infallible> {
        let f = Quadratic;
        let t = Bounds::from([
            (-1.1, 2.0),
            (Float::NEG_INFINITY, 10.0),
            (0.2, Float::INFINITY),
            (Float::NEG_INFINITY, Float::INFINITY),
        ]);
        let p = TransformedProblem::new(&f, &t);
        let x_ext = dvector![1.0, -1.5, 0.8, 0.5];
        let x_int = p.to_owned_internal(&x_ext);
        assert_relative_eq!(
            p.to_owned_external(&x_int),
            x_ext,
            epsilon = Float::EPSILON.sqrt()
        );

        let f_int = p.evaluate(&x_int, &())?;
        let f_ext = f.evaluate(&x_ext, &())?;
        assert_relative_eq!(f_int, f_ext, epsilon = Float::EPSILON.sqrt());

        let g_int = p.gradient(&x_int, &())?;
        let g_ext = f.gradient(&x_ext, &())?;
        let g_int_to_ext = p.pushforward_gradient(&x_int, &g_int);
        assert_relative_eq!(g_int_to_ext, g_ext, epsilon = Float::EPSILON.sqrt());

        let h_int = p.hessian(&x_int, &())?;
        let h_ext = f.hessian(&x_ext, &())?;
        let h_int_to_ext = p.pushforward_hessian(&x_int, &g_int, &h_int);
        assert_relative_eq!(h_int_to_ext, h_ext, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }

    #[test]
    fn test_minuit_bounds_transform() -> Result<(), Infallible> {
        let f = Quadratic;
        let t = Bounds::new(
            [
                (-1.1, 2.0),
                (Float::NEG_INFINITY, 10.0),
                (0.2, Float::INFINITY),
                (Float::NEG_INFINITY, Float::INFINITY),
            ],
            &[
                &MinuitBoundsTransform,
                &MinuitBoundsTransform,
                &MinuitBoundsTransform,
                &MinuitBoundsTransform,
            ],
        );
        let p = TransformedProblem::new(&f, &t);
        let x_ext = dvector![1.0, -1.5, 0.8, 0.5];
        let x_int = p.to_owned_internal(&x_ext);
        assert_relative_eq!(
            p.to_owned_external(&x_int),
            x_ext,
            epsilon = Float::EPSILON.sqrt()
        );

        let f_int = p.evaluate(&x_int, &())?;
        let f_ext = f.evaluate(&x_ext, &())?;
        assert_relative_eq!(f_int, f_ext, epsilon = Float::EPSILON.sqrt());

        let g_int = p.gradient(&x_int, &())?;
        let g_ext = f.gradient(&x_ext, &())?;
        let g_int_to_ext = p.pushforward_gradient(&x_int, &g_int);
        assert_relative_eq!(g_int_to_ext, g_ext, epsilon = Float::EPSILON.sqrt());

        let h_int = p.hessian(&x_int, &())?;
        let h_ext = f.hessian(&x_ext, &())?;
        let h_int_to_ext = p.pushforward_hessian(&x_int, &g_int, &h_int);
        assert_relative_eq!(h_int_to_ext, h_ext, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }

    #[test]
    fn test_spherical_transform() -> Result<(), Infallible> {
        let f = Quadratic;
        let t = SphericalTransform::new(0, 1, 2);
        let p = TransformedProblem::new(&f, &t);
        let x_ext = dvector![1.5, 0.2, 0.3];
        let x_int = p.to_owned_internal(&x_ext);
        assert_relative_eq!(
            p.to_owned_external(&x_int),
            x_ext,
            epsilon = Float::EPSILON.sqrt()
        );

        let f_int = p.evaluate(&x_int, &())?;
        let f_ext = f.evaluate(&x_ext, &())?;
        assert_relative_eq!(f_int, f_ext, epsilon = Float::EPSILON.sqrt());

        let g_int = p.gradient(&x_int, &())?;
        let g_ext = f.gradient(&x_ext, &())?;
        let g_int_to_ext = p.pushforward_gradient(&x_int, &g_int);
        assert_relative_eq!(g_int_to_ext, g_ext, epsilon = Float::EPSILON.sqrt());

        let h_int = p.hessian(&x_int, &())?;
        let h_ext = f.hessian(&x_ext, &())?;
        let h_int_to_ext = p.pushforward_hessian(&x_int, &g_int, &h_int);
        assert_relative_eq!(h_int_to_ext, h_ext, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }
}
