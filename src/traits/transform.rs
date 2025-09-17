use std::{borrow::Cow, ops::Deref};

use dyn_clone::DynClone;
use nalgebra::{DMatrix, DVector, LU};

use crate::{
    traits::{CostFunction, Gradient},
    Float,
};

/// A trait used to define a change of basis.
///
/// This can be used to restrict an algorithm to a space of valid coordinates, such as a bounded
/// space or a space satisfying some constraints between parameters.
pub trait Transform: DynClone {
    /// Map from internal to external coordinates.
    fn to_external<'a>(&'a self, z: &'a DVector<Float>) -> Cow<'a, DVector<Float>>;
    /// Map from external to internal coordinates.
    fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>>;

    /// Map from internal to external coordinates, returning an owned vector.
    #[inline]
    fn to_owned_external(&self, z: &DVector<Float>) -> DVector<Float> {
        self.to_external(z).into_owned()
    }
    /// Map from external to internal coordinates, returning an owned vector.
    #[inline]
    fn to_owned_internal(&self, x: &DVector<Float>) -> DVector<Float> {
        self.to_internal(x).into_owned()
    }
    /// The Jacobian of the map from internal to external coordinates.
    fn to_external_jacobian(&self, z: &DVector<Float>) -> DMatrix<Float>;
    /// The Hessian of the map from internal to external coordinates for the `a`th coordinate.
    fn to_external_component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float>;

    /// The Jacobian of the map from external to internal coordinates.
    #[inline]
    fn to_internal_jacobian(&self, x: &DVector<Float>) -> DMatrix<Float> {
        let z = self.to_internal(x);
        let j = self.to_external_jacobian(&z);
        #[allow(clippy::expect_used)]
        LU::new(j).try_inverse().expect("J is not invertible")
    }

    /// The Hessian of the map from external to internal coordinates for the `b`th coordinate.
    #[inline]
    fn to_internal_component_hessian(&self, b: usize, x: &DVector<Float>) -> DMatrix<Float> {
        let z = self.to_internal(x);
        let j = self.to_external_jacobian(&z);
        #[allow(clippy::expect_used)]
        let k = LU::new(j).try_inverse().expect("J is not invertible");
        let n = z.len();
        let mut g = DMatrix::zeros(n, n);
        for a in 0..n {
            let h_a = self.to_external_component_hessian(a, &z);
            let s = &k.transpose() * h_a * &k;
            g -= s * k[(b, a)];
        }
        g
    }
}
dyn_clone::clone_trait_object!(Transform);

/// A struct represengint the composition of two transforms.
///
/// Specifically, this struct is designed such that `t1.compose(t2)` will
/// yield the mapping `t2.to_external(t1.to_external(z))`, so this composition starts with internal
/// coordinates, conducts the `t1` transform, and then conducts the `t2` transform.
#[derive(Clone)]
pub struct Compose<T1, T2> {
    /// The first transform in the composition.
    pub t1: T1,
    /// The second transform in the composition.
    pub t2: T2,
}

/// A helper trait to compose transforms.
pub trait TransformExt: Sized {
    /// Compose a transform with another.
    ///
    /// The convention used is such that `t1.compose(t2)` will
    /// yield the mapping `t2.to_external(t1.to_external(z))`.
    fn compose<T2>(self, t2: T2) -> Compose<Self, T2> {
        Compose { t1: self, t2 }
    }
}
impl<T> TransformExt for T {}
impl<T1, T2> Transform for Compose<T1, T2>
where
    T1: Transform + Clone,
    T2: Transform + Clone,
{
    #[inline]
    fn to_external<'a>(&'a self, z: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        match self.t1.to_external(z) {
            Cow::Borrowed(z1) => self.t2.to_external(z1),
            Cow::Owned(x1) => match self.t2.to_external(&x1) {
                Cow::Borrowed(_) => Cow::Owned(x1), // t2 is identity
                Cow::Owned(x2) => Cow::Owned(x2),
            },
        }
    }

    #[inline]
    fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        match self.t2.to_internal(x) {
            Cow::Borrowed(x1) => self.t1.to_internal(x1),
            Cow::Owned(z1) => match self.t1.to_internal(&z1) {
                Cow::Borrowed(_) => Cow::Owned(z1), // t1 is identity
                Cow::Owned(z2) => Cow::Owned(z2),
            },
        }
    }

    #[inline]
    fn to_external_jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        let u = self.t1.to_external(z);
        let j2 = self.t2.to_external_jacobian(u.as_ref());
        let j1 = self.t1.to_external_jacobian(z);
        j2 * j1
    }

    #[inline]
    fn to_external_component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        let x = self.t1.to_external(z);
        let j1 = self.t1.to_external_jacobian(z);
        let h2a = self.t2.to_external_component_hessian(a, &x);
        let mut h = j1.transpose() * h2a * j1;
        let j2 = self.t2.to_external_jacobian(&x);
        for b in 0..j2.ncols() {
            let h1b = self.t1.to_external_component_hessian(b, z);
            h += h1b.scale(j2[(a, b)]);
        }
        h
    }

    #[inline]
    fn to_internal_jacobian(&self, x: &DVector<Float>) -> DMatrix<Float> {
        let z = self.t2.to_internal(x);
        let k1 = self.t1.to_internal_jacobian(&z);
        let k2 = self.t2.to_internal_jacobian(x);
        k1 * k2
    }

    #[inline]
    fn to_internal_component_hessian(&self, b: usize, x: &DVector<Float>) -> DMatrix<Float> {
        let z = self.t2.to_internal(x);
        let k2 = self.t2.to_internal_jacobian(x);
        let g1b = self.t1.to_internal_component_hessian(b, &z);
        let mut g = k2.transpose() * g1b * k2;
        let k1 = self.t1.to_internal_jacobian(&z);
        for a in 0..k1.ncols() {
            let g2a = self.t2.to_internal_component_hessian(a, x);
            g += g2a.scale(k1[(b, a)]);
        }
        g
    }
}

/// Some useful differential operations mapping gradients and Hessians between internal and
/// external spaces.
pub trait DiffOps {
    /// The gradient on the internal space given an internal coordinate `z` and the external
    /// gradient `g_ext`.
    fn pullback_gradient(&self, z: &DVector<Float>, g_ext: &DVector<Float>) -> DVector<Float>;
    /// The Hessian on the internal space given an internal coordinate `z`, the external
    /// gradient `g_ext`, and the external Hessian `h_ext`.
    fn pullback_hessian(
        &self,
        z: &DVector<Float>,
        g_ext: &DVector<Float>,
        h_ext: &DMatrix<Float>,
    ) -> DMatrix<Float>;

    /// The gradient on the external space given an internal coordinate `z` and the internal
    /// gradient `g_int`.
    fn pushforward_gradient(&self, z: &DVector<Float>, g_int: &DVector<Float>) -> DVector<Float>;
    /// The Hessian on the external space given an internal coordinate `z`, the internal
    /// gradient `g_int`, and the internal Hessian `h_int`.
    fn pushforward_hessian(
        &self,
        z: &DVector<Float>,
        g_int: &DVector<Float>,
        h_int: &DMatrix<Float>,
    ) -> DMatrix<Float>;
}

impl<T> DiffOps for T
where
    T: Transform,
{
    #[inline]
    fn pullback_gradient(&self, z: &DVector<Float>, g_ext: &DVector<Float>) -> DVector<Float> {
        self.to_external_jacobian(z).transpose() * g_ext
    }
    #[inline]
    fn pullback_hessian(
        &self,
        z: &DVector<Float>,
        g_ext: &DVector<Float>,
        h_ext: &DMatrix<Float>,
    ) -> DMatrix<Float> {
        let j = self.to_external_jacobian(z);
        let mut h = j.transpose() * h_ext * j;
        for a in 0..g_ext.len() {
            h += self.to_external_component_hessian(a, z) * g_ext[a];
        }
        h
    }

    #[inline]
    fn pushforward_gradient(&self, z: &DVector<Float>, g_int: &DVector<Float>) -> DVector<Float> {
        let x = self.to_external(z);
        let j_inv = self.to_internal_jacobian(&x).transpose();
        j_inv.transpose() * g_int
    }
    #[inline]
    fn pushforward_hessian(
        &self,
        z: &DVector<Float>,
        g_int: &DVector<Float>,
        h_int: &DMatrix<Float>,
    ) -> DMatrix<Float> {
        let x = self.to_external(z);
        let j_inv = self.to_internal_jacobian(&x);
        let mut h = j_inv.transpose() * h_int * j_inv;
        for b in 0..g_int.len() {
            h += self.to_internal_component_hessian(b, &x) * g_int[b];
        }
        h
    }
}

impl<T> Transform for Option<T>
where
    T: Transform + Clone,
{
    fn to_external<'a>(&'a self, z: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        self.as_ref().map_or(Cow::Borrowed(z), |t| t.to_external(z))
    }

    fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        self.as_ref().map_or(Cow::Borrowed(x), |t| t.to_internal(x))
    }

    fn to_external_jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        self.as_ref().map_or_else(
            || DMatrix::identity(z.len(), z.len()),
            |t| t.to_external_jacobian(z),
        )
    }

    fn to_external_component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        self.as_ref().map_or_else(
            || DMatrix::zeros(z.len(), z.len()),
            |t| t.to_external_component_hessian(a, z),
        )
    }
}

impl Transform for Box<dyn Transform> {
    fn to_external<'a>(&'a self, z: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        self.deref().to_external(z)
    }

    fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        self.deref().to_internal(x)
    }

    fn to_external_jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        self.deref().to_external_jacobian(z)
    }

    fn to_external_component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        self.deref().to_external_component_hessian(a, z)
    }
}

/// A wrapper for a problem that has been transformed.
///
/// [`CostFunction`]s and [`Gradient`]s of this struct are intended to be evaluated on internal
/// coordinates, the [`Gradient::gradient`] and [`Gradient::hessian`] methods will both provide
/// internal versions of the gradient and Hessian. The external gradient and Hessian can be
/// obtained via the [`TransformedProblem::pushforward_gradient`] and [`TransformedProblem::pushforward_hessian`]
/// methods.
pub struct TransformedProblem<'a, F, T>
where
    T: Transform,
{
    /// The problem being transformed.
    pub f: &'a F,
    /// The transform to apply.
    pub t: &'a T,
}
impl<'a, F, T> TransformedProblem<'a, F, T>
where
    T: Transform,
{
    /// Create a new transformed problem from the given problem and transform.
    pub const fn new(f: &'a F, t: &'a T) -> Self {
        Self { f, t }
    }
    /// Use the stored transform to map from internal coordinates to external coordinates.
    pub fn to_external<'b>(&'b self, z: &'b DVector<Float>) -> Cow<'b, DVector<Float>> {
        self.t.to_external(z)
    }

    /// Use the stored transform to map from external coordinates to internal coordinates.
    pub fn to_internal<'b>(&'b self, x: &'b DVector<Float>) -> Cow<'b, DVector<Float>> {
        self.t.to_internal(x)
    }

    /// Used the stored transform to map from internal coordinates to external coordinates,
    /// producing an owned vector.
    pub fn to_owned_external(&self, z: &DVector<Float>) -> DVector<Float> {
        self.to_external(z).into_owned()
    }

    /// Use the stored transform to map from external coordinates to internal coordinates,
    /// producing an owned vector.
    pub fn to_owned_internal(&self, x: &DVector<Float>) -> DVector<Float> {
        self.to_internal(x).into_owned()
    }

    /// The Jacobian of the map from internal to external coordinates.
    pub fn jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        self.t.to_external_jacobian(z)
    }

    /// The Hessian of the map from internal to external coordinates for the `a`th coordinate.
    pub fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        self.t.to_external_component_hessian(a, z)
    }

    /// The gradient on the internal space given an internal coordinate `z` and the external
    /// gradient `g_ext`.
    pub fn pullback_gradient(&self, z: &DVector<Float>, g_ext: &DVector<Float>) -> DVector<Float> {
        self.t.pullback_gradient(z, g_ext)
    }

    /// The Hessian on the internal space given an internal coordinate `z`, the external
    /// gradient `g_ext`, and the external Hessian `h_ext`.
    pub fn pullback_hessian(
        &self,
        z: &DVector<Float>,
        g_ext: &DVector<Float>,
        h_ext: &DMatrix<Float>,
    ) -> DMatrix<Float> {
        self.t.pullback_hessian(z, g_ext, h_ext)
    }

    /// The gradient on the external space given an internal coordinate `z` and the internal
    /// gradient `g_int`.
    pub fn pushforward_gradient(
        &self,
        z: &DVector<Float>,
        g_int: &DVector<Float>,
    ) -> DVector<Float> {
        self.t.pushforward_gradient(z, g_int)
    }

    /// The Hessian on the external space given an internal coordinate `z`, the internal
    /// gradient `g_int`, and the internal Hessian `h_int`.
    pub fn pushforward_hessian(
        &self,
        z: &DVector<Float>,
        g_int: &DVector<Float>,
        h_int: &DMatrix<Float>,
    ) -> DMatrix<Float> {
        self.t.pushforward_hessian(z, g_int, h_int)
    }
}

impl<'a, F, U, E, T> CostFunction<U, E> for TransformedProblem<'a, F, T>
where
    F: CostFunction<U, E>,
    T: Transform,
{
    #[inline]
    fn evaluate(&self, z: &DVector<Float>, args: &U) -> Result<Float, E> {
        self.f.evaluate(&self.t.to_external(z), args)
    }
}

impl<'a, F, U, E, T> Gradient<U, E> for TransformedProblem<'a, F, T>
where
    F: Gradient<U, E>,
    T: Transform,
{
    #[inline]
    fn gradient(&self, z: &DVector<Float>, args: &U) -> Result<DVector<Float>, E> {
        let y = self.t.to_external(z);
        let gy = self.f.gradient(y.as_ref(), args)?;
        Ok(self.t.pullback_gradient(z, &gy))
    }
    #[inline]
    fn hessian(&self, z: &DVector<Float>, args: &U) -> Result<DMatrix<Float>, E> {
        let y = self.t.to_external(z);
        let gy = self.f.gradient(y.as_ref(), args)?;
        let hy = self.f.hessian(y.as_ref(), args)?;
        Ok(self.t.pullback_hessian(z, &gy, &hy))
    }
}
