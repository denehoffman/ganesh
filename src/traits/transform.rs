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
    fn jacobian(&self, z: &DVector<Float>) -> DMatrix<Float>;
    /// The Hessian of the map from internal to external coordinates for the `a`th coordinate.
    #[allow(unused_variables)]
    fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float>;
}
dyn_clone::clone_trait_object!(Transform);

/// A struct represengint the composition of two transforms.
///
/// Specifically, this struct is designed such that `t1.compose(t2)` will
/// yield the mapping `t2.to_external(t1.to_external(z))`.
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
    fn jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        let u = self.t1.to_external(z);
        let j2 = self.t2.jacobian(u.as_ref());
        let j1 = self.t1.jacobian(z);
        j2 * j1
    }

    #[inline]
    fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        let u = self.t1.to_external(z);
        let j1 = self.t1.jacobian(z);
        let j2 = self.t2.jacobian(u.as_ref());
        let h2a = self.t2.component_hessian(a, u.as_ref());
        let mut h = j1.transpose() * h2a * j1;
        let grad_ga = j2.row(a);
        for k in 0..grad_ga.ncols() {
            let coeff = grad_ga[k];
            let h1k = self.t1.component_hessian(k, z);
            h += h1k.scale(coeff);
        }
        h
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
        self.jacobian(z).transpose() * g_ext
    }
    #[inline]
    fn pullback_hessian(
        &self,
        z: &DVector<Float>,
        g_ext: &DVector<Float>,
        h_ext: &DMatrix<Float>,
    ) -> DMatrix<Float> {
        let j = self.jacobian(z);
        let mut h = j.transpose() * h_ext * j;
        for a in 0..g_ext.len() {
            h += self.component_hessian(a, z) * g_ext[a];
        }
        h
    }

    #[inline]
    fn pushforward_gradient(&self, z: &DVector<Float>, g_int: &DVector<Float>) -> DVector<Float> {
        let jt = self.jacobian(z).transpose();
        #[allow(clippy::expect_used)]
        LU::new(jt).solve(g_int).expect("J^T not invertible")
    }
    #[inline]
    fn pushforward_hessian(
        &self,
        z: &DVector<Float>,
        g_int: &DVector<Float>,
        h_int: &DMatrix<Float>,
    ) -> DMatrix<Float> {
        let jt = self.jacobian(z).transpose();
        #[allow(clippy::expect_used)]
        let g_y = LU::new(jt.clone())
            .solve(g_int)
            .expect("J^T not invertible");
        let mut m = h_int.clone();
        for a in 0..g_y.len() {
            m -= self.component_hessian(a, z) * g_y[a];
        }
        #[allow(clippy::expect_used)]
        let w = LU::new(jt.clone()).solve(&m).expect("J^T not invertible");
        #[allow(clippy::expect_used)]
        let h_t = LU::new(jt)
            .solve(&w.transpose())
            .expect("J^T not invertible");
        h_t.transpose()
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

    fn jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        self.as_ref()
            .map_or_else(|| DMatrix::identity(z.len(), z.len()), |t| t.jacobian(z))
    }

    fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        self.as_ref().map_or_else(
            || DMatrix::zeros(z.len(), z.len()),
            |t| t.component_hessian(a, z),
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

    fn jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        self.deref().jacobian(z)
    }

    fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        self.deref().component_hessian(a, z)
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
        self.t.jacobian(z)
    }

    /// The Hessian of the map from internal to external coordinates for the `a`th coordinate.
    pub fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        self.t.component_hessian(a, z)
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
