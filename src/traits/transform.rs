use std::{borrow::Cow, ops::Deref};

use dyn_clone::DynClone;
use nalgebra::{DMatrix, DVector, LU};

use crate::{
    traits::{CostFunction, Gradient},
    Float,
};

pub trait Transform: DynClone {
    fn to_external<'a>(&'a self, z: &'a DVector<Float>) -> Cow<'a, DVector<Float>>;
    fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>>;

    #[inline]
    fn to_owned_external(&self, z: &DVector<Float>) -> DVector<Float> {
        self.to_external(z).into_owned()
    }
    #[inline]
    fn to_owned_internal(&self, x: &DVector<Float>) -> DVector<Float> {
        self.to_internal(x).into_owned()
    }
    fn jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        DMatrix::identity(z.len(), z.len()) // NOTE: default works iff linear
    }
    fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        DMatrix::zeros(z.len(), z.len()) // NOTE: default works iff linear
    }
}
dyn_clone::clone_trait_object!(Transform);

#[derive(Clone)]
pub struct Compose<T1, T2> {
    pub t1: T1,
    pub t2: T2,
}

pub trait TransformExt: Sized {
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
        todo!()
    }

    #[inline]
    fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        todo!()
        // let x1 = self.t1.to_external(z);
        // let j1 = self.t1.jacobian(z);
        // let j2 = self.t2.jacobian(x1.as_ref());
        // let mut h = j1.transpose() * self.t2.component_hessian(a, x1.as_ref()) * j1;
        // for b in 0..j2.ncols() {
        //     h += self.t1.component_hessian(b, z) * j2[(a, b)];
        // }
        // h
    }
}

pub trait DiffOps {
    fn pullback_gradient(&self, z: &DVector<Float>, g_ext: &DVector<Float>) -> DVector<Float>;
    fn pullback_hessian(
        &self,
        z: &DVector<Float>,
        g_ext: &DVector<Float>,
        h_ext: &DMatrix<Float>,
    ) -> DMatrix<Float>;

    fn pushforward_gradient(&self, z: &DVector<Float>, g_int: &DVector<Float>) -> DVector<Float>;
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
        let g_y = LU::new(jt.clone())
            .solve(g_int)
            .expect("J^T not invertible");
        let mut m = h_int.clone();
        for a in 0..g_y.len() {
            m -= self.component_hessian(a, z) * g_y[a];
        }
        let w = LU::new(jt.clone()).solve(&m).expect("J^T not invertible");
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
        match self {
            Some(t) => t.to_external(z),
            None => Cow::Borrowed(z),
        }
    }

    fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
        match self {
            Some(t) => t.to_internal(x),
            None => Cow::Borrowed(x),
        }
    }

    fn jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        match self {
            Some(t) => t.jacobian(z),
            None => DMatrix::identity(z.len(), z.len()),
        }
    }

    fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        match self {
            Some(t) => t.component_hessian(a, z),
            None => DMatrix::zeros(z.len(), z.len()),
        }
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

pub struct TransformedProblem<'a, F, T>
where
    T: Transform,
{
    pub f: &'a F,
    pub t: &'a T,
}
impl<'a, F, T> TransformedProblem<'a, F, T>
where
    T: Transform,
{
    pub fn new(f: &'a F, t: &'a T) -> Self {
        Self { f, t }
    }
    pub fn to_external<'b>(&'b self, z: &'b DVector<Float>) -> Cow<'b, DVector<Float>> {
        self.t.to_external(z)
    }

    pub fn to_internal<'b>(&'b self, x: &'b DVector<Float>) -> Cow<'b, DVector<Float>> {
        self.t.to_internal(x)
    }

    pub fn to_owned_external(&self, z: &DVector<Float>) -> DVector<Float> {
        self.to_external(z).into_owned()
    }

    pub fn to_owned_internal(&self, x: &DVector<Float>) -> DVector<Float> {
        self.to_internal(x).into_owned()
    }

    pub fn jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
        self.t.jacobian(z)
    }

    pub fn component_hessian(&self, a: usize, z: &DVector<Float>) -> DMatrix<Float> {
        self.t.component_hessian(a, z)
    }

    pub fn pullback_gradient(&self, z: &DVector<Float>, g_ext: &DVector<Float>) -> DVector<Float> {
        self.t.pullback_gradient(z, g_ext)
    }

    pub fn pullback_hessian(
        &self,
        z: &DVector<Float>,
        g_ext: &DVector<Float>,
        h_ext: &DMatrix<Float>,
    ) -> DMatrix<Float> {
        self.t.pullback_hessian(z, g_ext, h_ext)
    }

    pub fn pushforward_gradient(
        &self,
        z: &DVector<Float>,
        g_int: &DVector<Float>,
    ) -> DVector<Float> {
        self.t.pushforward_gradient(z, g_int)
    }

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
    F: CostFunction<U, E, Input = DVector<Float>>,
    T: Transform,
{
    type Input = DVector<Float>;
    #[inline]
    fn evaluate(&self, z: &Self::Input, args: &U) -> Result<Float, E> {
        self.f.evaluate(&self.t.to_external(z), args)
    }
}

impl<'a, F, U, E, T> Gradient<U, E> for TransformedProblem<'a, F, T>
where
    F: Gradient<U, E, Input = DVector<Float>>,
    T: Transform,
{
    #[inline]
    fn gradient(&self, z: &Self::Input, args: &U) -> Result<DVector<Float>, E> {
        let y = self.t.to_external(z);
        let gy = self.f.gradient(y.as_ref(), args)?;
        Ok(self.t.pullback_gradient(z, &gy))
    }
    #[inline]
    fn hessian(&self, z: &Self::Input, args: &U) -> Result<DMatrix<Float>, E> {
        let y = self.t.to_external(z);
        let gy = self.f.gradient(y.as_ref(), args)?;
        let hy = self.f.hessian(y.as_ref(), args)?;
        Ok(self.t.pullback_hessian(z, &gy, &hy))
    }
}
