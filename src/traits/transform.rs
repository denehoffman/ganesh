use std::borrow::Cow;

use dyn_clone::DynClone;

/// This trait represents coordinate transforms between an internal and external parameter space.
///
/// Particularly, we think of external parameters as those the user wishes to access, while
/// internal parameters are often easier to work with inside algorithms. [`Transform`]s may also be
/// [`chain`](`Transform::chain`)ed together to form pipelines of transforms (which may not be
/// commutative).
pub trait Transform<I: Clone>: DynClone {
    /// Transform a set of external parameters to an equivalent set of internal parameters.
    fn to_internal<'a>(&'a self, x: &'a I) -> Cow<'a, I>;
    /// Transform a set of internal parameters to an equivalent set of external parameters.
    fn to_external<'a>(&'a self, x: &'a I) -> Cow<'a, I>;
    /// Transform a set of external parameters to an equivalent owned set of internal parameters.
    #[allow(clippy::wrong_self_convention)]
    fn into_internal<'a>(&'a self, x: &'a I) -> I {
        self.to_internal(x).into_owned()
    }
    /// Transform a set of internal parameters to an equivalent owned set of external parameters.
    #[allow(clippy::wrong_self_convention)]
    fn into_external<'a>(&'a self, x: &'a I) -> I {
        self.to_external(x).into_owned()
    }

    /// Combine this transform with another one, such that the resulting transform applies them in
    /// sequence ()
    fn chain<T>(&self, other: &T) -> TransformChain<I>
    where
        Self: Sized + 'static,
        T: Transform<I> + Sized + 'static,
    {
        TransformChain(dyn_clone::clone_box(self), dyn_clone::clone_box(other))
    }
}

dyn_clone::clone_trait_object!(<I> Transform<I>);

/// A chain of two [`Transform`]s.
///
/// When going from external to internal coordinates, the first transform is applied first. When
/// going from internal to external coordinates, the second transform is applied first.
#[derive(Clone)]
pub struct TransformChain<I: Clone>(Box<dyn Transform<I>>, Box<dyn Transform<I>>);

impl<I: Clone> Transform<I> for TransformChain<I> {
    fn to_internal<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        match self.0.to_internal(x) {
            Cow::Borrowed(b) => self.1.to_internal(b),
            Cow::Owned(o) => Cow::Owned(self.1.to_internal(&o).into_owned()),
        }
    }

    fn to_external<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        match self.1.to_external(x) {
            Cow::Borrowed(b) => self.0.to_external(b),
            Cow::Owned(o) => Cow::Owned(self.0.to_external(&o).into_owned()),
        }
    }
}
impl<I: Clone, T> Transform<I> for &T
where
    T: Transform<I>,
{
    fn to_internal<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        (*self).to_internal(x)
    }

    fn to_external<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        (*self).to_external(x)
    }
}

impl<I: Clone> Transform<I> for Box<dyn Transform<I>> {
    fn to_internal<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        self.as_ref().to_internal(x)
    }

    fn to_external<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        self.as_ref().to_external(x)
    }
}

impl<I, T> Transform<I> for Option<T>
where
    I: Clone,
    T: Transform<I> + Clone,
{
    fn to_internal<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        self.as_ref()
            .map_or_else(|| Cow::Borrowed(x), |t| t.to_internal(x))
    }

    fn to_external<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        self.as_ref()
            .map_or_else(|| Cow::Borrowed(x), |t| t.to_external(x))
    }
}
