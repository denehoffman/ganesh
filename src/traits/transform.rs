use std::borrow::Cow;

use dyn_clone::DynClone;

/// This trait represents coordinate transforms between an interior and exterior parameter space.
///
/// Particularly, we think of exterior parameters as those the user wishes to access, while
/// interior parameters are often easier to work with inside algorithms. [`Transform`]s may also be
/// [`chain`](`Transform::chain`)ed together to form pipelines of transforms (which may not be
/// commutative).
pub trait Transform<I: Clone>: DynClone {
    /// Transform a set of exterior parameters to an equivalent set of interior parameters.
    fn exterior_to_interior<'a>(&'a self, x: &'a I) -> Cow<'a, I>;
    /// Transform a set of interior parameters to an equivalent set of exterior parameters.
    fn interior_to_exterior<'a>(&'a self, x: &'a I) -> Cow<'a, I>;

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
/// When going from exterior to interior coordinates, the first transform is applied first. When
/// going from interior to exterior coordinates, the second transform is applied first.
#[derive(Clone)]
pub struct TransformChain<I: Clone>(Box<dyn Transform<I>>, Box<dyn Transform<I>>);

impl<I: Clone> Transform<I> for TransformChain<I> {
    fn exterior_to_interior<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        match self.0.exterior_to_interior(x) {
            Cow::Borrowed(b) => self.1.exterior_to_interior(b),
            Cow::Owned(o) => Cow::Owned(self.1.exterior_to_interior(&o).into_owned()),
        }
    }

    fn interior_to_exterior<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        match self.1.interior_to_exterior(x) {
            Cow::Borrowed(b) => self.0.interior_to_exterior(b),
            Cow::Owned(o) => Cow::Owned(self.0.interior_to_exterior(&o).into_owned()),
        }
    }
}
impl<I: Clone, T> Transform<I> for &T
where
    T: Transform<I>,
{
    fn exterior_to_interior<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        (*self).exterior_to_interior(x)
    }

    fn interior_to_exterior<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        (*self).interior_to_exterior(x)
    }
}

impl<I: Clone> Transform<I> for Box<dyn Transform<I>> {
    fn exterior_to_interior<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        self.as_ref().exterior_to_interior(x)
    }

    fn interior_to_exterior<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        self.as_ref().interior_to_exterior(x)
    }
}

impl<I, T> Transform<I> for Option<T>
where
    I: Clone,
    T: Transform<I> + Clone,
{
    fn exterior_to_interior<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        match self {
            Some(t) => t.exterior_to_interior(x),
            None => Cow::Borrowed(x),
        }
    }

    fn interior_to_exterior<'a>(&'a self, x: &'a I) -> Cow<'a, I> {
        match self {
            Some(t) => t.interior_to_exterior(x),
            None => Cow::Borrowed(x),
        }
    }
}
