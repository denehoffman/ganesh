/// Module containing the Nelder-Mead minimization algorithm.
pub mod nelder_mead;
pub use nelder_mead::NelderMead;

/// Module containing various line-search methods.
pub mod line_search;

/// Module containing the BFGS method.
pub mod bfgs;
pub use bfgs::BFGS;

/// Module containing the L-BFGS method.
pub mod lbfgs;
pub use lbfgs::LBFGS;

/// Module containing the L-BFGS-B method.
pub mod lbfgsb;
pub use lbfgsb::LBFGSB;

/// Module containing the Particle Swarm Optimization method.
pub mod pso;
pub use pso::PSO;

use nalgebra::DVector;
use num::{
    traits::{float::TotalOrder, NumAssign},
    Float, FromPrimitive,
};
use std::{cmp::Ordering, fmt::Debug};

use crate::{Bound, Function};

/// Describes a point in parameter space that can be used in [`Algorithm`](`crate::Algorithm`)s.
#[derive(Eq, PartialEq, Clone, Default, Debug)]
pub struct Point<T>
where
    T: Clone + Debug + Float + 'static,
{
    x: DVector<T>,
    fx: T,
}
impl<T> Point<T>
where
    T: Clone + Debug + Float,
{
    fn len(&self) -> usize {
        self.x.len()
    }
    fn into_vec_val(self) -> (Vec<T>, T) {
        (self.x.data.into(), self.fx)
    }
}
impl<T> Point<T>
where
    T: Float + FromPrimitive + Debug + NumAssign + TotalOrder,
{
    fn evaluate<U, E>(&mut self, func: &dyn Function<T, U, E>, user_data: &mut U) -> Result<(), E> {
        self.fx = func.evaluate(self.x.as_slice(), user_data)?;
        Ok(())
    }
    fn evaluate_bounded<U, E>(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.fx = func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?;
        Ok(())
    }
    fn total_cmp(&self, other: &Self) -> Ordering {
        self.fx.total_cmp(&other.fx)
    }
}
impl<T> From<DVector<T>> for Point<T>
where
    T: Float + Debug,
{
    fn from(value: DVector<T>) -> Self {
        Self {
            x: value,
            fx: T::nan(),
        }
    }
}
impl<T> From<Vec<T>> for Point<T>
where
    T: Float + Debug + 'static,
{
    fn from(value: Vec<T>) -> Self {
        Self {
            x: DVector::from_vec(value),
            fx: T::nan(),
        }
    }
}
impl<'a, T> From<&'a Point<T>> for &'a Vec<T>
where
    T: Debug + Float,
{
    fn from(value: &'a Point<T>) -> Self {
        value.x.data.as_vec()
    }
}
impl<T> From<&[T]> for Point<T>
where
    T: Float + Debug + 'static,
{
    fn from(value: &[T]) -> Self {
        Self {
            x: DVector::from_column_slice(value),
            fx: T::nan(),
        }
    }
}
impl<'a, T> From<&'a Point<T>> for &'a [T]
where
    T: Debug + Float,
{
    fn from(value: &'a Point<T>) -> Self {
        value.x.data.as_slice()
    }
}
impl<T> PartialOrd for Point<T>
where
    T: PartialOrd + Debug + Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.fx.partial_cmp(&other.fx)
    }
}
