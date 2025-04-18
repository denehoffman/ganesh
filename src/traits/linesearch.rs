use dyn_clone::DynClone;
use nalgebra::DVector;

use crate::{traits::Status, Float};

use super::cost_function::CostFunction;

/// A trait which defines the methods for a line search algorithm.
///
/// Line searches are one-dimensional minimizers typically used to determine optimal step sizes for
/// [`Algorithm`](`crate::traits::Algorithm`)s which only provide a direction for the next optimal step.
pub trait LineSearch<S: Status, U, E>: DynClone {
    /// The search method takes the current position of the minimizer, `x`, the search direction
    /// `p`, the objective function `func`, optional bounds `bounds`, and any arguments to the
    /// objective function `user_data`, and returns a [`Result`] containing the tuple,
    /// `(valid, step_size, func(x + step_size * p), grad(x + step_size * p))`. Returns a [`None`]
    /// [`Result`] if the algorithm fails to find improvement.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`CostFunction::evaluate`] for more
    /// information.
    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        x: &DVector<Float>,
        p: &DVector<Float>,
        max_step: Option<Float>,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        status: &mut S,
    ) -> Result<(bool, Float, Float, DVector<Float>), E>;
}
dyn_clone::clone_trait_object!(<S:Status, U, E> LineSearch<S, U, E>);
