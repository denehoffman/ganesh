use crate::{
    core::Bounds,
    traits::{Gradient, Status},
    DVector, Float,
};
use dyn_clone::DynClone;

/// A struct containing the output of a line search in the direction $`\vec{p}`$.
pub struct LineSearchOutput {
    /// The step size $`\alpha`$ obtained from the line search
    pub alpha: Float,
    /// The value of the objective function at $`\vec{x} + \alpha \vec{p}`$
    pub fx: Float,
    /// The value of the gradient at $`\vec{x} + \alpha \vec{p}`$
    pub g: DVector<Float>,
}

/// A trait which defines the methods for a line search algorithm.
///
/// Line searches are one-dimensional minimizers typically used to determine optimal step sizes for
/// [`Algorithm`](`crate::traits::Algorithm`)s which only provide a direction for the next optimal step.
pub trait LineSearch<S: Status, U, E>: DynClone {
    /// The search method takes the current position of the minimizer, `x`, the search direction
    /// `p`, the objective function `func`, optional bounds `bounds`, and any arguments to the
    /// objective function `args`, and returns a [`Result`] containing another [`Result`]. The
    /// outer [`Result`] tells the caller if the line search encountered any errors in evaluating
    /// cost functions or gradients, while the inner [`Result`] indicates if the line search
    /// algorithm found a valid step. Even if the line search failed to find a valid step, it will still return the best [`LineSearchOutput`] found, as there are some cases where this is recoverable. For example, some line searches will hit the maximum number of iterations for an interval bisection if `x` is the true minimum, in which case a search between `x + eps` and `x + anything` will never improve upon `x` itself. Individual algorithms should determine how to handle these edge cases.
    ///
    /// # Notes
    ///
    /// Passing `bounds` usually implies a bounds transform is intented as most line search algorithms do not
    /// support bounded parameters by design.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See
    /// [`CostFunction::evaluate`](`crate::traits::CostFunction::evaluate`) for more
    /// information.
    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        x: &DVector<Float>,
        p: &DVector<Float>,
        max_step: Option<Float>,
        problem: &dyn Gradient<U, E>,
        bounds: Option<&Bounds>,
        args: &U,
        status: &mut S,
    ) -> Result<Result<LineSearchOutput, LineSearchOutput>, E>;
}
dyn_clone::clone_trait_object!(<S:Status, U, E> LineSearch<S, U, E>);
