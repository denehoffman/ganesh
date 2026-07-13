use crate::core::{EvalCounts, LinearAlgebra, NalgebraBackend, RealScalar, Vector};
use crate::traits::Gradient;
use crate::{
    core::Bounds,
    traits::{LegacyGradient, Status},
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
pub trait LineSearch<S: Status, U, E>: DynClone + Send + Sync {
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
    /// [`LegacyCostFunction::evaluate`](`crate::traits::LegacyCostFunction::evaluate`) for more
    /// information.
    #[allow(clippy::too_many_arguments)]
    fn search(
        &mut self,
        x: &DVector<Float>,
        p: &DVector<Float>,
        max_step: Option<Float>,
        problem: &dyn LegacyGradient<U, E>,
        bounds: Option<&Bounds>,
        args: &U,
        status: &mut S,
    ) -> Result<Result<LineSearchOutput, LineSearchOutput>, E>;
}
dyn_clone::clone_trait_object!(<S:Status, U, E> LineSearch<S, U, E>);

/// Output of a scalar- and backend-generic line search.
#[derive(Clone, Debug)]
pub struct BackendLineSearchOutput<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Accepted step size.
    pub alpha: T,
    /// Objective value at the accepted point.
    pub fx: T,
    /// Gradient at the accepted point.
    pub gradient: Vector<T, B>,
}

/// Successful or best-effort line-search output.
pub type BackendLineSearchResult<T, B> =
    Result<BackendLineSearchOutput<T, B>, BackendLineSearchOutput<T, B>>;

/// Backend-generic line-search contract.
pub trait BackendLineSearch<T, B, P, U = (), E = std::convert::Infallible>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: Gradient<T, B, U, E> + ?Sized,
{
    /// Search along `direction`, returning the best result even when line-search conditions fail.
    ///
    /// # Errors
    /// Returns an objective/gradient evaluation error.
    fn search(
        &mut self,
        x: &Vector<T, B>,
        direction: &Vector<T, B>,
        max_step: Option<T>,
        problem: &P,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<BackendLineSearchResult<T, B>, E>;
}
