use crate::core::{EvalCounts, LinearAlgebra, NalgebraProvider, RealScalar, Vector};
use crate::traits::Gradient;

/// Output of a scalar- and linear-algebra-generic line search.
#[derive(Clone, Debug)]
pub struct LineSearchOutput<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Accepted step size.
    pub alpha: T,
    /// Objective value at the accepted point.
    pub fx: T,
    /// Gradient at the accepted point.
    pub gradient: Vector<T, B>,
}

/// Successful or best-effort line-search output.
pub type LineSearchResult<T, B> = Result<LineSearchOutput<T, B>, LineSearchOutput<T, B>>;

/// A line-search starting point with an already evaluated objective and gradient.
#[derive(Debug)]
pub struct LineSearchStart<'a, T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Starting coordinates.
    pub x: &'a Vector<T, B>,
    /// Objective value at `x`.
    pub fx: T,
    /// Gradient at `x`.
    pub gradient: &'a Vector<T, B>,
}

/// Generic line-search contract.
///
/// Implement [`Self::search_from`] to consume an already evaluated starting point.
/// The default [`Self::search`] evaluates the starting point before delegating to it.
pub trait LineSearch<T, B, P, U = (), E = std::convert::Infallible>
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
    ) -> Result<LineSearchResult<T, B>, E> {
        evals.record_fg();
        let (fx, gradient) = problem.evaluate_with_gradient(x, args)?;
        self.search_from(
            LineSearchStart {
                x,
                fx,
                gradient: &gradient,
            },
            direction,
            max_step,
            problem,
            args,
            evals,
        )
    }

    /// Search using an existing evaluation of the starting point.
    ///
    /// The supplied value and gradient must belong to `start.x` for the same problem and
    /// arguments. Reused evaluations are not added to `evals`. Implementations should use
    /// these supplied values rather than evaluating the starting point again.
    ///
    /// # Errors
    /// Returns an objective/gradient evaluation error.
    fn search_from(
        &mut self,
        start: LineSearchStart<'_, T, B>,
        direction: &Vector<T, B>,
        max_step: Option<T>,
        problem: &P,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<LineSearchResult<T, B>, E>;
}
