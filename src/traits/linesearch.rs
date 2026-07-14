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

/// Generic line-search contract.
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
    ) -> Result<LineSearchResult<T, B>, E>;
}
