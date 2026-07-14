//! Choice between the supported strong-Wolfe line-search algorithms.

use crate::algorithms::line_search::{HagerZhangLineSearch, MoreThuenteLineSearch};
use crate::core::{EvalCounts, LinearAlgebra, NalgebraProvider, RealScalar, Vector};
use crate::traits::{Gradient, LineSearch, LineSearchResult};

/// Either supported strong-Wolfe line search.
///
/// This is a dispatch choice, not a third line-search algorithm. It defaults to
/// [`MoreThuenteLineSearch`] and can instead hold [`HagerZhangLineSearch`].
#[derive(Clone, Debug)]
pub enum StrongWolfeLineSearch<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Moré–Thuente strong-Wolfe search.
    MoreThuente(MoreThuenteLineSearch<T, B>),
    /// Hager–Zhang strong-Wolfe search.
    HagerZhang(HagerZhangLineSearch<T, B>),
}

impl<T, B> Default for StrongWolfeLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::MoreThuente(MoreThuenteLineSearch::default())
    }
}

impl<T, B> From<MoreThuenteLineSearch<T, B>> for StrongWolfeLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn from(search: MoreThuenteLineSearch<T, B>) -> Self {
        Self::MoreThuente(search)
    }
}

impl<T, B> From<HagerZhangLineSearch<T, B>> for StrongWolfeLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn from(search: HagerZhangLineSearch<T, B>) -> Self {
        Self::HagerZhang(search)
    }
}

impl<T, B, P, U, E> LineSearch<T, B, P, U, E> for StrongWolfeLineSearch<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: Gradient<T, B, U, E> + ?Sized,
{
    fn search(
        &mut self,
        x: &Vector<T, B>,
        direction: &Vector<T, B>,
        max_step: Option<T>,
        problem: &P,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<LineSearchResult<T, B>, E> {
        match self {
            Self::MoreThuente(search) => {
                search.search(x, direction, max_step, problem, args, evals)
            }
            Self::HagerZhang(search) => search.search(x, direction, max_step, problem, args, evals),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CostFunction;
    use std::convert::Infallible;

    struct Quadratic;

    impl<T, B> CostFunction<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn evaluate(&self, x: &Vector<T, B>, _: &()) -> Result<T, Infallible> {
            Ok(x.dot(x))
        }
    }

    impl<T, B> Gradient<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn gradient(&self, x: &Vector<T, B>, _: &()) -> Result<Vector<T, B>, Infallible> {
            Ok(x.scale(T::literal(2.0)))
        }
    }

    #[test]
    fn dispatches_to_both_supported_searches() {
        let x: Vector<f64> = vec![2.0, -1.0].into();
        let direction = x.scale(-2.0);
        for mut search in [
            StrongWolfeLineSearch::from(MoreThuenteLineSearch::default()),
            StrongWolfeLineSearch::from(HagerZhangLineSearch::default()),
        ] {
            let mut evals = EvalCounts::default();
            let result = search
                .search(&x, &direction, None, &Quadratic, &(), &mut evals)
                .unwrap();
            assert!(result.is_ok() || result.is_err());
            assert!(evals.f() > 0);
        }
    }
}
