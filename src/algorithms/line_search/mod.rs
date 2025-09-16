use crate::{
    algorithms::gradient::GradientStatus,
    core::Bounds,
    traits::{linesearch::LineSearchOutput, Gradient, LineSearch},
    DVector, Float,
};

/// Implementation of the backtracking line search algorithm.
pub mod backtracking_line_search;
pub use backtracking_line_search::BacktrackingLineSearch;

/// Implementation of the Moré-Thuente line search algorithm.
pub mod more_thuente_line_search;
pub use more_thuente_line_search::MoreThuenteLineSearch;

/// Implementation of the Hager-Zhang line search algorithm.
pub mod hager_zhang_line_search;
pub use hager_zhang_line_search::HagerZhangLineSearch;

/// Line searches which obey strong Wolfe conditions.
#[derive(Clone)]
pub enum StrongWolfeLineSearch {
    /// The Moré-Thuente line search algorithm.
    MoreThuente(MoreThuenteLineSearch),
    /// The Hager-Zhang line search algorithm.
    HagerZhang(HagerZhangLineSearch),
}
impl Default for StrongWolfeLineSearch {
    fn default() -> Self {
        Self::MoreThuente(Default::default())
    }
}
impl<U, E> LineSearch<GradientStatus, U, E> for StrongWolfeLineSearch {
    fn search(
        &mut self,
        x: &DVector<Float>,
        p: &DVector<Float>,
        max_step: Option<Float>,
        problem: &dyn Gradient<U, E, Input = DVector<Float>>,
        bounds: Option<&Bounds>,
        args: &U,
        status: &mut GradientStatus,
    ) -> Result<Result<LineSearchOutput, LineSearchOutput>, E> {
        match self {
            Self::MoreThuente(more_thuente_line_search) => {
                more_thuente_line_search.search(x, p, max_step, problem, bounds, args, status)
            }
            Self::HagerZhang(hager_zhang_line_search) => {
                hager_zhang_line_search.search(x, p, max_step, problem, bounds, args, status)
            }
        }
    }
}
