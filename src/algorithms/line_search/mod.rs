use crate::{
    algorithms::gradient::LegacyGradientStatus,
    core::Bounds,
    traits::{LegacyGradient, LegacyLineSearch, LegacyLineSearchOutput},
    DVector, Float,
};

/// Implementation of the backtracking line search algorithm.
pub mod backtracking_line_search;
#[doc(hidden)]
pub use backtracking_line_search::BackendBacktrackingLineSearch;
pub use backtracking_line_search::BackendBacktrackingLineSearch as BacktrackingLineSearch;
#[doc(hidden)]
pub use backtracking_line_search::BacktrackingLineSearch as LegacyBacktrackingLineSearch;

pub mod backend_strong_wolfe;
/// Implementation of the Moré-Thuente line search algorithm.
pub mod more_thuente_line_search;
pub use backend_strong_wolfe::BackendMoreThuenteLineSearch as MoreThuenteLineSearch;
pub use backend_strong_wolfe::{
    BackendHagerZhangLineSearch, BackendMoreThuenteLineSearch, BackendStrongWolfeLineSearch,
};
#[doc(hidden)]
pub use more_thuente_line_search::MoreThuenteLineSearch as LegacyMoreThuenteLineSearch;

/// Implementation of the Hager-Zhang line search algorithm.
pub mod hager_zhang_line_search;
pub use backend_strong_wolfe::BackendHagerZhangLineSearch as HagerZhangLineSearch;
#[doc(hidden)]
pub use hager_zhang_line_search::HagerZhangLineSearch as LegacyHagerZhangLineSearch;

/// Default generic strong-Wolfe line search.
pub type StrongWolfeLineSearch<T = f64, B = crate::NalgebraBackend> =
    BackendStrongWolfeLineSearch<T, B>;

/// Line searches which obey strong Wolfe conditions.
#[derive(Clone)]
pub enum LegacyStrongWolfeLineSearch {
    /// The Moré-Thuente line search algorithm.
    MoreThuente(LegacyMoreThuenteLineSearch),
    /// The Hager-Zhang line search algorithm.
    HagerZhang(LegacyHagerZhangLineSearch),
}
impl Default for LegacyStrongWolfeLineSearch {
    fn default() -> Self {
        Self::MoreThuente(Default::default())
    }
}
impl<U, E> LegacyLineSearch<LegacyGradientStatus, U, E> for LegacyStrongWolfeLineSearch {
    fn search(
        &mut self,
        x: &DVector<Float>,
        p: &DVector<Float>,
        max_step: Option<Float>,
        problem: &dyn LegacyGradient<U, E>,
        bounds: Option<&Bounds>,
        args: &U,
        status: &mut LegacyGradientStatus,
    ) -> Result<Result<LegacyLineSearchOutput, LegacyLineSearchOutput>, E> {
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
