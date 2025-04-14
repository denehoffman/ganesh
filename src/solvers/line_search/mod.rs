/// Implementation of the backtracking line search algorithm.
pub mod backtracking_line_search;
/// Implementation of the Strong Wolfe line search algorithm.
pub mod strong_wolfe_line_search;
pub use backtracking_line_search::BacktrackingLineSearch;
pub use strong_wolfe_line_search::StrongWolfeLineSearch;
