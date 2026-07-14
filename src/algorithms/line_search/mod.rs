/// Backtracking line search.
pub mod backtracking_line_search;
pub use backtracking_line_search::BacktrackingLineSearch;

/// Hager-Zhang line search.
pub mod hager_zhang;
pub use hager_zhang::HagerZhangLineSearch;

/// More-Thuente line search.
pub mod more_thuente;
pub use more_thuente::MoreThuenteLineSearch;

/// Strong-Wolfe line search.
pub mod strong_wolfe;
pub use strong_wolfe::StrongWolfeLineSearch;
