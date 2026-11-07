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

// Retain a rejected trial's error so an exhausted search can return it intact.
fn reject_trial<V, E>(
    result: Result<V, E>,
    recoverable: impl FnOnce(&E) -> bool,
    rejected: &mut Option<E>,
) -> Result<Option<V>, E> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(error) if recoverable(&error) => {
            *rejected = Some(error);
            Ok(None)
        }
        Err(error) => Err(error),
    }
}
