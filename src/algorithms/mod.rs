/// Module containing the Nelder-Mead minimization algorithm.
pub mod nelder_mead;
pub use nelder_mead::NelderMead;

/// Module containing various line-search methods.
pub mod line_search;

/// Module containing the BFGS method.
pub mod bfgs;
pub use bfgs::BFGS;

/// Module containing the L-BFGS method.
pub mod lbfgs;
pub use lbfgs::LBFGS;

/// Module containing the L-BFGS-B method.
pub mod lbfgsb;
pub use lbfgsb::LBFGSB;
