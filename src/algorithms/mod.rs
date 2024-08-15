/// Module containing the Gradient Descent minimization algorithm
pub mod gradient_descent;
pub use gradient_descent::{GradientDescent, GradientDescentOptions};
/// Module containing the Newton-Raphson minimization algorithm
pub mod newton;
pub use newton::{Newton, NewtonOptions};
/// Module containing the Nelder-Mead minimization algorithm
pub mod nelder_mead;
pub use nelder_mead::{NelderMead, NelderMeadOptions};
/// Module containing Line Search algorithms
pub mod line_search;
