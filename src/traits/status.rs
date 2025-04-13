use std::fmt::Display;

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::{core::Config, Float};

/// A trait which holds the status of a [`Solver`] and has to be implemented for own [`Solver`]s that need
/// different status information than the ones implemented in this crate.
pub trait Status: Display + Clone + Default + Serialize + for<'a> Deserialize<'a> {
    /// Resets the status to its default state. This is called at the beginning of every
    /// [`Solver`] run.
    /// Take care that this does not reset the [`Config`] struct or any other information that is
    /// needed to be kept between runs.
    fn reset(&mut self);
    /// Returns the [`Config`] struct of the [`Status`].
    fn config(&self) -> &Config;
    /// Returns a mutable reference to the [`Config`] struct of the [`Status`].
    fn config_mut(&mut self) -> &mut Config;
    /// Sets the [`Config`] struct of the [`Status`].
    fn with_config(self, config: Config) -> Self;
    /// Returns the current parameters of the minimization.
    fn x(&self) -> &DVector<Float>;
    /// Returns the current value of the minimization problem function at [`Status::x`].
    fn fx(&self) -> Float;
    /// Returns the convergence flag of the minimization.
    fn converged(&self) -> bool;
    /// Returns the message of the minimization.
    fn message(&self) -> &str;
    /// Sets the message of the minimization.
    fn update_message(&mut self, message: &str);
}
