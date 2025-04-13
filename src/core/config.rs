use serde::{Deserialize, Serialize};

use super::Bound;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// The dimension of the minimization problem.
    pub dimension: usize,
    /// The bounds used for the minimization.
    pub bounds: Option<Vec<Bound>>,
    /// Optional parameter names
    pub parameter_names: Option<Vec<String>>,
    /// Max steps for the minimization
    pub max_steps: usize,
}

impl Config {
    /// Sets all [`Bound`]s of the [`Problem`]. This can be [`None`] for an unbounded problem, or
    /// [`Some`] [`Vec<(T, T)>`] with length equal to the number of free parameters. Individual
    /// upper or lower bounds can be unbounded by setting them equal to `T::infinity()` or
    /// `T::neg_infinity()` (e.g. `f64::INFINITY` and `f64::NEG_INFINITY`).
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bounds is not equal to the number of free
    /// parameters.
    pub fn with_bounds<I: IntoIterator<Item = B>, B: Into<Bound>>(mut self, bounds: I) -> Self {
        let bounds = bounds.into_iter().map(Into::into).collect::<Vec<Bound>>();
        assert!(bounds.len() == self.dimension);
        self.bounds = Some(bounds);
        self
    }

    pub fn with_parameter_names<I: IntoIterator<Item = String>>(mut self, names: I) -> Self {
        let names = names.into_iter().collect::<Vec<String>>();
        assert!(names.len() == self.dimension);
        self.parameter_names = Some(names);
        self
    }
    /// Set the maximum number of steps to perform before failure (default: 4000).
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Check parameters against the bounds
    pub fn assert_parameters(&self, x: &[f64]) {
        if let Some(bounds) = &self.bounds {
            for (i, (x_i, bound_i)) in x.iter().zip(bounds).enumerate() {
                assert!(
                    bound_i.contains(*x_i),
                    "Parameter #{} = {} is outside of the given bound: {}",
                    i,
                    x_i,
                    bound_i
                )
            }
        }
    }
}
