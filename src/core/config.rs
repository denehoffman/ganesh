use serde::{Deserialize, Serialize};

use super::Bound;

const DEFAULT_MAX_STEPS: usize = 4000;

/// The configuration struct for the minimization problem. This struct contains basic information that
/// every [`Solver`](crate::traits::Solver) should need to run, such as the number of free parameters, the bounds for the parameters,
/// and the maximum number of steps to take before failing.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for Config {
    fn default() -> Self {
        Self {
            dimension: 0,
            bounds: None,
            parameter_names: None,
            max_steps: DEFAULT_MAX_STEPS,
        }
    }
}

impl Config {
    /// Sets all [`Bound`]s of the [`Config`] used by the [`Solver`](crate::traits::Solver). This can be [`None`] for an unbounded problem, or
    /// [`Some`] [`Vec<(T, T)>`] with length equal to the number of free parameters. Individual
    /// upper or lower bounds can be unbounded by setting them equal to `T::infinity()` or
    /// `T::neg_infinity()` (e.g. `f64::INFINITY` and `f64::NEG_INFINITY`).
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bounds is not equal to the number of free
    /// parameters.
    pub fn with_bounds<I: IntoIterator<Item = B>, B: Into<Bound>>(
        &mut self,
        bounds: I,
    ) -> &mut Self {
        let bounds = bounds.into_iter().map(Into::into).collect::<Vec<Bound>>();
        assert!(bounds.len() == self.dimension);
        self.bounds = Some(bounds);
        self
    }

    /// Sets the names of the parameters. This is only used for printing and debugging purposes.
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bounds is not equal to the number of free
    /// parameters.
    pub fn with_parameter_names<I: IntoIterator<Item = String>>(&mut self, names: I) -> &mut Self {
        let names = names.into_iter().collect::<Vec<String>>();
        assert!(names.len() == self.dimension);
        self.parameter_names = Some(names);
        self
    }
    /// Set the maximum number of steps to perform before failure (default: 4000).
    pub fn with_max_steps(&mut self, max_steps: usize) -> &mut Self {
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
