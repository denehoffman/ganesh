use nalgebra::DVector;
use serde::{Deserialize, Serialize};

use crate::Float;

use super::Bound;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// The dimension of the minimization problem.
    pub dimension: usize,
    /// The initial position of the minimization.
    pub x0: DVector<Float>,
    /// The bounds used for the minimization.
    pub bounds: Option<Vec<Bound>>,
    /// Optional parameter names
    pub parameter_names: Option<Vec<String>>,
    /// Max steps for the minimization
    pub max_steps: usize,
}
