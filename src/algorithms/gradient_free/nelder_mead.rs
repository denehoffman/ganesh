//! Scalar- and linear-algebra-generic Nelder-Mead optimization.

use crate::algorithms::gradient_free::GradientFreeStatus;
use crate::core::{
    Callbacks, LinearAlgebra, Matrix, MinimizationSummary, NalgebraProvider, RealScalar, Vector,
};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{
    Algorithm, CheckpointableAlgorithm, CostFunction, Status, SupportsParameterNames, Terminator,
    Transform, TransformedProblem,
};
use std::{marker::PhantomData, ops::ControlFlow};

#[derive(Clone, Debug)]
struct Vertex<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    x: Vector<T, B>,
    fx: T,
}

impl<T: RealScalar, B: LinearAlgebra<T>> SupportsParameterNames for NelderMeadConfig<T, B> {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

/// Selects how an improving expansion step chooses its replacement vertex.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SimplexExpansionMethod {
    /// Keep the better of the reflected and expanded vertices.
    #[default]
    GreedyMinimization,
    /// Always keep the expanded vertex.
    GreedyExpansion,
}

/// Objective-value termination criteria for [`NelderMead`].
#[derive(Debug, Clone)]
pub enum NelderMeadFTerminator<T: RealScalar = f64> {
    /// Relative difference between the worst and best objective values.
    Amoeba {
        /// Relative tolerance.
        eps_rel: T,
    },
    /// Absolute difference between the worst and best objective values.
    Absolute {
        /// Absolute tolerance.
        eps_abs: T,
    },
    /// Standard deviation of all simplex objective values.
    StdDev {
        /// Absolute standard-deviation tolerance.
        eps_abs: T,
    },
}

impl<T: RealScalar> Default for NelderMeadFTerminator<T> {
    fn default() -> Self {
        Self::StdDev {
            eps_abs: T::epsilon().sqrt().sqrt(),
        }
    }
}

/// Position-based termination criteria for [`NelderMead`].
#[derive(Debug, Clone)]
pub enum NelderMeadXTerminator<T: RealScalar = f64> {
    /// Maximum infinity-norm distance from the best vertex.
    Diameter {
        /// Absolute diameter tolerance.
        eps_abs: T,
    },
    /// Maximum relative one-norm distance from the best vertex.
    Higham {
        /// Relative distance tolerance.
        eps_rel: T,
    },
    /// Best-to-worst distance relative to its initial value.
    Rowan {
        /// Relative distance tolerance.
        eps_rel: T,
    },
    /// Linearized simplex volume relative to its initial value.
    Singer {
        /// Relative linearized-volume tolerance.
        eps_rel: T,
    },
}

impl<T: RealScalar> Default for NelderMeadXTerminator<T> {
    fn default() -> Self {
        Self::Singer {
            eps_rel: T::epsilon().sqrt().sqrt(),
        }
    }
}

/// Configuration for linear-algebra-generic Nelder-Mead.
pub struct NelderMeadConfig<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Reflection coefficient.
    reflection: T,
    /// Expansion coefficient.
    expansion: T,
    /// Contraction coefficient.
    contraction: T,
    /// Shrink coefficient.
    shrink: T,
    /// Relative displacement for nonzero coordinates in the initial scaled-orthogonal simplex.
    initial_step: T,
    /// Absolute displacement for zero coordinates in the initial scaled-orthogonal simplex.
    initial_zero_step: T,
    /// Policy used when an expansion is attempted.
    expansion_method: SimplexExpansionMethod,
    /// Optional externally expressed custom simplex with exactly `dimension + 1` vertices.
    initial_simplex: Option<Vec<Vector<T, B>>>,
    /// Optional names for the optimized parameters.
    parameter_names: Option<Vec<String>>,
    /// Optional coordinate transform.
    transform: Option<Box<dyn Transform<T, B>>>,
}

impl<T, B> Default for NelderMeadConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            reflection: T::one(),
            expansion: T::literal(2.0),
            contraction: T::literal(0.5),
            shrink: T::literal(0.5),
            initial_step: T::literal(0.05),
            initial_zero_step: T::literal(0.00025),
            expansion_method: SimplexExpansionMethod::default(),
            initial_simplex: None,
            parameter_names: None,
            transform: None,
        }
    }
}

impl<T, B> NelderMeadConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Create a configuration with default hyperparameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reflection coefficient.
    pub fn with_alpha(mut self, value: T) -> GaneshResult<Self> {
        if value <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Reflection coefficient alpha must be greater than 0".to_string(),
            ));
        }
        self.reflection = value;
        Ok(self)
    }

    /// Set the expansion coefficient.
    pub fn with_beta(mut self, value: T) -> GaneshResult<Self> {
        if value <= T::one() {
            return Err(GaneshError::ConfigError(
                "Expansion coefficient beta must be greater than 1".to_string(),
            ));
        }
        if value <= self.reflection {
            return Err(GaneshError::ConfigError(format!(
                "Expansion coefficient beta must be greater than reflection coefficient alpha ({})",
                self.reflection
            )));
        }
        self.expansion = value;
        Ok(self)
    }

    /// Set the reflection and expansion coefficients together.
    pub fn with_alpha_beta(mut self, alpha: T, beta: T) -> GaneshResult<Self> {
        if alpha <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Reflection coefficient alpha must be greater than 0".to_string(),
            ));
        }
        if beta <= T::one() {
            return Err(GaneshError::ConfigError(
                "Expansion coefficient beta must be greater than 1".to_string(),
            ));
        }
        if beta <= alpha {
            return Err(GaneshError::ConfigError(
                "Expansion coefficient beta must be greater than reflection coefficient alpha"
                    .to_string(),
            ));
        }
        self.reflection = alpha;
        self.expansion = beta;
        Ok(self)
    }

    /// Set the contraction coefficient.
    pub fn with_gamma(mut self, value: T) -> GaneshResult<Self> {
        if value <= T::zero() || value >= T::one() {
            return Err(GaneshError::ConfigError(
                "Contraction coefficient gamma must be in (0, 1)".to_string(),
            ));
        }
        self.contraction = value;
        Ok(self)
    }

    /// Set the shrink coefficient.
    pub fn with_delta(mut self, value: T) -> GaneshResult<Self> {
        if value <= T::zero() || value >= T::one() {
            return Err(GaneshError::ConfigError(
                "Shrink coefficient delta must be in (0, 1)".to_string(),
            ));
        }
        self.shrink = value;
        Ok(self)
    }

    /// Set the relative displacement used for nonzero initial coordinates.
    pub fn with_initial_step(mut self, value: T) -> GaneshResult<Self> {
        if !value.is_finite() || value <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Initial simplex step must be finite and greater than 0".to_string(),
            ));
        }
        self.initial_step = value;
        Ok(self)
    }

    /// Set the absolute displacement used for zero initial coordinates.
    pub fn with_initial_zero_step(mut self, value: T) -> GaneshResult<Self> {
        if !value.is_finite() || value <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Initial zero-coordinate simplex step must be finite and greater than 0"
                    .to_string(),
            ));
        }
        self.initial_zero_step = value;
        Ok(self)
    }

    /// Use the Gao–Han adaptive coefficients for a problem of dimension `dimension`.
    ///
    /// # Errors
    /// Returns a configuration error when `dimension` is zero.
    pub fn with_adaptive(mut self, dimension: usize) -> GaneshResult<Self> {
        if dimension == 0 {
            return Err(GaneshError::ConfigError(
                "Adaptive hyperparameters requires input dimension >= 1".to_string(),
            ));
        }
        let dimension = T::literal(dimension as f64);
        self.reflection = T::one();
        self.expansion = T::one() + T::literal(2.0) / dimension;
        self.contraction = T::literal(0.75) - T::one() / (T::literal(2.0) * dimension);
        self.shrink = T::one() - T::one() / dimension;
        Ok(self)
    }

    /// Select the expansion replacement policy.
    pub const fn with_expansion_method(mut self, expansion_method: SimplexExpansionMethod) -> Self {
        self.expansion_method = expansion_method;
        self
    }

    /// Use a caller-supplied simplex expressed in external coordinates.
    ///
    /// # Errors
    /// Returns a configuration error unless the simplex contains exactly `dimension + 1`
    /// same-dimensional vertices.
    pub fn with_initial_simplex(mut self, simplex: Vec<Vector<T, B>>) -> GaneshResult<Self> {
        let Some(first) = simplex.first() else {
            return Err(GaneshError::ConfigError(
                "custom simplex must not be empty".to_string(),
            ));
        };
        let dimension = first.len();
        if simplex.len() != dimension + 1 {
            return Err(GaneshError::ConfigError(format!(
                "custom simplex requires exactly dimension + 1 vertices (received {} for dimension {dimension})",
                simplex.len()
            )));
        }
        if simplex.iter().any(|vertex| vertex.len() != dimension) {
            return Err(GaneshError::ConfigError(
                "custom simplex vertices must have the same dimension".to_string(),
            ));
        }
        self.initial_simplex = Some(simplex);
        Ok(self)
    }

    /// Configure a coordinate transform or bounds transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: Transform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

/// Scalar- and linear-algebra-generic Nelder-Mead optimizer.
#[derive(Clone, Debug)]
pub struct NelderMead<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    simplex: Vec<Vertex<T, B>>,
    coordinate_sum: Vector<T, B>,
    initial_best: Vector<T, B>,
    initial_worst: Vector<T, B>,
    relative_volume: T,
    _provider: PhantomData<B>,
}

/// Step-boundary checkpoint for [`NelderMead`].
#[derive(Clone, Debug)]
pub struct NelderMeadCheckpoint<T: RealScalar, B: LinearAlgebra<T>> {
    simplex: Vec<(Vector<T, B>, T)>,
    coordinate_sum: Vector<T, B>,
    initial_best: Vector<T, B>,
    initial_worst: Vector<T, B>,
    relative_volume: T,
    status: GradientFreeStatus<T, B>,
    next_step: usize,
}

impl<T, B> Default for NelderMead<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            simplex: Vec::new(),
            coordinate_sum: Vector::zeros(0),
            initial_best: Vector::zeros(0),
            initial_worst: Vector::zeros(0),
            relative_volume: T::one(),
            _provider: PhantomData,
        }
    }
}

impl<T, B> NelderMead<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn sort_simplex(&mut self) {
        self.simplex
            .sort_by(|left, right| left.fx.total_cmp(&right.fx));
    }

    fn centroid_without_worst(&self) -> Vector<T, B> {
        let dimension = self.simplex[0].x.len();
        let divisor = T::literal(dimension as f64);
        let worst = &self.simplex[self.simplex.len() - 1];
        Vector::from_vec(
            (0..dimension)
                .map(|coordinate| {
                    (self.coordinate_sum.get(coordinate) - worst.x.get(coordinate)) / divisor
                })
                .collect(),
        )
    }

    fn replace_vertex(&mut self, index: usize, replacement: Vertex<T, B>) {
        for coordinate in 0..replacement.x.len() {
            self.coordinate_sum.set(
                coordinate,
                self.coordinate_sum.get(coordinate) - self.simplex[index].x.get(coordinate)
                    + replacement.x.get(coordinate),
            );
        }
        self.simplex[index] = replacement;
    }

    fn replace_worst_sorted(&mut self, replacement: Vertex<T, B>) {
        let mut index = self.simplex.len() - 1;
        self.replace_vertex(index, replacement);
        while index > 0 && self.simplex[index].fx < self.simplex[index - 1].fx {
            self.simplex.swap(index, index - 1);
            index -= 1;
        }
    }

    fn evaluate<P, U, E>(
        problem: &TransformedProblem<'_, P, T, B>,
        x: Vector<T, B>,
        args: &U,
        status: &mut GradientFreeStatus<T, B>,
    ) -> Result<Vertex<T, B>, E>
    where
        P: CostFunction<T, B, U, E>,
    {
        let fx = problem.evaluate(&x, args)?;
        status.evals.record_f();
        Ok(Vertex { x, fx })
    }
}

impl<T, B, P, U, E>
    Terminator<NelderMead<T, B>, P, GradientFreeStatus<T, B>, U, E, NelderMeadConfig<T, B>>
    for NelderMeadFTerminator<T>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut NelderMead<T, B>,
        _problem: &P,
        status: &mut GradientFreeStatus<T, B>,
        _args: &U,
        _config: &NelderMeadConfig<T, B>,
    ) -> ControlFlow<()> {
        let best = &algorithm.simplex[0];
        let worst = &algorithm.simplex[algorithm.simplex.len() - 1];
        let message = match self {
            Self::Amoeba { eps_rel } => {
                let denominator = worst.fx.abs() + best.fx.abs();
                (T::literal(2.0) * (worst.fx - best.fx) / denominator <= *eps_rel)
                    .then_some("term_f = AMOEBA")
            }
            Self::Absolute { eps_abs } => {
                (worst.fx - best.fx <= *eps_abs).then_some("term_f = ABSOLUTE")
            }
            Self::StdDev { eps_abs } => {
                let count = T::literal(algorithm.simplex.len() as f64);
                let mean = algorithm
                    .simplex
                    .iter()
                    .fold(T::zero(), |sum, vertex| sum + vertex.fx)
                    / count;
                let variance = algorithm
                    .simplex
                    .iter()
                    .fold(T::zero(), |sum, vertex| sum + (vertex.fx - mean).powi(2))
                    / count;
                (variance.sqrt() <= *eps_abs).then_some("term_f = STDDEV")
            }
        };
        message.map_or(ControlFlow::Continue(()), |message| {
            status.set_message().succeed_with_message(message);
            ControlFlow::Break(())
        })
    }
}

impl<T, B, P, U, E>
    Terminator<NelderMead<T, B>, P, GradientFreeStatus<T, B>, U, E, NelderMeadConfig<T, B>>
    for NelderMeadXTerminator<T>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut NelderMead<T, B>,
        _problem: &P,
        status: &mut GradientFreeStatus<T, B>,
        _args: &U,
        _config: &NelderMeadConfig<T, B>,
    ) -> ControlFlow<()> {
        let best = &algorithm.simplex[0].x;
        let worst = &algorithm.simplex[algorithm.simplex.len() - 1].x;
        let max_distance = |one_norm: bool| {
            algorithm
                .simplex
                .iter()
                .skip(1)
                .fold(T::zero(), |largest, vertex| {
                    let distance = if one_norm {
                        (0..best.len()).fold(T::zero(), |sum, index| {
                            sum + (vertex.x.get(index) - best.get(index)).abs()
                        })
                    } else {
                        (0..best.len()).fold(T::zero(), |maximum, index| {
                            let value = (vertex.x.get(index) - best.get(index)).abs();
                            if value > maximum {
                                value
                            } else {
                                maximum
                            }
                        })
                    };
                    if distance > largest {
                        distance
                    } else {
                        largest
                    }
                })
        };
        let message = match self {
            Self::Diameter { eps_abs } => {
                (max_distance(false) <= *eps_abs).then_some("term_x = DIAMETER")
            }
            Self::Higham { eps_rel } => {
                let best_norm =
                    (0..best.len()).fold(T::zero(), |sum, index| sum + best.get(index).abs());
                let denominator = if best_norm > T::one() {
                    best_norm
                } else {
                    T::one()
                };
                (max_distance(true) / denominator <= *eps_rel).then_some("term_x = HIGHAM")
            }
            Self::Rowan { eps_rel } => {
                let initial = algorithm.initial_worst.sub(&algorithm.initial_best).norm();
                (worst.sub(best).norm() <= *eps_rel * initial).then_some("term_x = ROWAN")
            }
            Self::Singer { eps_rel } => (algorithm.relative_volume
                <= eps_rel.powi(best.len() as i32))
            .then_some("term_x = SINGER"),
        };
        message.map_or(ControlFlow::Continue(()), |message| {
            status.set_message().succeed_with_message(message);
            ControlFlow::Break(())
        })
    }
}

impl<T, B, P, U, E> Algorithm<P, GradientFreeStatus<T, B>, U, E> for NelderMead<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    type Summary = MinimizationSummary<T, B>;
    type Config = NelderMeadConfig<T, B>;
    type Init = Vector<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut GradientFreeStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        let origin = transformed.to_internal(init);
        self.simplex.clear();
        if let Some(simplex) = config.initial_simplex.as_ref() {
            for vertex in simplex {
                let vertex = transformed.to_internal(vertex);
                self.simplex
                    .push(Self::evaluate(&transformed, vertex, args, status)?);
            }
        } else {
            self.simplex
                .push(Self::evaluate(&transformed, origin.clone(), args, status)?);
            for coordinate in 0..origin.len() {
                let mut vertex = origin.clone();
                let value = origin.get(coordinate);
                vertex.set(
                    coordinate,
                    if value == T::zero() {
                        config.initial_zero_step
                    } else {
                        value * (T::one() + config.initial_step)
                    },
                );
                self.simplex
                    .push(Self::evaluate(&transformed, vertex, args, status)?);
            }
        }
        self.coordinate_sum = Vector::from_vec(
            (0..origin.len())
                .map(|coordinate| {
                    self.simplex
                        .iter()
                        .fold(T::zero(), |sum, vertex| sum + vertex.x.get(coordinate))
                })
                .collect(),
        );
        self.sort_simplex();
        self.initial_best = self.simplex[0].x.clone();
        self.initial_worst = self.simplex[self.simplex.len() - 1].x.clone();
        self.relative_volume = T::one();
        status.initialize(
            transformed.to_external(&self.simplex[0].x),
            self.simplex[0].fx,
        );
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut GradientFreeStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        let previous_best_fx = self.simplex[0].fx;
        let centroid = self.centroid_without_worst();
        let worst_index = self.simplex.len() - 1;
        let worst = self.simplex[worst_index].clone();
        let reflected_x = centroid.add_scaled(&centroid.sub(&worst.x), config.reflection);
        let reflected = Self::evaluate(&transformed, reflected_x, args, status)?;

        let step_message = if reflected.fx < self.simplex[0].fx {
            let expanded_x = centroid.add_scaled(&reflected.x.sub(&centroid), config.expansion);
            let expanded = Self::evaluate(&transformed, expanded_x, args, status)?;
            let replacement = match config.expansion_method {
                SimplexExpansionMethod::GreedyMinimization => {
                    if expanded.fx < reflected.fx {
                        expanded
                    } else {
                        reflected
                    }
                }
                SimplexExpansionMethod::GreedyExpansion => expanded,
            };
            self.replace_worst_sorted(replacement);
            self.relative_volume = self.relative_volume * config.reflection * config.expansion;
            "EXPAND"
        } else if reflected.fx < self.simplex[worst_index - 1].fx {
            self.replace_worst_sorted(reflected);
            self.relative_volume = self.relative_volume * config.reflection;
            "REFLECT"
        } else {
            let outside = reflected.fx < worst.fx;
            let (contraction_target, contraction_value) = if outside {
                (reflected.x, reflected.fx)
            } else {
                (worst.x, worst.fx)
            };
            let contracted_x =
                centroid.add_scaled(&contraction_target.sub(&centroid), config.contraction);
            let contracted = Self::evaluate(&transformed, contracted_x, args, status)?;
            if contracted.fx < contraction_value {
                self.replace_worst_sorted(contracted);
                let factor = if outside {
                    config.reflection * config.contraction
                } else {
                    config.contraction
                };
                self.relative_volume = self.relative_volume * factor;
                if outside {
                    "CONTRACT OUT"
                } else {
                    "CONTRACT IN"
                }
            } else {
                let best = self.simplex[0].clone();
                for index in 1..self.simplex.len() {
                    let shrunk = best
                        .x
                        .add_scaled(&self.simplex[index].x.sub(&best.x), config.shrink);
                    let replacement = Self::evaluate(&transformed, shrunk, args, status)?;
                    self.replace_vertex(index, replacement);
                }
                self.sort_simplex();
                self.relative_volume =
                    self.relative_volume * config.shrink.powi(best.x.len() as i32);
                "SHRINK"
            }
        };
        if self.simplex[0].fx < previous_best_fx {
            status.set_position(
                transformed.to_external(&self.simplex[0].x),
                self.simplex[0].fx,
            );
        } else {
            status.set_message().step();
        }
        status.set_message().step_with_message(step_message);
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &GradientFreeStatus<T, B>,
        _args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        let dimension = status.x.len();
        Ok(MinimizationSummary {
            bounds: config
                .transform
                .as_deref()
                .and_then(|transform| transform.parameter_bounds())
                .map(Vec::from),
            parameter_names: config.parameter_names.clone(),
            message: status.message.clone(),
            x0: init.clone(),
            x: status.x.clone(),
            std: crate::core::summary::unknown_uncertainties(dimension),
            fx: status.fx,
            evals: status.evals,
            covariance: Matrix::identity(dimension),
        })
    }

    fn reset(&mut self) {
        self.simplex.clear();
        self.coordinate_sum = Vector::zeros(0);
        self.initial_best = Vector::zeros(0);
        self.initial_worst = Vector::zeros(0);
        self.relative_volume = T::one();
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientFreeStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty()
            .with_terminator(NelderMeadFTerminator::default())
            .with_terminator(NelderMeadXTerminator::default())
    }
}

impl<T, B, P, U, E> CheckpointableAlgorithm<P, GradientFreeStatus<T, B>, U, E> for NelderMead<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    type Checkpoint = NelderMeadCheckpoint<T, B>;

    fn checkpoint(&self, status: &GradientFreeStatus<T, B>, next_step: usize) -> Self::Checkpoint {
        NelderMeadCheckpoint {
            simplex: self
                .simplex
                .iter()
                .map(|vertex| (vertex.x.clone(), vertex.fx))
                .collect(),
            coordinate_sum: self.coordinate_sum.clone(),
            initial_best: self.initial_best.clone(),
            initial_worst: self.initial_worst.clone(),
            relative_volume: self.relative_volume,
            status: status.clone(),
            next_step,
        }
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        _config: &Self::Config,
    ) -> (GradientFreeStatus<T, B>, usize) {
        self.simplex = checkpoint
            .simplex
            .iter()
            .map(|(x, fx)| Vertex {
                x: x.clone(),
                fx: *fx,
            })
            .collect();
        self.coordinate_sum = checkpoint.coordinate_sum.clone();
        self.initial_best = checkpoint.initial_best.clone();
        self.initial_worst = checkpoint.initial_worst.clone();
        self.relative_volume = checkpoint.relative_volume;
        (checkpoint.status.clone(), checkpoint.next_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::Rosenbrock;
    use crate::traits::Bounds;

    #[test]
    fn nelder_mead_runs_with_f32_and_bounds() {
        let bounds = Bounds::new([(-2.0_f32, 2.0), (-1.0, 3.0)]).unwrap();
        let config = NelderMeadConfig::<f32>::default().with_transform(bounds);
        let mut algorithm = NelderMead::<f32>::default();
        let result = algorithm
            .process(
                &Rosenbrock { n: 2 },
                &(),
                Vector::from_vec(vec![-1.2, 1.0]),
                config,
                NelderMead::<f32>::default_callbacks(),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 24.2);
        assert!(result.std.to_vec().iter().all(|value| value.is_nan()));
        assert_eq!(result.bounds.as_ref().map(Vec::len), Some(2));
        assert!(result.to_string().contains("Bounds"));
    }

    #[test]
    fn adaptive_coefficients_match_gao_han() {
        let config = NelderMeadConfig::<f64>::default().with_adaptive(2).unwrap();
        assert_eq!(config.reflection, 1.0);
        assert_eq!(config.expansion, 2.0);
        assert_eq!(config.contraction, 0.5);
        assert_eq!(config.shrink, 0.5);
    }

    #[test]
    fn coefficient_builders_report_the_specific_invalid_parameter() {
        let alpha = NelderMeadConfig::<f64>::default()
            .with_alpha(0.0)
            .err()
            .unwrap();
        assert!(alpha.to_string().contains("alpha"));

        let beta = NelderMeadConfig::<f64>::default()
            .with_beta(1.0)
            .err()
            .unwrap();
        assert!(beta.to_string().contains("beta"));

        let ordering = NelderMeadConfig::<f64>::default()
            .with_alpha_beta(2.0, 1.5)
            .err()
            .unwrap();
        assert!(ordering.to_string().contains("beta"));
        assert!(ordering.to_string().contains("alpha"));

        let gamma = NelderMeadConfig::<f64>::default()
            .with_gamma(1.0)
            .err()
            .unwrap();
        assert!(gamma.to_string().contains("gamma"));

        let delta = NelderMeadConfig::<f64>::default()
            .with_delta(0.0)
            .err()
            .unwrap();
        assert!(delta.to_string().contains("delta"));

        let adaptive = NelderMeadConfig::<f64>::default()
            .with_adaptive(0)
            .err()
            .unwrap();
        assert!(adaptive.to_string().contains("dimension"));
    }

    #[test]
    fn custom_simplex_and_expansion_policy_are_supported() {
        let names = vec!["x".to_string(), "y".to_string()];
        let config = NelderMeadConfig::<f64>::default()
            .with_expansion_method(SimplexExpansionMethod::GreedyExpansion)
            .with_initial_simplex(vec![
                Vector::from_vec(vec![-1.2, 1.0]),
                Vector::from_vec(vec![-1.1, 1.0]),
                Vector::from_vec(vec![-1.2, 1.1]),
            ])
            .unwrap();
        let config = NelderMeadConfig {
            parameter_names: Some(names.clone()),
            ..config
        };
        let problem = Rosenbrock { n: 2 };
        let init = Vector::from_vec(vec![-1.2, 1.0]);
        let mut algorithm: NelderMead = Default::default();
        let mut status = GradientFreeStatus::default();
        algorithm
            .initialize(&problem, &mut status, &(), &init, &config)
            .unwrap();
        algorithm
            .step(0, &problem, &mut status, &(), &config)
            .unwrap();
        let result = algorithm
            .summarize(1, &problem, &status, &(), &init, &config)
            .unwrap();
        assert!(result.evals.f() >= 4);
        assert_eq!(result.parameter_names, Some(names));
    }
}
