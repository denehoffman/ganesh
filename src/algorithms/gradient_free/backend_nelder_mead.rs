//! Scalar- and backend-generic Nelder-Mead optimization.

use crate::algorithms::gradient_free::BackendGradientFreeStatus;
use crate::core::{
    BackendMinimizationSummary, Callbacks, LinearAlgebra, Matrix, MaxSteps, NalgebraBackend,
    RealScalar, Vector,
};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{Algorithm, BackendTransform, BackendTransformedProblem, CostFunction, Status};
use std::marker::PhantomData;

#[derive(Clone, Debug)]
struct Vertex<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    x: Vector<T, B>,
    fx: T,
}

/// Configuration for backend-generic Nelder-Mead.
pub struct BackendNelderMeadConfig<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Reflection coefficient.
    pub reflection: T,
    /// Expansion coefficient.
    pub expansion: T,
    /// Contraction coefficient.
    pub contraction: T,
    /// Shrink coefficient.
    pub shrink: T,
    /// Initial axis-aligned simplex displacement.
    pub initial_step: T,
    /// Objective spread convergence tolerance.
    pub f_tolerance: T,
    /// Simplex diameter convergence tolerance.
    pub x_tolerance: T,
    /// Optional coordinate transform.
    pub transform: Option<Box<dyn BackendTransform<T, B>>>,
}

impl<T, B> Default for BackendNelderMeadConfig<T, B>
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
            f_tolerance: T::epsilon().sqrt(),
            x_tolerance: T::epsilon().sqrt(),
            transform: None,
        }
    }
}

impl<T, B> BackendNelderMeadConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Configure a coordinate transform or bounds transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: BackendTransform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }

    /// Validate all Nelder-Mead coefficients.
    ///
    /// # Errors
    /// Returns a configuration error for non-finite or invalid coefficients.
    pub fn validate(&self) -> GaneshResult<()> {
        if !self.reflection.is_finite()
            || !self.expansion.is_finite()
            || !self.contraction.is_finite()
            || !self.shrink.is_finite()
            || self.reflection <= T::zero()
            || self.expansion <= T::one()
            || self.contraction <= T::zero()
            || self.contraction >= T::one()
            || self.shrink <= T::zero()
            || self.shrink >= T::one()
            || self.initial_step <= T::zero()
            || self.f_tolerance <= T::zero()
            || self.x_tolerance <= T::zero()
        {
            return Err(GaneshError::ConfigError(
                "invalid Nelder-Mead coefficients or tolerances".to_string(),
            ));
        }
        Ok(())
    }
}

/// Scalar- and backend-generic Nelder-Mead optimizer.
#[derive(Clone, Debug)]
pub struct BackendNelderMead<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    simplex: Vec<Vertex<T, B>>,
    _backend: PhantomData<B>,
}

impl<T, B> Default for BackendNelderMead<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            simplex: Vec::new(),
            _backend: PhantomData,
        }
    }
}

impl<T, B> BackendNelderMead<T, B>
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
        let mut centroid = Vector::zeros(dimension);
        for vertex in &self.simplex[..self.simplex.len() - 1] {
            centroid = centroid.add(&vertex.x);
        }
        centroid.scale(T::one() / T::literal((self.simplex.len() - 1) as f64))
    }

    fn converged(&self, config: &BackendNelderMeadConfig<T, B>) -> bool {
        let best = &self.simplex[0];
        let f_spread = self
            .simplex
            .iter()
            .map(|vertex| (vertex.fx - best.fx).abs())
            .fold(
                T::zero(),
                |largest, value| {
                    if value > largest {
                        value
                    } else {
                        largest
                    }
                },
            );
        let diameter = self
            .simplex
            .iter()
            .skip(1)
            .map(|vertex| vertex.x.sub(&best.x).norm())
            .fold(
                T::zero(),
                |largest, value| {
                    if value > largest {
                        value
                    } else {
                        largest
                    }
                },
            );
        f_spread <= config.f_tolerance && diameter <= config.x_tolerance
    }

    fn evaluate<P, U, E>(
        problem: &BackendTransformedProblem<'_, P, T, B>,
        x: Vector<T, B>,
        args: &U,
        status: &mut BackendGradientFreeStatus<T, B>,
    ) -> Result<Vertex<T, B>, E>
    where
        P: CostFunction<T, B, U, E>,
    {
        let fx = problem.evaluate(&x, args)?;
        status.evals.record_f();
        Ok(Vertex { x, fx })
    }
}

impl<T, B, P, U, E> Algorithm<P, BackendGradientFreeStatus<T, B>, U, E> for BackendNelderMead<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    type Summary = BackendMinimizationSummary<T, B>;
    type Config = BackendNelderMeadConfig<T, B>;
    type Init = Vector<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut BackendGradientFreeStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        debug_assert!(config.validate().is_ok());
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        let origin = transformed.to_internal(init);
        self.simplex.clear();
        self.simplex
            .push(Self::evaluate(&transformed, origin.clone(), args, status)?);
        for coordinate in 0..origin.len() {
            let mut vertex = origin.clone();
            let scale = T::one() + origin.get(coordinate).abs();
            vertex.set(
                coordinate,
                origin.get(coordinate) + config.initial_step * scale,
            );
            self.simplex
                .push(Self::evaluate(&transformed, vertex, args, status)?);
        }
        self.sort_simplex();
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
        status: &mut BackendGradientFreeStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        if self.converged(config) {
            status
                .set_message()
                .succeed_with_message("SIMPLEX CONVERGED");
            return Ok(());
        }
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        let centroid = self.centroid_without_worst();
        let worst_index = self.simplex.len() - 1;
        let worst = self.simplex[worst_index].clone();
        let reflected_x = centroid.add_scaled(&centroid.sub(&worst.x), config.reflection);
        let reflected = Self::evaluate(&transformed, reflected_x, args, status)?;

        if reflected.fx < self.simplex[0].fx {
            let expanded_x = centroid.add_scaled(&reflected.x.sub(&centroid), config.expansion);
            let expanded = Self::evaluate(&transformed, expanded_x, args, status)?;
            self.simplex[worst_index] = if expanded.fx < reflected.fx {
                expanded
            } else {
                reflected
            };
        } else if reflected.fx < self.simplex[worst_index - 1].fx {
            self.simplex[worst_index] = reflected;
        } else {
            let (contraction_target, contraction_value) = if reflected.fx < worst.fx {
                (reflected.x, reflected.fx)
            } else {
                (worst.x, worst.fx)
            };
            let contracted_x =
                centroid.add_scaled(&contraction_target.sub(&centroid), config.contraction);
            let contracted = Self::evaluate(&transformed, contracted_x, args, status)?;
            if contracted.fx < contraction_value {
                self.simplex[worst_index] = contracted;
            } else {
                let best = self.simplex[0].clone();
                for index in 1..self.simplex.len() {
                    let shrunk = best
                        .x
                        .add_scaled(&self.simplex[index].x.sub(&best.x), config.shrink);
                    self.simplex[index] = Self::evaluate(&transformed, shrunk, args, status)?;
                }
            }
        }
        self.sort_simplex();
        status.set_position(
            transformed.to_external(&self.simplex[0].x),
            self.simplex[0].fx,
        );
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &BackendGradientFreeStatus<T, B>,
        _args: &U,
        init: &Self::Init,
        _config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        let dimension = status.x.len();
        Ok(BackendMinimizationSummary {
            parameter_names: None,
            message: status.message.clone(),
            x0: init.clone(),
            x: status.x.clone(),
            std: Vector::zeros(dimension),
            fx: status.fx,
            evals: status.evals,
            covariance: Matrix::identity(dimension),
        })
    }

    fn reset(&mut self) {
        self.simplex.clear();
    }

    fn default_callbacks() -> Callbacks<Self, P, BackendGradientFreeStatus<T, B>, U, E, Self::Config>
    {
        Callbacks::empty().with_terminator(MaxSteps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::Rosenbrock;
    use crate::traits::BackendBounds;

    #[test]
    fn nelder_mead_runs_with_f32_and_bounds() {
        let bounds = BackendBounds::new([(-2.0_f32, 2.0), (-1.0, 3.0)]).unwrap();
        let config = BackendNelderMeadConfig {
            f_tolerance: 1e-5,
            x_tolerance: 1e-4,
            ..BackendNelderMeadConfig::default()
        }
        .with_transform(bounds);
        let mut algorithm = BackendNelderMead::<f32>::default();
        let result = algorithm
            .process(
                &Rosenbrock { n: 2 },
                &(),
                Vector::from_vec(vec![-1.2, 1.0]),
                config,
                BackendNelderMead::<f32>::default_callbacks(),
            )
            .unwrap();
        assert!(result.fx < 1e-3);
    }
}
