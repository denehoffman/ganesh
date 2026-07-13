//! Scalar- and backend-generic full-covariance CMA-ES.

use crate::algorithms::gradient_free::BackendGradientFreeStatus;
use crate::core::utils::sample_standard_normal;
use crate::core::{
    BackendMinimizationSummary, Callbacks, LinearAlgebra, Matrix, MaxSteps, NalgebraBackend,
    RandomScalar, SymmetricEigen, Vector,
};
use crate::traits::{Algorithm, BackendTransform, BackendTransformedProblem, CostFunction, Status};
use fastrand::Rng;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
struct Candidate<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    x: Vector<T, B>,
    y: Vector<T, B>,
    fx: T,
}

/// Configuration for backend-generic CMA-ES.
pub struct BackendCMAESConfig<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Population size; zero selects the standard logarithmic default.
    pub population_size: usize,
    /// Initial global step size.
    pub initial_sigma: T,
    /// Convergence threshold for the largest principal-axis step.
    pub tolerance: T,
    /// Optional coordinate transform.
    pub transform: Option<Box<dyn BackendTransform<T, B>>>,
}

impl<T, B> Default for BackendCMAESConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            population_size: 0,
            initial_sigma: T::literal(0.5),
            tolerance: T::epsilon().sqrt(),
            transform: None,
        }
    }
}

impl<T, B> BackendCMAESConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Configure a coordinate transform or smooth bounds.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: BackendTransform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

/// Scalar- and backend-generic covariance-matrix adaptation evolution strategy.
#[derive(Clone, Debug)]
pub struct BackendCMAES<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    rng: Rng,
    mean: Vector<T, B>,
    covariance: Matrix<T, B>,
    path_c: Vector<T, B>,
    path_sigma: Vector<T, B>,
    sigma: T,
    weights: Vec<T>,
    mu_eff: T,
    generation: usize,
    best_x: Vector<T, B>,
    best_fx: T,
    _backend: PhantomData<B>,
}

impl<T, B> BackendCMAES<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Construct with an optional deterministic seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::new, Rng::with_seed),
            mean: Vector::zeros(0),
            covariance: Matrix::zeros(0, 0),
            path_c: Vector::zeros(0),
            path_sigma: Vector::zeros(0),
            sigma: T::one(),
            weights: Vec::new(),
            mu_eff: T::one(),
            generation: 0,
            best_x: Vector::zeros(0),
            best_fx: T::infinity(),
            _backend: PhantomData,
        }
    }

    fn outer(left: &Vector<T, B>, right: &Vector<T, B>) -> Matrix<T, B> {
        let mut matrix = Matrix::zeros(left.len(), right.len());
        for row in 0..left.len() {
            for column in 0..right.len() {
                matrix.set(row, column, left.get(row) * right.get(column));
            }
        }
        matrix
    }

    fn parameters(&self, dimension: usize) -> (T, T, T, T, T, T) {
        let n = T::literal(dimension as f64);
        let one = T::one();
        let cc = (T::literal(4.0) + self.mu_eff / n)
            / (n + T::literal(4.0) + T::literal(2.0) * self.mu_eff / n);
        let cs = (self.mu_eff + T::literal(2.0)) / (n + self.mu_eff + T::literal(5.0));
        let c1 = T::literal(2.0) / ((n + T::literal(1.3)).powi(2) + self.mu_eff);
        let cmu_candidate = T::literal(2.0) * (self.mu_eff - T::literal(2.0) + one / self.mu_eff)
            / ((n + T::literal(2.0)).powi(2) + self.mu_eff);
        let cmu = if cmu_candidate < one - c1 {
            cmu_candidate
        } else {
            one - c1
        };
        let damping_extra = ((self.mu_eff - one) / (n + one)).sqrt() - one;
        let damping = one
            + T::literal(2.0)
                * if damping_extra > T::zero() {
                    damping_extra
                } else {
                    T::zero()
                }
            + cs;
        let chi = n.sqrt() * (one - one / (T::literal(4.0) * n) + one / (T::literal(21.0) * n * n));
        (cc, cs, c1, cmu, damping, chi)
    }
}

impl<T, B> Default for BackendCMAES<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::new(None)
    }
}

impl<T, B, P, U, E> Algorithm<P, BackendGradientFreeStatus<T, B>, U, E> for BackendCMAES<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T> + SymmetricEigen<T>,
    P: CostFunction<T, B, U, E>,
{
    type Summary = BackendMinimizationSummary<T, B>;
    type Config = BackendCMAESConfig<T, B>;
    type Init = Vector<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut BackendGradientFreeStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        self.mean = transformed.to_internal(init);
        let dimension = self.mean.len();
        self.covariance = Matrix::identity(dimension);
        self.path_c = Vector::zeros(dimension);
        self.path_sigma = Vector::zeros(dimension);
        self.sigma = config.initial_sigma;
        self.generation = 0;
        let lambda = if config.population_size == 0 {
            4 + (3.0 * (dimension as f64).ln()).floor() as usize
        } else {
            config.population_size.max(2)
        };
        let mu = lambda / 2;
        self.weights = (0..mu)
            .map(|index| T::literal((mu as f64 + 0.5).ln() - ((index + 1) as f64).ln()))
            .collect();
        let weight_sum = self
            .weights
            .iter()
            .copied()
            .fold(T::zero(), |sum, value| sum + value);
        for weight in &mut self.weights {
            *weight = *weight / weight_sum;
        }
        self.mu_eff = T::one()
            / self
                .weights
                .iter()
                .copied()
                .map(|weight| weight * weight)
                .fold(T::zero(), |sum, value| sum + value);
        self.best_x = self.mean.clone();
        self.best_fx = transformed.evaluate(&self.mean, args)?;
        status.evals.record_f();
        status.initialize(init.clone(), self.best_fx);
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
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        let dimension = self.mean.len();
        let Some((eigenvalues, eigenvectors)) = self.covariance.symmetric_eigen() else {
            status
                .set_message()
                .fail_with_message("COVARIANCE EIGENDECOMPOSITION FAILED");
            return Ok(());
        };
        let clamped: Vector<T, B> = Vector::from_vec(
            (0..dimension)
                .map(|index| {
                    let value = eigenvalues.get(index);
                    if value > T::epsilon() {
                        value
                    } else {
                        T::epsilon()
                    }
                })
                .collect(),
        );
        let lambda = if config.population_size == 0 {
            4 + (3.0 * (dimension as f64).ln()).floor() as usize
        } else {
            config.population_size.max(2)
        };
        let mut population = Vec::with_capacity(lambda);
        for _ in 0..lambda {
            let normal: Vector<T, B> = Vector::from_vec(
                (0..dimension)
                    .map(|_| sample_standard_normal(&mut self.rng))
                    .collect(),
            );
            let scaled: Vector<T, B> = Vector::from_vec(
                (0..dimension)
                    .map(|index| normal.get(index) * clamped.get(index).sqrt())
                    .collect(),
            );
            let y = eigenvectors.mul_vec(&scaled);
            let x = self.mean.add_scaled(&y, self.sigma);
            let fx = transformed.evaluate(&x, args)?;
            status.evals.record_f();
            population.push(Candidate { x, y, fx });
        }
        population.sort_by(|left, right| left.fx.total_cmp(&right.fx));
        if population[0].fx < self.best_fx {
            self.best_fx = population[0].fx;
            self.best_x = population[0].x.clone();
        }
        let old_mean = self.mean.clone();
        self.mean = Vector::zeros(dimension);
        for (candidate, weight) in population.iter().zip(&self.weights) {
            self.mean = self.mean.add_scaled(&candidate.x, *weight);
        }
        let y_weighted = self.mean.sub(&old_mean).scale(T::one() / self.sigma);
        let coordinates = eigenvectors.transpose().mul_vec(&y_weighted);
        let inverse_scaled = Vector::from_vec(
            (0..dimension)
                .map(|index| coordinates.get(index) / clamped.get(index).sqrt())
                .collect(),
        );
        let inverse_sqrt_y = eigenvectors.mul_vec(&inverse_scaled);
        let (cc, cs, c1, cmu, damping, chi) = self.parameters(dimension);
        let path_sigma_scale = (cs * (T::literal(2.0) - cs) * self.mu_eff).sqrt();
        self.path_sigma = self
            .path_sigma
            .scale(T::one() - cs)
            .add_scaled(&inverse_sqrt_y, path_sigma_scale);
        let decay_exponent =
            i32::try_from(self.generation.saturating_add(1).saturating_mul(2)).unwrap_or(i32::MAX);
        let decay = (T::one() - cs).powi(decay_exponent);
        let normalized_path = self.path_sigma.norm() / (T::one() - decay).sqrt() / chi;
        let h_sigma = normalized_path
            < T::literal(1.4) + T::literal(2.0) / (T::literal(dimension as f64) + T::one());
        self.path_c = self.path_c.scale(T::one() - cc);
        if h_sigma {
            self.path_c = self.path_c.add_scaled(
                &y_weighted,
                (cc * (T::literal(2.0) - cc) * self.mu_eff).sqrt(),
            );
        }
        let old_covariance = self.covariance.clone();
        let h_correction = if h_sigma {
            T::zero()
        } else {
            cc * (T::literal(2.0) - cc)
        };
        let covariance_decay = old_covariance.scale(T::one() - c1 - cmu + c1 * h_correction);
        let rank_one = Self::outer(&self.path_c, &self.path_c).scale(c1);
        self.covariance = &covariance_decay + &rank_one;
        for (candidate, weight) in population.iter().zip(&self.weights) {
            let contribution = Self::outer(&candidate.y, &candidate.y).scale(cmu * *weight);
            self.covariance = &self.covariance + &contribution;
        }
        self.sigma =
            self.sigma * ((cs / damping) * (self.path_sigma.norm() / chi - T::one())).exp();
        self.generation += 1;
        let max_axis = (0..dimension).map(|index| clamped.get(index).sqrt()).fold(
            T::zero(),
            |largest, value| {
                if value > largest {
                    value
                } else {
                    largest
                }
            },
        );
        status.set_position(transformed.to_external(&self.best_x), self.best_fx);
        if self.sigma * max_axis <= config.tolerance {
            status
                .set_message()
                .succeed_with_message("CMA-ES STEP SIZE CONVERGED");
        }
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
            covariance: self.covariance.scale(self.sigma * self.sigma),
        })
    }

    fn reset(&mut self) {
        self.mean = Vector::zeros(0);
        self.covariance = Matrix::zeros(0, 0);
        self.path_c = Vector::zeros(0);
        self.path_sigma = Vector::zeros(0);
        self.weights.clear();
        self.best_x = Vector::zeros(0);
        self.best_fx = T::infinity();
        self.generation = 0;
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
    fn cmaes_runs_f32_with_full_covariance_and_bounds() {
        let bounds = BackendBounds::new([(-3.0_f32, 3.0), (-3.0, 3.0)]).unwrap();
        let config = BackendCMAESConfig {
            population_size: 20,
            initial_sigma: 0.7,
            tolerance: 1e-5,
            ..BackendCMAESConfig::default()
        }
        .with_transform(bounds);
        let mut algorithm = BackendCMAES::<f32>::new(Some(31));
        let result = algorithm
            .process(
                &Rosenbrock { n: 2 },
                &(),
                Vector::from_vec(vec![-1.2, 1.0]),
                config,
                Callbacks::empty().with_terminator(MaxSteps(2_000)),
            )
            .unwrap();
        assert!(result.fx < 1e-3);
    }
}
