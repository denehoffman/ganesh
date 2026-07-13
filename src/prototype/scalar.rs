//! A narrow scalar-generic optimization surface for migration experiments.
//!
//! This module is intentionally separate from the main [`Algorithm`](`crate::traits::Algorithm`)
//! surface. It measures how much generic scalar code is needed before status, callback, summary,
//! transform, and Python types are migrated too.

use fastrand::Rng;
use std::marker::PhantomData;

use crate::core::{
    BackendMinimizationSummary, Callbacks, EvalCounts, LinearAlgebra, LinearSolve, Matrix,
    NalgebraBackend, PseudoInverse, RandomScalar, RealScalar, Vector,
};
pub use crate::traits::{
    Algorithm, BackendTransform, BackendTransformedProblem, CostFunction, Gradient, Status,
};
use crate::{
    algorithms::{gradient::BackendGradientStatus, gradient_free::BackendGradientFreeStatus},
    core::MaxSteps,
};

/// Minimal output shared by the scalar-generic optimizer prototypes.
#[derive(Clone, Debug)]
pub struct MinimizationResult<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Final parameters produced by the optimizer.
    pub x: Vector<T, B>,
    /// Final objective value.
    pub fx: T,
    /// Evaluation counts performed by the prototype.
    pub evals: EvalCounts,
}

/// Subproblem used by [`TrustRegion`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TrustRegionSubproblem {
    /// Cauchy-point steepest descent step.
    CauchyPoint,
    /// Powell dogleg step.
    #[default]
    Dogleg,
}

/// Minimal scalar-generic trust-region configuration.
pub struct TrustRegionConfig<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Subproblem solver used for each step.
    pub subproblem: TrustRegionSubproblem,
    /// Initial trust-region radius.
    pub initial_radius: T,
    /// Maximum trust-region radius.
    pub max_radius: T,
    /// Minimum ratio of actual to predicted reduction needed to accept a step.
    pub eta: T,
    /// Optional differentiable coordinate transform.
    pub transform: Option<Box<dyn BackendTransform<T, B>>>,
}

impl<T, B> Default for TrustRegionConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            subproblem: TrustRegionSubproblem::default(),
            initial_radius: T::one(),
            max_radius: T::literal(1_000.0),
            eta: T::literal(1e-4),
            transform: None,
        }
    }
}

impl<T, B> TrustRegionConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Configure a differentiable coordinate transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: BackendTransform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

/// RealScalar-generic trust-region prototype.
#[derive(Clone, Debug)]
pub struct TrustRegion<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    x: Vector<T, B>,
    f: T,
    g: Vector<T, B>,
    h: Matrix<T, B>,
    radius: T,
    max_radius: T,
    _backend: PhantomData<B>,
}

impl<T, B> Default for TrustRegion<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            x: Vector::<T, B>::zeros(0),
            f: T::zero(),
            g: Vector::<T, B>::zeros(0),
            h: Matrix::<T, B>::zeros(0, 0),
            radius: T::one(),
            max_radius: T::literal(1_000.0),
            _backend: PhantomData,
        }
    }
}

impl<T, B> TrustRegion<T, B>
where
    T: RealScalar,
    B: LinearSolve<T>,
{
    fn predicted_reduction(&self, p: &Vector<T, B>) -> T {
        -T::literal(0.5).mul_add(p.dot(&self.h.mul_vec(p)), self.g.dot(p))
    }

    fn cauchy_point(&self) -> Vector<T, B> {
        let g_norm = self.g.norm();
        if g_norm <= T::epsilon() {
            return Vector::<T, B>::zeros(self.g.len());
        }

        let g_bg = self.g.dot(&self.h.mul_vec(&self.g));
        let tau = if g_bg <= T::zero() {
            T::one()
        } else {
            let candidate = g_norm.powi(3) / (self.radius * g_bg);
            if candidate < T::one() {
                candidate
            } else {
                T::one()
            }
        };
        self.g.scale(-(tau * self.radius / g_norm))
    }

    fn dogleg(&self) -> Vector<T, B> {
        let p_u = {
            let g_bg = self.g.dot(&self.h.mul_vec(&self.g));
            if g_bg <= T::epsilon() {
                self.cauchy_point()
            } else {
                self.g.scale(-(self.g.dot(&self.g) / g_bg))
            }
        };
        if p_u.norm() >= self.radius {
            return p_u.scale(self.radius / p_u.norm());
        }

        let p_b = self.h.lu_solve(&self.g.neg()).filter(Vector::all_finite);
        let Some(p_b) = p_b else {
            return self.cauchy_point();
        };
        if p_b.norm() <= self.radius {
            return p_b;
        }

        let diff = p_b.sub(&p_u);
        let two = T::literal(2.0);
        let four = T::literal(4.0);
        let a = diff.dot(&diff);
        let b = two * p_u.dot(&diff);
        let c = self.radius.mul_add(-self.radius, p_u.dot(&p_u));
        let disc = b.mul_add(b, -four * a * c);
        let disc = if disc > T::zero() { disc } else { T::zero() };
        let tau = (-b + disc.sqrt()) / (two * a);
        p_u.add(&diff.scale(tau))
    }

    fn step_direction(&self, subproblem: TrustRegionSubproblem) -> Vector<T, B> {
        match subproblem {
            TrustRegionSubproblem::CauchyPoint => self.cauchy_point(),
            TrustRegionSubproblem::Dogleg => self.dogleg(),
        }
    }

    /// Run a fixed number of trust-region steps.
    ///
    /// # Errors
    ///
    /// Returns an error if the objective or derivatives fail to evaluate.
    pub fn run_steps<P, U, E>(
        &mut self,
        problem: &P,
        args: &U,
        init: Vector<T, B>,
        config: TrustRegionConfig<T, B>,
        steps: usize,
    ) -> Result<MinimizationResult<T, B>, E>
    where
        P: Gradient<T, B, U, E>,
    {
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        self.x = transformed.to_internal(&init);
        (self.f, self.g, self.h) = transformed.evaluate_with_gradient_and_hessian(&self.x, args)?;
        self.radius = config.initial_radius;
        self.max_radius = if config.max_radius > config.initial_radius {
            config.max_radius
        } else {
            config.initial_radius
        };

        let mut evals = EvalCounts::new(1, 1, 1);
        let quarter = T::literal(0.25);
        let three_quarters = T::literal(0.75);
        let two = T::literal(2.0);
        let boundary_tol_scale = T::literal(1e-12);

        for _ in 0..steps {
            let p = self.step_direction(config.subproblem);
            let predicted = self.predicted_reduction(&p);
            if predicted <= T::epsilon() {
                self.radius = self.radius * quarter;
                continue;
            }

            let x_trial = self.x.add(&p);
            let f_trial = transformed.evaluate(&x_trial, args)?;
            evals.record_f();
            let rho = (self.f - f_trial) / predicted;
            let hits_boundary =
                (p.norm() - self.radius).abs() <= boundary_tol_scale * (T::one() + self.radius);

            if rho < quarter {
                self.radius = quarter * p.norm();
            } else if rho > three_quarters && hits_boundary {
                let candidate = two * self.radius;
                self.radius = if candidate < self.max_radius {
                    candidate
                } else {
                    self.max_radius
                };
            }

            if rho > config.eta {
                let (g_trial, h_trial) = transformed.gradient_with_hessian(&x_trial, args)?;
                evals.record_gh();
                self.x = x_trial;
                self.f = f_trial;
                self.g = g_trial;
                self.h = h_trial;
            }
        }

        Ok(MinimizationResult {
            x: transformed.to_external(&self.x),
            fx: self.f,
            evals,
        })
    }
}

impl<T, B, P, U, E> Algorithm<P, BackendGradientStatus<T, B>, U, E> for TrustRegion<T, B>
where
    T: RealScalar,
    B: LinearSolve<T> + PseudoInverse<T>,
    P: Gradient<T, B, U, E>,
{
    type Summary = BackendMinimizationSummary<T, B>;
    type Config = TrustRegionConfig<T, B>;
    type Init = Vector<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut BackendGradientStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        self.x = transformed.to_internal(init);
        (self.f, self.g, self.h) = transformed.evaluate_with_gradient_and_hessian(&self.x, args)?;
        self.radius = config.initial_radius;
        self.max_radius = if config.max_radius > config.initial_radius {
            config.max_radius
        } else {
            config.initial_radius
        };
        status.evals.record_fgh();
        status.initialize(init.clone(), self.f);
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut BackendGradientStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        if self.g.norm() <= T::epsilon().cbrt() {
            status
                .set_message()
                .succeed_with_message("GRADIENT CONVERGED");
            return Ok(());
        }

        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        let p = self.step_direction(config.subproblem);
        let predicted = self.predicted_reduction(&p);
        let quarter = T::literal(0.25);
        if predicted <= T::epsilon() {
            self.radius = self.radius * quarter;
            return Ok(());
        }

        let x_trial = self.x.add(&p);
        let f_trial = transformed.evaluate(&x_trial, args)?;
        status.evals.record_f();
        let rho = (self.f - f_trial) / predicted;
        let three_quarters = T::literal(0.75);
        let hits_boundary =
            (p.norm() - self.radius).abs() <= T::literal(1e-12) * (T::one() + self.radius);
        if rho < quarter {
            self.radius = quarter * p.norm();
        } else if rho > three_quarters && hits_boundary {
            let candidate = T::literal(2.0) * self.radius;
            self.radius = if candidate < self.max_radius {
                candidate
            } else {
                self.max_radius
            };
        }

        if rho > config.eta {
            let (gradient, hessian) = transformed.gradient_with_hessian(&x_trial, args)?;
            status.evals.record_gh();
            self.x = x_trial;
            self.f = f_trial;
            self.g = gradient;
            self.h = hessian;
            status.set_position(transformed.to_external(&self.x), self.f);
        }
        Ok(())
    }

    fn postprocessing(
        &mut self,
        problem: &P,
        status: &mut BackendGradientStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        let hessian = problem.hessian(&transformed.to_external(&self.x), args)?;
        status.evals.record_h();
        status.set_hess(hessian);
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &BackendGradientStatus<T, B>,
        _args: &U,
        init: &Self::Init,
        _config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        let n = status.x.len();
        Ok(BackendMinimizationSummary {
            parameter_names: None,
            message: status.message.clone(),
            x0: init.clone(),
            x: status.x.clone(),
            std: status.err.clone().unwrap_or_else(|| Vector::zeros(n)),
            fx: status.fx,
            evals: status.evals,
            covariance: status.cov.clone().unwrap_or_else(|| Matrix::identity(n)),
        })
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn default_callbacks() -> Callbacks<Self, P, BackendGradientStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty().with_terminator(MaxSteps::default())
    }
}

/// Minimal scalar-generic differential evolution configuration.
pub struct DifferentialEvolutionConfig<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Population size used during the run.
    pub population_size: usize,
    /// Differential weight `F`.
    pub differential_weight: T,
    /// Binomial crossover probability `CR`.
    pub crossover_probability: T,
    /// Half-width of the initial uniform perturbation around the starting point.
    pub initial_scale: T,
    /// Optional coordinate transform, including backend-generic bounds.
    pub transform: Option<Box<dyn BackendTransform<T, B>>>,
}

impl<T, B> Default for DifferentialEvolutionConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            population_size: 0,
            differential_weight: T::literal(0.8),
            crossover_probability: T::literal(0.9),
            initial_scale: T::one(),
            transform: None,
        }
    }
}

impl<T, B> DifferentialEvolutionConfig<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Configure a coordinate transform or smooth bounds transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: BackendTransform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

#[derive(Clone, Debug)]
struct Candidate<T: RealScalar, B: LinearAlgebra<T>> {
    x: Vector<T, B>,
    fx: T,
}

/// RealScalar-generic `DE/rand/1/bin` prototype.
#[derive(Clone, Debug)]
pub struct DifferentialEvolution<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    rng: Rng,
    population: Vec<Candidate<T, B>>,
    best: Candidate<T, B>,
    _backend: PhantomData<B>,
}

impl<T, B> DifferentialEvolution<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Construct a differential evolution prototype with an optional seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::new, Rng::with_seed),
            population: Vec::new(),
            best: Candidate {
                x: Vector::<T, B>::zeros(0),
                fx: T::zero(),
            },
            _backend: PhantomData,
        }
    }

    fn sample_offset(&mut self, dim: usize, scale: T) -> Vector<T, B> {
        let two = T::literal(2.0);
        Vector::<T, B>::from_vec(
            (0..dim)
                .map(|_| T::random_unit(&mut self.rng).mul_add(two * scale, -scale))
                .collect(),
        )
    }

    fn choose_distinct_indices(&mut self, population_size: usize, target: usize) -> [usize; 3] {
        let mut a = self.rng.usize(0..population_size);
        while a == target {
            a = self.rng.usize(0..population_size);
        }
        let mut b = self.rng.usize(0..population_size);
        while b == target || b == a {
            b = self.rng.usize(0..population_size);
        }
        let mut c = self.rng.usize(0..population_size);
        while c == target || c == a || c == b {
            c = self.rng.usize(0..population_size);
        }
        [a, b, c]
    }

    /// Run a fixed number of differential evolution generations.
    ///
    /// # Errors
    ///
    /// Returns an error if the objective fails to evaluate.
    pub fn run_steps<P, U, E>(
        &mut self,
        problem: &P,
        args: &U,
        init: Vector<T, B>,
        config: DifferentialEvolutionConfig<T, B>,
        steps: usize,
    ) -> Result<MinimizationResult<T, B>, E>
    where
        P: CostFunction<T, B, U, E>,
    {
        let transformed = BackendTransformedProblem::new(problem, config.transform.as_deref());
        let init_external = init;
        let init = transformed.to_internal(&init_external);
        let pop_size = if config.population_size == 0 {
            (10 * init.len()).max(4)
        } else {
            config.population_size.max(4)
        };
        self.population.clear();
        self.population.reserve(pop_size);

        let fx = transformed.evaluate(&init, args)?;
        self.best = Candidate {
            x: init.clone(),
            fx,
        };
        self.population.push(self.best.clone());

        for _ in 1..pop_size {
            let x = init.add(&self.sample_offset(init.len(), config.initial_scale));
            let candidate = Candidate {
                fx: transformed.evaluate(&x, args)?,
                x,
            };
            if candidate.fx < self.best.fx {
                self.best = candidate.clone();
            }
            self.population.push(candidate);
        }

        let mut evals = EvalCounts::new(self.population.len(), 0, 0);
        for _ in 0..steps {
            let snapshot = self.population.clone();
            let dim = self.best.x.len();
            for i in 0..self.population.len() {
                let [a, b, c] = self.choose_distinct_indices(snapshot.len(), i);
                let mutant = snapshot[a].x.add(
                    &snapshot[b]
                        .x
                        .sub(&snapshot[c].x)
                        .scale(config.differential_weight),
                );
                let forced_index = self.rng.usize(0..dim);
                let trial_x = Vector::<T, B>::from_vec(
                    (0..dim)
                        .map(|j| {
                            if j == forced_index
                                || T::random_unit(&mut self.rng) < config.crossover_probability
                            {
                                mutant.get(j)
                            } else {
                                self.population[i].x.get(j)
                            }
                        })
                        .collect(),
                );
                let trial = Candidate {
                    fx: transformed.evaluate(&trial_x, args)?,
                    x: trial_x,
                };
                evals.record_f();
                if trial.fx <= self.population[i].fx {
                    self.population[i] = trial;
                    if self.population[i].fx < self.best.fx {
                        self.best = self.population[i].clone();
                    }
                }
            }
        }

        Ok(MinimizationResult {
            x: transformed.to_external(&self.best.x),
            fx: self.best.fx,
            evals,
        })
    }
}

impl<T, B, P, U, E> Algorithm<P, BackendGradientFreeStatus<T, B>, U, E>
    for DifferentialEvolution<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    P: CostFunction<T, B, U, E>,
{
    type Summary = BackendMinimizationSummary<T, B>;
    type Config = DifferentialEvolutionConfig<T, B>;
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
        let init = transformed.to_internal(init);
        let population_size = if config.population_size == 0 {
            (10 * init.len()).max(4)
        } else {
            config.population_size.max(4)
        };
        self.population.clear();
        self.population.reserve(population_size);

        self.best = Candidate {
            fx: transformed.evaluate(&init, args)?,
            x: init.clone(),
        };
        self.population.push(self.best.clone());
        for _ in 1..population_size {
            let x = init.add(&self.sample_offset(init.len(), config.initial_scale));
            let candidate = Candidate {
                fx: transformed.evaluate(&x, args)?,
                x,
            };
            if candidate.fx < self.best.fx {
                self.best = candidate.clone();
            }
            self.population.push(candidate);
        }
        status.evals.record_many_f(self.population.len());
        status.initialize(transformed.to_external(&self.best.x), self.best.fx);
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
        let snapshot = self.population.clone();
        let dimension = self.best.x.len();
        for index in 0..self.population.len() {
            let [a, b, c] = self.choose_distinct_indices(snapshot.len(), index);
            let mutant = snapshot[a].x.add(
                &snapshot[b]
                    .x
                    .sub(&snapshot[c].x)
                    .scale(config.differential_weight),
            );
            let forced_index = self.rng.usize(0..dimension);
            let trial_x = Vector::from_vec(
                (0..dimension)
                    .map(|coordinate| {
                        if coordinate == forced_index
                            || T::random_unit(&mut self.rng) < config.crossover_probability
                        {
                            mutant.get(coordinate)
                        } else {
                            self.population[index].x.get(coordinate)
                        }
                    })
                    .collect(),
            );
            let trial = Candidate {
                fx: transformed.evaluate(&trial_x, args)?,
                x: trial_x,
            };
            status.evals.record_f();
            if trial.fx <= self.population[index].fx {
                self.population[index] = trial;
                if self.population[index].fx < self.best.fx {
                    self.best = self.population[index].clone();
                }
            }
        }
        status.set_position(transformed.to_external(&self.best.x), self.best.fx);
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
        self.population.clear();
        self.best = Candidate {
            x: Vector::zeros(0),
            fx: T::zero(),
        };
    }

    fn default_callbacks() -> Callbacks<Self, P, BackendGradientFreeStatus<T, B>, U, E, Self::Config>
    {
        Callbacks::empty().with_terminator(MaxSteps::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "backend-ndarray")]
    use crate::core::NdArrayBackend;
    use crate::traits::{BackendBounds, BackendScaleTransform};
    use approx::assert_relative_eq;
    use std::convert::Infallible;

    struct Quadratic;

    impl<T, B> CostFunction<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn evaluate(&self, x: &Vector<T, B>, _args: &()) -> Result<T, Infallible> {
            Ok(x.dot(x))
        }
    }

    impl<T, B> Gradient<T, B> for Quadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn gradient(&self, x: &Vector<T, B>, _args: &()) -> Result<Vector<T, B>, Infallible> {
            Ok(x.scale(T::literal(2.0)))
        }

        fn hessian(&self, x: &Vector<T, B>, _args: &()) -> Result<Matrix<T, B>, Infallible> {
            Ok(Matrix::<T, B>::identity(x.len()).scale(T::literal(2.0)))
        }
    }

    fn vector<T, B>(values: &[f64]) -> Vector<T, B>
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        Vector::<T, B>::from_vec(values.iter().map(|value| T::literal(*value)).collect())
    }

    #[test]
    fn trust_region_f64_reaches_quadratic_minimum() {
        let result = TrustRegion::<f64>::default()
            .run_steps(
                &Quadratic,
                &(),
                vector::<f64, NalgebraBackend>(&[3.0, -2.0]),
                TrustRegionConfig::default(),
                12,
            )
            .unwrap();
        assert!(result.fx < 1e-20);
        assert_relative_eq!(Vector::get(&result.x, 0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(Vector::get(&result.x, 1), 0.0, epsilon = 1e-10);
        assert!(result.evals.g() > 0);
        assert!(result.evals.h() > 0);
    }

    #[test]
    fn default_scalar_syntax_is_f64_and_explicit_f32_compiles() {
        let _: TrustRegion = TrustRegion::default();
        let _: TrustRegion<f32> = TrustRegion::<f32>::default();
        let _: TrustRegionConfig = TrustRegionConfig::default();
        let _: DifferentialEvolution = DifferentialEvolution::new(Some(1));
        let _: DifferentialEvolutionConfig = DifferentialEvolutionConfig::default();
    }

    #[test]
    fn differential_evolution_f64_is_seeded() {
        let run = || {
            let config = DifferentialEvolutionConfig {
                population_size: 24,
                initial_scale: 2.0,
                ..DifferentialEvolutionConfig::default()
            };
            DifferentialEvolution::<f64>::new(Some(7))
                .run_steps(
                    &Quadratic,
                    &(),
                    vector::<f64, NalgebraBackend>(&[3.0, -2.0]),
                    config,
                    80,
                )
                .unwrap()
        };
        let result_a = run();
        let result_b = run();
        assert_eq!(result_a.fx, result_b.fx);
        assert_eq!(result_a.x, result_b.x);
        assert!(result_a.fx < 1e-4);
    }

    #[test]
    fn differential_evolution_uses_algorithm_lifecycle_with_bounds() {
        let bounds =
            BackendBounds::<f32, NalgebraBackend>::new([(-4.0, 4.0), (-4.0, 4.0)]).unwrap();
        let config = DifferentialEvolutionConfig {
            population_size: 24,
            initial_scale: 1.0,
            ..DifferentialEvolutionConfig::default()
        }
        .with_transform(bounds);
        let mut algorithm = DifferentialEvolution::<f32>::new(Some(7));
        let result = algorithm
            .process(
                &Quadratic,
                &(),
                vector::<f32, NalgebraBackend>(&[3.0, -2.0]),
                config,
                Callbacks::empty().with_terminator(MaxSteps(80)),
            )
            .unwrap();
        assert!(result.fx < 1e-4);
        assert!(result.x.get(0).abs() < 0.1);
        assert!(result.x.get(1).abs() < 0.1);
        assert!(result.evals.f() > 24);
    }

    #[test]
    fn trust_region_f32_builds_in_the_default_crate() {
        let result = TrustRegion::<f32>::default()
            .run_steps(
                &Quadratic,
                &(),
                vector::<f32, NalgebraBackend>(&[3.0, -2.0]),
                TrustRegionConfig::default(),
                12,
            )
            .unwrap();
        assert!(result.fx < 1e-10);
    }

    #[test]
    fn trust_region_uses_the_production_algorithm_lifecycle() {
        let mut algorithm = TrustRegion::<f32>::default();
        let result = algorithm
            .process_default(
                &Quadratic,
                &(),
                vector::<f32, NalgebraBackend>(&[3.0, -2.0]),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 1e-8);
        assert!(result.evals.f() > 0);
        assert!(result.evals.g() > 0);
        assert!(result.evals.h() > 0);
    }

    #[test]
    fn trust_region_runs_through_backend_generic_transform() {
        let transform =
            BackendScaleTransform::<f32, NalgebraBackend>::from_parameter_scales([2.0, 0.5])
                .unwrap();
        let config = TrustRegionConfig::default().with_transform(transform);
        let mut algorithm = TrustRegion::<f32>::default();
        let result = algorithm
            .process(
                &Quadratic,
                &(),
                vector::<f32, NalgebraBackend>(&[3.0, -2.0]),
                config,
                TrustRegion::<f32>::default_callbacks(),
            )
            .unwrap();
        assert!(result.message.success());
        assert!(result.fx < 1e-8);
    }

    #[cfg(feature = "backend-ndarray")]
    #[test]
    fn trust_region_runs_with_ndarray_backend() {
        let result = TrustRegion::<f64, NdArrayBackend>::default()
            .run_steps(
                &Quadratic,
                &(),
                vector::<f64, NdArrayBackend>(&[3.0, -2.0]),
                TrustRegionConfig::default(),
                12,
            )
            .unwrap();
        assert!(result.fx < 1e-20);
        assert_relative_eq!(Vector::get(&result.x, 0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(Vector::get(&result.x, 1), 0.0, epsilon = 1e-10);
    }

    #[cfg(feature = "backend-ndarray")]
    #[test]
    fn differential_evolution_runs_with_ndarray_backend() {
        let config = DifferentialEvolutionConfig {
            population_size: 24,
            initial_scale: 2.0,
            ..DifferentialEvolutionConfig::default()
        };
        let result = DifferentialEvolution::<f64, NdArrayBackend>::new(Some(7))
            .run_steps(
                &Quadratic,
                &(),
                vector::<f64, NdArrayBackend>(&[3.0, -2.0]),
                config,
                80,
            )
            .unwrap();
        assert!(result.fx < 1e-4);
    }
}
