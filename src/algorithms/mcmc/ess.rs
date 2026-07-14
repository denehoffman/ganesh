//! Scalar- and linear-algebra-generic ensemble slice sampling.

#![allow(clippy::suboptimal_flops)]

use crate::algorithms::mcmc::{aies::validate_walkers, ChainStorageMode, EnsembleStatus};
use crate::core::{
    utils::sample_standard_normal, Callbacks, LinearAlgebra, MCMCSummary, NalgebraProvider,
    RandomScalar, Vector,
};
use crate::traits::{Algorithm, LogDensity, SupportsParameterNames, Transform, TransformedProblem};
use fastrand::Rng;
use std::marker::PhantomData;

mod dpgm {
    //! Dirichlet Process Gaussian Mixture
    //!
    //! Code is taken almost verbatim (converting numpy to nalgebra) from
    //! <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_bayesian_mixture.py#L74>
    //! with some modifications to only use the "full"" covariance mode, the "kmeans" initialization
    //! method, and the "`dirichlet_process`" weight concentration prior. See the readme/crate
    //! documentation for the proper citation.
    use fastrand::Rng;
    use logsumexp::LogSumExp;
    use nalgebra::{Cholesky, DMatrix, DVector};
    use spec_math::{Beta, Gamma};

    pub(super) struct DPGMResult {
        pub labels: Vec<usize>,
        pub means: Vec<DVector<f64>>,
        pub covariances: Vec<DMatrix<f64>>,
    }

    fn kmeans(components: usize, data: &DMatrix<f64>, rng: &mut Rng) -> Vec<usize> {
        let limits = data
            .column_iter()
            .map(|column| {
                column
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
                        (low.min(*value), high.max(*value))
                    })
            })
            .collect::<Vec<_>>();
        let mut centroids = (0..components)
            .map(|_| {
                DVector::from_iterator(
                    data.ncols(),
                    limits
                        .iter()
                        .map(|(low, high)| low + (high - low) * rng.f64()),
                )
            })
            .collect::<Vec<_>>();
        let mut labels = vec![0; data.nrows()];
        for _ in 0..50 {
            for (row_index, row) in data.row_iter().enumerate() {
                labels[row_index] = centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, left), (_, right)| {
                        (row.transpose() - *left)
                            .norm_squared()
                            .total_cmp(&(row.transpose() - *right).norm_squared())
                    })
                    .map_or(0, |(index, _)| index);
            }
            for (component, centroid) in centroids.iter_mut().enumerate() {
                let mut sum = DVector::zeros(data.ncols());
                let mut count = 0;
                for (label, row) in labels.iter().zip(data.row_iter()) {
                    if *label == component {
                        sum += row.transpose();
                        count += 1;
                    }
                }
                if count > 0 {
                    sum /= count as f64;
                }
                *centroid = sum;
            }
        }
        labels
    }

    fn covariance_prior(data: &DMatrix<f64>) -> DMatrix<f64> {
        let dimensions = data.ncols();
        let walkers = data.nrows();
        let mean = DVector::from_iterator(
            dimensions,
            (0..dimensions).map(|column| data.column(column).iter().sum::<f64>() / walkers as f64),
        );
        let mut covariance = DMatrix::zeros(dimensions, dimensions);
        for row in data.row_iter() {
            let difference = row.transpose() - &mean;
            covariance += &difference * difference.transpose();
        }
        covariance / (walkers.saturating_sub(1).max(1) as f64)
    }

    fn gaussian_parameters(
        data: &DMatrix<f64>,
        responsibilities: &DMatrix<f64>,
    ) -> (DVector<f64>, DMatrix<f64>, Vec<DMatrix<f64>>) {
        let components = responsibilities.ncols();
        let dimensions = data.ncols();
        let mut counts = DVector::zeros(components);
        let mut means = DMatrix::zeros(components, dimensions);
        for component in 0..components {
            counts[component] =
                responsibilities.column(component).iter().sum::<f64>() + 10.0 * f64::EPSILON;
            for walker in 0..data.nrows() {
                for dimension in 0..dimensions {
                    means[(component, dimension)] +=
                        responsibilities[(walker, component)] * data[(walker, dimension)];
                }
            }
            for dimension in 0..dimensions {
                means[(component, dimension)] /= counts[component];
            }
        }
        let covariances = (0..components)
            .map(|component| {
                let mut covariance = DMatrix::zeros(dimensions, dimensions);
                for walker in 0..data.nrows() {
                    let difference =
                        data.row(walker).transpose() - means.row(component).transpose();
                    covariance += (&difference * difference.transpose())
                        * responsibilities[(walker, component)];
                }
                covariance /= counts[component];
                for index in 0..dimensions {
                    covariance[(index, index)] += 1e-6;
                }
                covariance
            })
            .collect();
        (counts, means, covariances)
    }

    fn weights(counts: &DVector<f64>, prior: f64) -> (DVector<f64>, DVector<f64>) {
        let components = counts.len();
        let a = counts.map(|value| value + 1.0);
        let mut b = DVector::zeros(components);
        let mut tail = 0.0;
        for component in (0..components).rev() {
            b[component] = tail + prior;
            tail += counts[component];
        }
        (a, b)
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn posterior(
        counts: &DVector<f64>,
        empirical_means: &DMatrix<f64>,
        empirical_covariances: &[DMatrix<f64>],
        mean_prior: &DVector<f64>,
        covariance_prior: &DMatrix<f64>,
    ) -> Option<(
        DVector<f64>,
        DMatrix<f64>,
        DVector<f64>,
        Vec<DMatrix<f64>>,
        Vec<DMatrix<f64>>,
    )> {
        let components = counts.len();
        let dimensions = mean_prior.len();
        let mean_precision = counts.map(|count| count + 1.0);
        let mut means = DMatrix::zeros(components, dimensions);
        let degrees = counts.map(|count| count + dimensions as f64);
        let mut covariances = Vec::with_capacity(components);
        let mut precision_cholesky = Vec::with_capacity(components);
        for component in 0..components {
            for dimension in 0..dimensions {
                means[(component, dimension)] = (counts[component]
                    * empirical_means[(component, dimension)]
                    + mean_prior[dimension])
                    / mean_precision[component];
            }
            let difference = empirical_means.row(component).transpose() - mean_prior;
            let covariance = (covariance_prior
                + &empirical_covariances[component] * counts[component]
                + (&difference * difference.transpose())
                    * (counts[component] / mean_precision[component]))
                / degrees[component];
            let decomposition = Cholesky::new(covariance.clone())?;
            let inverse_lower = decomposition
                .l()
                .solve_lower_triangular(&DMatrix::identity(dimensions, dimensions))?;
            covariances.push(covariance);
            precision_cholesky.push(inverse_lower.transpose());
        }
        Some((
            mean_precision,
            means,
            degrees,
            covariances,
            precision_cholesky,
        ))
    }

    fn e_step(
        data: &DMatrix<f64>,
        means: &DMatrix<f64>,
        precision_cholesky: &[DMatrix<f64>],
        mean_precision: &DVector<f64>,
        degrees: &DVector<f64>,
        concentration: &(DVector<f64>, DVector<f64>),
    ) -> (f64, DMatrix<f64>) {
        let dimensions = data.ncols();
        let components = means.nrows();
        let mut log_probability = DMatrix::zeros(data.nrows(), components);
        let mut expected_log_weights = DVector::zeros(components);
        let mut cumulative = 0.0;
        for component in 0..components {
            let sum = concentration.0[component] + concentration.1[component];
            expected_log_weights[component] =
                concentration.0[component].digamma() - sum.digamma() + cumulative;
            cumulative += concentration.1[component].digamma() - sum.digamma();
            let log_determinant = (0..dimensions)
                .map(|index| precision_cholesky[component][(index, index)].ln())
                .sum::<f64>();
            let log_lambda = dimensions as f64 * 2.0_f64.ln()
                + (0..dimensions)
                    .map(|index| (0.5 * (degrees[component] - index as f64)).digamma())
                    .sum::<f64>();
            for walker in 0..data.nrows() {
                let centered = data.row(walker) - means.row(component);
                let transformed = centered * &precision_cholesky[component];
                let square = transformed.iter().map(|value| value * value).sum::<f64>();
                let gaussian = log_determinant
                    - 0.5 * (dimensions as f64 * (2.0 * std::f64::consts::PI).ln() + square)
                    - 0.5 * dimensions as f64 * degrees[component].ln()
                    + 0.5 * (log_lambda - dimensions as f64 / mean_precision[component]);
                log_probability[(walker, component)] = gaussian + expected_log_weights[component];
            }
        }
        let mut mean_log_norm = 0.0;
        for walker in 0..data.nrows() {
            let norm = log_probability.row(walker).iter().copied().ln_sum_exp();
            mean_log_norm += norm;
            for component in 0..components {
                log_probability[(walker, component)] -= norm;
            }
        }
        (mean_log_norm / data.nrows() as f64, log_probability)
    }

    pub(super) fn fit(components: usize, data: &DMatrix<f64>, rng: &mut Rng) -> Option<DPGMResult> {
        if data.nrows() < components || data.ncols() == 0 {
            return None;
        }
        let mean_prior = DVector::from_iterator(
            data.ncols(),
            (0..data.ncols())
                .map(|column| data.column(column).iter().sum::<f64>() / data.nrows() as f64),
        );
        let covariance_prior = covariance_prior(data);
        let mut responsibilities = DMatrix::zeros(data.nrows(), components);
        for (walker, label) in kmeans(components, data, rng).into_iter().enumerate() {
            responsibilities[(walker, label)] = 1.0;
        }
        let concentration_prior = 1.0 / components as f64;
        let mut lower_bound = f64::NEG_INFINITY;
        let mut final_state = None;
        for _ in 0..100 {
            let (counts, empirical_means, empirical_covariances) =
                gaussian_parameters(data, &responsibilities);
            let concentration = weights(&counts, concentration_prior);
            let (mean_precision, means, degrees, covariances, precision_cholesky) = posterior(
                &counts,
                &empirical_means,
                &empirical_covariances,
                &mean_prior,
                &covariance_prior,
            )?;
            let (new_bound, log_responsibilities) = e_step(
                data,
                &means,
                &precision_cholesky,
                &mean_precision,
                &degrees,
                &concentration,
            );
            responsibilities = log_responsibilities.map(f64::exp);
            final_state = Some((
                means,
                covariances,
                log_responsibilities,
                concentration,
                mean_precision,
                degrees,
                precision_cholesky,
            ));
            if (new_bound - lower_bound).abs() < 1e-3 {
                break;
            }
            lower_bound = new_bound;
        }
        let (means, covariances, log_responsibilities, concentration, _, _, _) = final_state?;
        // Evaluate the stick-breaking normalization as in the variational DPGM model.
        let mut remaining = 1.0;
        let mut mixture_weights = DVector::zeros(components);
        for component in 0..components {
            let sum = concentration.0[component] + concentration.1[component];
            mixture_weights[component] = remaining * concentration.0[component] / sum;
            remaining *= concentration.1[component] / sum;
        }
        let normalization = mixture_weights.sum();
        if normalization > 0.0 {
            mixture_weights /= normalization;
        }
        // Keep the exact beta-normalization calculation exercised by the original implementation.
        let _log_norm_weight = -(0..components)
            .map(|index| concentration.0[index].lbeta(concentration.1[index]))
            .sum::<f64>();
        Some(DPGMResult {
            labels: (0..data.nrows())
                .map(|walker| {
                    (0..components)
                        .max_by(|left, right| {
                            log_responsibilities[(walker, *left)]
                                .total_cmp(&log_responsibilities[(walker, *right)])
                        })
                        .unwrap_or(0)
                })
                .collect(),
            means: means.row_iter().map(|row| row.transpose()).collect(),
            covariances,
        })
    }

    pub(super) fn sample(
        mean: &DVector<f64>,
        covariance: &DMatrix<f64>,
        rng: &mut Rng,
    ) -> Option<DVector<f64>> {
        let factor = Cholesky::new(covariance.clone())?.l();
        let normal = DVector::from_iterator(
            mean.len(),
            (0..mean.len()).map(|_| {
                let u1 = rng.f64().max(f64::MIN_POSITIVE);
                let u2 = rng.f64();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            }),
        );
        Some(mean + factor * normal)
    }
}

/// Direction proposal used by linear-algebra-generic ensemble slice sampling.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ESSMove<T: RandomScalar = f64> {
    /// Difference between two complementary walkers.
    Differential,
    /// Gaussian combination of complementary-walker deviations.
    Gaussian,
    /// Broad ensemble Gaussian proposal for multimodal exploration.
    Global {
        /// Within-ensemble proposal scale.
        scale: T,
        /// Additional covariance rescaling used for broad jumps.
        rescale_covariance: T,
        /// Requested mixture-component count.
        components: usize,
    },
}

impl<T: RandomScalar> ESSMove<T> {
    /// A differential move paired with a selection weight.
    pub const fn differential(weight: T) -> (Self, T) {
        (Self::Differential, weight)
    }

    /// A Gaussian move paired with a selection weight.
    pub const fn gaussian(weight: T) -> (Self, T) {
        (Self::Gaussian, weight)
    }

    /// A global move with standard hyperparameters paired with a selection weight.
    pub fn global(weight: T) -> (Self, T) {
        (
            Self::Global {
                scale: T::one(),
                rescale_covariance: T::literal(0.001),
                components: 5,
            },
            weight,
        )
    }

    /// A global move with custom hyperparameters.
    ///
    /// # Errors
    /// Returns a configuration error for non-positive scales or fewer than two components.
    pub fn custom_global(
        weight: T,
        scale: Option<T>,
        rescale_covariance: Option<T>,
        components: Option<usize>,
    ) -> crate::error::GaneshResult<(Self, T)> {
        let scale = scale.unwrap_or_else(T::one);
        let rescale_covariance = rescale_covariance.unwrap_or_else(|| T::literal(0.001));
        let components = components.unwrap_or(5);
        if !scale.is_finite()
            || scale <= T::zero()
            || !rescale_covariance.is_finite()
            || rescale_covariance <= T::zero()
            || components < 2
        {
            return Err(crate::error::GaneshError::ConfigError(
                "ESS global move requires positive finite scales and at least two components"
                    .to_string(),
            ));
        }
        Ok((
            Self::Global {
                scale,
                rescale_covariance,
                components,
            },
            weight,
        ))
    }
}

/// Configuration for linear-algebra-generic ensemble slice sampling.
pub struct ESSConfig<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Initial one-dimensional slice bracket width.
    bracket_width: T,
    /// Maximum bracket-shrink evaluations per walker and ensemble step.
    max_shrink_steps: usize,
    /// Weighted direction-move mixture; an empty list uses a differential move.
    moves: Vec<(ESSMove<T>, T)>,
    /// Number of initial steps during which the direction scale is adapted.
    adaptive_steps: usize,
    /// Initial direction scale.
    direction_scale: T,
    /// Chain retention policy.
    chain_storage: ChainStorageMode,
    /// Optional names for the sampled parameters.
    parameter_names: Option<Vec<String>>,
    /// Optional coordinate transform.
    transform: Option<Box<dyn Transform<T, B>>>,
}

impl<T: RandomScalar, B: LinearAlgebra<T>> SupportsParameterNames for ESSConfig<T, B> {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

impl<T, B> Default for ESSConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            bracket_width: T::one(),
            max_shrink_steps: 10_000,
            moves: Vec::new(),
            adaptive_steps: 0,
            direction_scale: T::one(),
            chain_storage: ChainStorageMode::default(),
            parameter_names: None,
            transform: None,
        }
    }
}

impl<T, B> ESSConfig<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Create a configuration with default move settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the one-dimensional slice bracket width.
    pub fn with_bracket_width(mut self, width: T) -> crate::error::GaneshResult<Self> {
        if !width.is_finite() || width <= T::zero() {
            return Err(crate::error::GaneshError::ConfigError(
                "ESS bracket width must be finite and positive".to_string(),
            ));
        }
        self.bracket_width = width;
        Ok(self)
    }

    /// Set the maximum bracket expansion and contraction evaluations.
    pub fn with_max_shrink_steps(mut self, steps: usize) -> crate::error::GaneshResult<Self> {
        if steps == 0 {
            return Err(crate::error::GaneshError::ConfigError(
                "ESS maximum shrink steps must be at least 1".to_string(),
            ));
        }
        self.max_shrink_steps = steps;
        Ok(self)
    }

    /// Set the number of initial adaptive moves.
    pub const fn with_n_adaptive(self, steps: usize) -> Self {
        self.with_adaptive_steps(steps)
    }

    /// Set the maximum expansion and contraction count.
    pub const fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_shrink_steps = steps;
        self
    }

    /// Set the adaptive direction scale.
    pub fn with_mu(self, scale: T) -> crate::error::GaneshResult<Self> {
        self.with_direction_scale(scale)
    }

    /// Replace the weighted direction-move mixture.
    ///
    /// # Errors
    /// Returns a configuration error when weights are invalid or all zero.
    pub fn with_moves<I>(mut self, moves: I) -> crate::error::GaneshResult<Self>
    where
        I: IntoIterator<Item = (ESSMove<T>, T)>,
    {
        let moves: Vec<_> = moves.into_iter().collect();
        if moves.is_empty()
            || moves
                .iter()
                .any(|(_, weight)| !weight.is_finite() || *weight < T::zero())
            || moves.iter().all(|(_, weight)| *weight == T::zero())
        {
            return Err(crate::error::GaneshError::ConfigError(
                "ESS move weights must be finite, non-negative, and include a positive entry"
                    .to_string(),
            ));
        }
        self.moves = moves;
        Ok(self)
    }

    /// Set the number of initial scale-adaptation steps.
    pub const fn with_adaptive_steps(mut self, adaptive_steps: usize) -> Self {
        self.adaptive_steps = adaptive_steps;
        self
    }

    /// Set the positive direction scale.
    ///
    /// # Errors
    /// Returns a configuration error when the scale is non-finite or non-positive.
    pub fn with_direction_scale(mut self, direction_scale: T) -> crate::error::GaneshResult<Self> {
        if !direction_scale.is_finite() || direction_scale <= T::zero() {
            return Err(crate::error::GaneshError::ConfigError(
                "ESS direction scale must be finite and positive".to_string(),
            ));
        }
        self.direction_scale = direction_scale;
        Ok(self)
    }

    /// Select how much chain history is retained.
    pub const fn with_chain_storage(mut self, chain_storage: ChainStorageMode) -> Self {
        self.chain_storage = chain_storage;
        self
    }

    /// Configure a coordinate transform.
    pub fn with_transform<X>(mut self, transform: X) -> Self
    where
        X: Transform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        self
    }
}

/// Validated starting walkers for an [`ESS`] run.
#[derive(Clone, Debug)]
pub struct ESSInit<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    walkers: Vec<Vector<T, B>>,
}

impl<T: RandomScalar, B: LinearAlgebra<T>> ESSInit<T, B> {
    /// Validate and store starting walker positions.
    pub fn new(walkers: Vec<Vector<T, B>>) -> crate::error::GaneshResult<Self> {
        validate_walkers(&walkers, "ESS", 3)?;
        Ok(Self { walkers })
    }
}

/// Scalar- and linear-algebra-generic differential-direction ensemble slice sampler.
#[derive(Clone, Debug)]
pub struct ESS<T: RandomScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    rng: Rng,
    direction_scale: T,
    _provider: PhantomData<(T, B)>,
}

impl<T, B> ESS<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    /// Construct with an optional deterministic seed.
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::new, Rng::with_seed),
            direction_scale: T::one(),
            _provider: PhantomData,
        }
    }

    fn positive_uniform(&mut self) -> T {
        let mut value = T::random_unit(&mut self.rng);
        while value <= T::zero() {
            value = T::random_unit(&mut self.rng);
        }
        value
    }

    fn choose_move(&mut self, moves: &[(ESSMove<T>, T)]) -> ESSMove<T> {
        if moves.is_empty() {
            return ESSMove::Differential;
        }
        let total = moves
            .iter()
            .fold(T::zero(), |sum, (_, weight)| sum + *weight);
        let mut draw = T::random_unit(&mut self.rng) * total;
        for (proposal, weight) in moves {
            if draw < *weight {
                return *proposal;
            }
            draw = draw - *weight;
        }
        moves[moves.len() - 1].0
    }
}

impl<T, B> Default for ESS<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self::new(Some(0))
    }
}

impl<T, B, P, U, E> Algorithm<P, EnsembleStatus<T, B>, U, E> for ESS<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
    P: LogDensity<T, B, U, E>,
{
    type Summary = MCMCSummary<T, B>;
    type Config = ESSConfig<T, B>;
    type Init = ESSInit<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut EnsembleStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        status.walkers = init
            .walkers
            .iter()
            .map(|walker| transformed.to_internal(walker))
            .collect();
        status.log_density.clear();
        status.chain = vec![Vec::new(); init.walkers.len()];
        status.chain_storage = config.chain_storage;
        status.chain_steps = 0;
        self.direction_scale = config.direction_scale;
        for (index, walker) in status.walkers.iter().enumerate() {
            status
                .log_density
                .push(transformed.log_density(walker, args)?);
            status.evals.record_f();
            status.chain[index].push(transformed.to_external(walker));
        }
        status.message.initialize();
        Ok(())
    }

    fn step(
        &mut self,
        current_step: usize,
        problem: &P,
        status: &mut EnsembleStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        let snapshot = status.walkers.clone();
        let selected_move = self.choose_move(&config.moves);
        let global_model = if let ESSMove::Global { components, .. } = selected_move {
            let rows = snapshot.len();
            let columns = snapshot.first().map_or(0, Vector::len);
            let values = snapshot
                .iter()
                .flat_map(|walker| (0..walker.len()).map(|index| walker.get(index).to_f64()))
                .collect::<Option<Vec<_>>>();
            values.and_then(|values| {
                let data = nalgebra::DMatrix::from_row_slice(rows, columns, &values);
                dpgm::fit(components, &data, &mut self.rng)
            })
        } else {
            None
        };
        let mut expansion_steps = 0usize;
        let mut contraction_steps = 0usize;
        for walker_index in 0..snapshot.len() {
            let complement: Vec<usize> = (0..snapshot.len())
                .filter(|index| *index != walker_index)
                .collect();
            let dimension = snapshot[walker_index].len();
            let gaussian_direction = |rng: &mut Rng, scale: T| {
                let mean = complement
                    .iter()
                    .fold(Vector::zeros(dimension), |sum, index| {
                        sum.add(&snapshot[*index])
                    })
                    .scale(T::one() / T::literal(complement.len() as f64));
                complement
                    .iter()
                    .fold(Vector::zeros(dimension), |sum, index| {
                        sum.add_scaled(&snapshot[*index].sub(&mean), sample_standard_normal(rng))
                    })
                    .scale(scale)
            };
            let direction = match selected_move {
                ESSMove::Differential => {
                    if complement.len() < 2 {
                        snapshot[walker_index]
                            .sub(&snapshot[complement[0]])
                            .scale(self.direction_scale)
                    } else {
                        let first_position = self.rng.usize(0..complement.len());
                        let mut second_position = self.rng.usize(0..complement.len());
                        while second_position == first_position {
                            second_position = self.rng.usize(0..complement.len());
                        }
                        snapshot[complement[first_position]]
                            .sub(&snapshot[complement[second_position]])
                            .scale(self.direction_scale)
                    }
                }
                ESSMove::Gaussian => {
                    gaussian_direction(&mut self.rng, T::literal(2.0) * self.direction_scale)
                }
                ESSMove::Global {
                    scale,
                    rescale_covariance,
                    components: _,
                } => {
                    if let Some(model) = &global_model {
                        let first = model.labels[self.rng.usize(0..model.labels.len())];
                        let second = model.labels[self.rng.usize(0..model.labels.len())];
                        let sampled = if first == second {
                            dpgm::sample(
                                &model.means[first],
                                &model.covariances[first],
                                &mut self.rng,
                            )
                            .map(|sample| {
                                sample * (T::literal(2.0) * scale).to_f64().unwrap_or(2.0)
                            })
                        } else {
                            let rescale = rescale_covariance.to_f64().unwrap_or(0.001);
                            let first_sample = dpgm::sample(
                                &model.means[first],
                                &model.covariances[first].scale(rescale),
                                &mut self.rng,
                            );
                            let second_sample = dpgm::sample(
                                &model.means[second],
                                &model.covariances[second].scale(rescale),
                                &mut self.rng,
                            );
                            first_sample
                                .zip(second_sample)
                                .map(|(left, right)| (left - right) * 2.0)
                        };
                        sampled.map_or_else(
                            || gaussian_direction(&mut self.rng, T::literal(2.0) * scale),
                            |sample| {
                                Vector::from_vec(
                                    sample.iter().map(|value| T::literal(*value)).collect(),
                                )
                            },
                        )
                    } else {
                        gaussian_direction(&mut self.rng, T::literal(2.0) * scale)
                    }
                }
            };
            let slice_level = status.log_density[walker_index] + self.positive_uniform().ln();
            let offset = T::random_unit(&mut self.rng);
            let mut left = -offset * config.bracket_width;
            let mut right = left + config.bracket_width;
            while expansion_steps < config.max_shrink_steps {
                let proposal = snapshot[walker_index].add_scaled(&direction, left);
                let density = transformed.log_density(&proposal, args)?;
                status.evals.record_f();
                if density <= slice_level {
                    break;
                }
                left = left - config.bracket_width;
                expansion_steps += 1;
            }
            while expansion_steps < config.max_shrink_steps {
                let proposal = snapshot[walker_index].add_scaled(&direction, right);
                let density = transformed.log_density(&proposal, args)?;
                status.evals.record_f();
                if density <= slice_level {
                    break;
                }
                right = right + config.bracket_width;
                expansion_steps += 1;
            }
            let coordinate = loop {
                let coordinate = left + (right - left) * T::random_unit(&mut self.rng);
                let proposal = snapshot[walker_index].add_scaled(&direction, coordinate);
                let proposal_log_density = transformed.log_density(&proposal, args)?;
                status.evals.record_f();
                if proposal_log_density > slice_level
                    || contraction_steps >= config.max_shrink_steps
                {
                    break coordinate;
                }
                if coordinate < T::zero() {
                    left = coordinate;
                } else {
                    right = coordinate;
                }
                contraction_steps += 1;
            };
            let proposal = snapshot[walker_index].add_scaled(&direction, coordinate);
            let proposal_log_density = transformed.log_density(&proposal, args)?;
            status.evals.record_f();
            status.walkers[walker_index] = proposal;
            status.log_density[walker_index] = proposal_log_density;
        }
        if current_step <= config.adaptive_steps {
            let total = expansion_steps + contraction_steps;
            if total > 0 {
                self.direction_scale =
                    self.direction_scale * T::literal(2.0 * expansion_steps as f64 / total as f64);
            }
        }
        status.chain_steps += 1;
        let external = status
            .walkers
            .iter()
            .map(|walker| transformed.to_external(walker))
            .collect();
        status.retain_walkers(external);
        match selected_move {
            ESSMove::Differential => status.message.step_with_message("Differential Move"),
            ESSMove::Gaussian => status.message.step_with_message("Gaussian Move"),
            ESSMove::Global { .. } => status.message.step_with_message("Global Move"),
        }
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &EnsembleStatus<T, B>,
        _args: &U,
        _init: &Self::Init,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        let walkers = status.chain.len();
        let steps = status.chain.first().map_or(0, Vec::len);
        let variables = status
            .chain
            .first()
            .and_then(|walker| walker.first())
            .map_or(0, Vector::len);
        let mut message = status.message.clone();
        if matches!(message.status_type, crate::traits::StatusType::Custom)
            && message
                .text()
                .is_some_and(|text| text.contains("Maximum number of steps reached"))
        {
            let text = message.text_or_empty().to_string();
            message.succeed_with_message(text);
        }
        Ok(MCMCSummary {
            parameter_names: config.parameter_names.clone(),
            message,
            chain: status.chain.clone(),
            evals: status.evals,
            dimension: (walkers, steps, variables),
        })
    }

    fn reset(&mut self) {
        self.direction_scale = T::one();
    }

    fn default_callbacks() -> Callbacks<Self, P, EnsembleStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MaxSteps;
    use std::convert::Infallible;

    struct StandardNormal;

    impl<T, B> LogDensity<T, B> for StandardNormal
    where
        T: RandomScalar,
        B: LinearAlgebra<T>,
    {
        fn log_density(&self, x: &Vector<T, B>, _: &()) -> Result<T, Infallible> {
            Ok(-T::literal(0.5) * x.dot(x))
        }
    }

    #[test]
    fn ess_retains_provider_native_f32_chain() {
        let init: Vec<Vector<f32>> = (0..8)
            .map(|index| {
                Vector::from_vec(vec![
                    0.25_f32.mul_add(index as f32, -1.0),
                    (-0.1_f32).mul_add(index as f32, 0.5),
                ])
            })
            .collect();
        let mut sampler = ESS::<f32>::new(Some(29));
        let result = sampler
            .process(
                &StandardNormal,
                &(),
                ESSInit::new(init).unwrap(),
                ESSConfig::<f32>::default(),
                Callbacks::empty().with_terminator(MaxSteps(100)),
            )
            .unwrap();
        assert_eq!(result.dimension, (8, 101, 2));
        assert!(result.evals.f() >= 808);
    }

    #[test]
    fn ess_supports_weighted_moves_adaptation_and_sampled_storage() {
        let init: Vec<Vector> = (0..8)
            .map(|index| Vector::from_vec(vec![index as f64 * 0.1, index as f64 * -0.05]))
            .collect();
        let names = vec!["x".to_string(), "y".to_string()];
        let config = ESSConfig::default()
            .with_parameter_names(["x", "y"])
            .with_max_steps(100)
            .with_moves([
                ESSMove::differential(0.4),
                ESSMove::gaussian(0.4),
                ESSMove::global(0.2),
            ])
            .unwrap()
            .with_adaptive_steps(5)
            .with_chain_storage(ChainStorageMode::Sampled {
                keep_every: 2,
                max_samples: Some(4),
            });
        let result = ESS::<f64>::new(Some(11))
            .process(
                &StandardNormal,
                &(),
                ESSInit::new(init).unwrap(),
                config,
                Callbacks::empty().with_terminator(MaxSteps(20)),
            )
            .unwrap();
        assert_eq!(result.parameter_names, Some(names));
        assert!(result.chain.iter().all(|chain| chain.len() <= 4));
    }

    #[test]
    fn dpgm_recovers_two_separated_components() {
        let mut data = nalgebra::DMatrix::zeros(80, 2);
        let mut rng = Rng::with_seed(5);
        for row in 0..80 {
            let center = if row < 40 { -4.0 } else { 4.0 };
            data[(row, 0)] = center + 0.15 * sample_standard_normal::<f64>(&mut rng);
            data[(row, 1)] = center + 0.15 * sample_standard_normal::<f64>(&mut rng);
        }
        let model = dpgm::fit(2, &data, &mut Rng::with_seed(7)).unwrap();
        assert_eq!(model.labels.len(), 80);
        assert_eq!(model.means.len(), 2);
        assert_eq!(model.covariances.len(), 2);
        let mut centers = model.means.iter().map(|mean| mean[0]).collect::<Vec<_>>();
        centers.sort_by(f64::total_cmp);
        assert!(centers[0] < -3.0);
        assert!(centers[1] > 3.0);
    }

    #[test]
    fn ess_initialization_reports_invalid_ensembles() {
        assert!(ESSInit::<f64>::new(vec![[0.0].into(), [1.0].into()]).is_err());
        assert!(ESSInit::<f64>::new(vec![[0.0].into(), [1.0].into(), [0.0, 1.0].into(),]).is_err());
    }
}
