// https://arxiv.org/abs/2002.06212
use std::sync::Arc;

use fastrand::Rng;
use nalgebra::{Cholesky, DMatrix, DVector};
use parking_lot::RwLock;

use crate::{
    core::Point,
    traits::CostFunction,
    utils::{generate_random_vector_in_limits, RandChoice, SampleFloat},
    Float, PI,
};

use super::{Ensemble, MCMCAlgorithm};

/// A move used by the [`ESS`] algorithm
///
/// See Karamanis & Beutler[^1] for step implementation algorithms
///
/// [^1]: Karamanis, M., & Beutler, F. (2020). Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions. arXiv Preprint arXiv: 2002.06212.
#[derive(Copy, Clone)]
pub enum ESSMove {
    /// The Differential move described in Algorithm 2 of Karamanis & Beutler
    Differential,
    /// The Gaussian move described in Algorithm 3 of Karamanis & Beutler
    Gaussian,
    /// The Global move described in Algorithm 4 of Karamanis & Beutler
    Global {
        /// A scale factor that is applied if the walker jumps within its own cluster
        scale: Float,
        /// A rescaling factor applied to the covariance which promotes mode jumping
        rescale_cov: Float,
        /// The number of mixture coefficients
        n_components: usize,
    },
}
impl ESSMove {
    /// Create a new [`ESSMove::Differential`] with a usage weight
    pub const fn differential(weight: Float) -> WeightedESSMove {
        (Self::Differential, weight)
    }
    /// Create a new [`ESSMove::Gaussian`] with a usage weight
    pub const fn gaussian(weight: Float) -> WeightedESSMove {
        (Self::Gaussian, weight)
    }
    /// Create a new [`ESSMove::Global`] with a usage weight
    pub fn global(
        weight: Float,
        scale: Option<Float>,
        rescale_cov: Option<Float>,
        n_components: Option<usize>,
    ) -> WeightedESSMove {
        (
            Self::Global {
                scale: scale.unwrap_or(1.0),
                rescale_cov: rescale_cov.unwrap_or(0.001),
                n_components: n_components.unwrap_or(5),
            },
            weight,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn step<U, E>(
        &self,
        step: usize,
        n_adaptive: usize,
        max_steps: usize,
        mu: &mut Float,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        ensemble: &mut Ensemble,
        rng: &mut Rng,
    ) -> Result<(), E> {
        let mut positions = Vec::with_capacity(ensemble.len());
        let mut n_expand = 0;
        let mut n_contract = 0;
        let n = ensemble.walkers[0].get_latest().read().x.len();
        let mut dpgm_result = None;
        for (i, walker) in ensemble.iter().enumerate() {
            let x_k = walker.get_latest();
            let eta = match self {
                Self::Differential => {
                    // Given a walker Xₖ and complementary set of walkers S, pick two walkers Xₗ and Xₘ from S (without
                    // replacement) and compute direction vector ηₖ = μ(Xₗ - Xₘ)
                    let s = &ensemble.get_compliment_walkers(i, 2, rng);
                    let x_l = s[0].get_latest();
                    let x_m = s[1].get_latest();
                    let eta = (&x_l.read().x - &x_m.read().x).scale(*mu);
                    eta
                }
                Self::Gaussian => {
                    // Cₛ = 1/|S|   ⅀ (Xₗ - X̅ₛ)(Xₗ - X̅ₛ)†
                    //            Xₗ∈S
                    // sample ηₖ/(2μ) ∝ Norm(0, Cₛ)
                    //
                    // We can do this faster by selecting Zₗ ~ Norm(μ=0, σ=1) and
                    //
                    // W = ⅀ Zₗ(Xₗ - X̅ₛ)
                    //   Xₗ∈S
                    let x_s = ensemble.mean_compliment(i);
                    let n_s = ensemble.len();
                    ensemble
                        .iter_compliment(i)
                        .map(|x_l| (&x_l.read().x - &x_s).scale(rng.normal(0.0, 1.0)))
                        .sum::<DVector<Float>>()
                        .scale(2.0 * *mu)
                }
                Self::Global {
                    scale,
                    rescale_cov,
                    n_components,
                } => {
                    let dpgm =
                        dpgm_result.get_or_insert_with(|| dpgm(*n_components, ensemble, rng));
                    let labels = &dpgm.labels;
                    let means = &dpgm.means;
                    let covariances = &dpgm.covariances;
                    let indices = rng.choose_multiple(labels.iter(), 2);
                    let a = indices[0];
                    let b = indices[1];
                    // TODO: the multivariate sampling could be faster if the input was the
                    // Cholesky decomposition of the covariance matrix
                    if a == b {
                        rng.mv_normal(&means[*a], &covariances[*a])
                            .scale(2.0 * scale)
                    } else {
                        (rng.mv_normal(&means[*a], &covariances[*a].scale(*rescale_cov))
                            - rng.mv_normal(&means[*b], &covariances[*b].scale(*rescale_cov)))
                        .scale(2.0)
                    }
                }
            };
            // Y ~ U(0, f(Xₖ(t)))
            let y = x_k.read().fx_checked() + rng.float().ln();
            // U ~ U(0, 1)
            // L <- -U
            let mut l = -rng.float();
            let mut p_l = Point::from(&x_k.read().x + eta.scale(l));
            p_l.evaluate(func, user_data)?;
            // R <- L + 1
            let mut r = l + 1.0;
            let mut p_r = Point::from(&x_k.read().x + eta.scale(r));
            p_r.evaluate(func, user_data)?;
            // while Y < f(L) do
            while y < p_l.fx_checked() && n_expand < max_steps {
                // L <- L - 1
                l -= 1.0;
                p_l.set_position(&x_k.read().x + eta.scale(l));
                p_l.evaluate(func, user_data)?;
                // N₊(t) <- N₊(t) + 1
                n_expand += 1;
            }
            // while Y < f(R) do
            while y < p_r.fx_checked() && n_expand < max_steps {
                // R <- R + 1
                r += 1.0;
                p_r.set_position(&x_k.read().x + eta.scale(r));
                p_r.evaluate(func, user_data)?;
                // N₊(t) <- N₊(t) + 1
                n_expand += 1;
            }
            // while True do
            let xprime = loop {
                // X' ~ U(L, R)
                let xprime = rng.range(l, r);
                // Y' <- f(X'ηₖ + Xₖ(t))
                let mut p_yprime = Point::from(&x_k.read().x + eta.scale(xprime));
                p_yprime.evaluate(func, user_data)?;
                if y < p_yprime.fx_checked() || n_contract >= max_steps {
                    // if Y < Y' then break
                    break xprime;
                }
                if xprime < 0.0 {
                    // if X' < 0 then L <- X'
                    l = xprime;
                } else {
                    // else R <- X'
                    r = xprime;
                }
                // N₋(t) <- N₋(t) + 1
                n_contract += 1;
            };
            // Xₖ(t+1) <- X'ηₖ + Xₖ(t)
            let mut proposal = Point::from(&x_k.read().x + eta.scale(xprime));
            proposal.evaluate(func, user_data)?;
            positions.push(Arc::new(RwLock::new(proposal)))
        }
        // μ(t+1) <- TuneLengthScale(t, μ(t), N₊(t), N₋(t), M[adapt])
        if step <= n_adaptive {
            *mu *= 2.0 * (n_expand as Float) / (n_expand + n_contract) as Float
        }
        ensemble.push(positions);
        Ok(())
    }
}

/// The Ensemble Slice Sampler
///
/// This sampler follows Algorithm 5 in Karamanis & Beutler.[^1].
///
/// [^1]: Karamanis, M., & Beutler, F. (2020). Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions. arXiv Preprint arXiv: 2002.06212.
#[derive(Clone)]
pub struct ESS {
    rng: Rng,
    moves: Vec<WeightedESSMove>,
    n_adaptive: usize,
    max_steps: usize,
    mu: Float,
}

/// A [`ESSMove`] coupled with a weight
pub type WeightedESSMove = (ESSMove, Float);

impl ESS {
    /// Create a new Ensemble Slice Sampler from a list of weighted [`ESSMove`]s
    pub fn new<T: AsRef<[WeightedESSMove]>>(moves: T, rng: Rng) -> Self {
        Self {
            rng,
            moves: moves.as_ref().to_vec(),
            n_adaptive: 0,
            max_steps: 10000,
            mu: 1.0,
        }
    }
    /// Set the number of adaptive moves to perform at the start of sampling (default: `0`)
    pub const fn with_n_adaptive(mut self, n_adaptive: usize) -> Self {
        self.n_adaptive = n_adaptive;
        self
    }
    /// Set the maximum number of expansion/contractions to perform at each step (default: `10000`)
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
    /// Set the adaptive scaling parameter, $`\mu`$ (default: `1.0`)
    pub const fn with_mu(mut self, mu: Float) -> Self {
        self.mu = mu;
        self
    }
}

impl<U, E> MCMCAlgorithm<U, E> for ESS {
    fn initialize(
        &mut self,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        ensemble: &mut Ensemble,
    ) -> Result<(), E> {
        ensemble.evaluate_latest(func, user_data)?;
        Ok(())
    }
    fn step(
        &mut self,
        i_step: usize,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        ensemble: &mut Ensemble,
    ) -> Result<(), E> {
        let step_type_index = self
            .rng
            .choice_weighted(&self.moves.iter().map(|s| s.1).collect::<Vec<Float>>())
            .unwrap_or(0);
        let step_type = self.moves[step_type_index].0;
        step_type.step(
            i_step,
            self.n_adaptive,
            self.max_steps,
            &mut self.mu,
            func,
            user_data,
            ensemble,
            &mut self.rng,
        )?;
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        func: &dyn CostFunction<U, E>,
        user_data: &mut U,
        chains: &mut Ensemble,
    ) -> Result<bool, E> {
        Ok(false)
    }
}

// Calculate the k-means cluster of a set of points
//
// n_clusters: number of clusters
// data: (n_walkers, n_parameters)
//
// # Returns
//
// labels: Vec<usize> (n_walkers,)
#[allow(clippy::unwrap_used)]
fn kmeans(n_clusters: usize, data: &DMatrix<Float>, rng: &mut Rng) -> Vec<usize> {
    let n_walkers = data.nrows();
    let n_parameters = data.ncols();
    let limits = data
        .column_iter()
        .map(|col| (col.min(), col.max()))
        .collect::<Vec<_>>();
    let mut centroids: Vec<DVector<Float>> = (0..n_clusters)
        .map(|_| generate_random_vector_in_limits(&limits, rng))
        .collect();
    let mut labels = vec![0; n_walkers];
    for _ in 0..50 {
        for (i, walker) in data.row_iter().enumerate() {
            labels[i] = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (walker.transpose() - *a)
                        .norm_squared()
                        .partial_cmp(&(walker.transpose() - *b).norm_squared())
                        .unwrap()
                })
                .map(|(j, _)| j)
                .unwrap();
        }
        for (j, centroid) in centroids.iter_mut().enumerate() {
            let mut sum = DVector::zeros(n_parameters);
            let mut count = 0;
            for (l, w) in labels.iter().zip(data.row_iter()) {
                if *l == j {
                    sum += w.transpose();
                    count += 1;
                }
            }
            if count > 0 {
                sum /= count as Float;
            }
            *centroid = sum;
        }
    }
    labels
}

// Computes the covariance matrix of a given matrix
//
// m: (N, M)
//
// # Returns
//
// cov: (N, N)
fn cov(m: &DMatrix<Float>) -> DMatrix<Float> {
    let mean: DVector<Float> = m
        .row_iter()
        .map(|row| row.mean())
        .collect::<Vec<Float>>()
        .into();
    let centered = m.clone() - mean * DMatrix::from_element(1, m.ncols(), 1.0);
    &centered * centered.transpose() / (m.ncols() as Float - 1.0)
}

// data: (n_walkers, n_parameters)
// resp: (n_walkers, n_components)
// reg_covar: Float
//
// # Returns
//
// nk: (n_components,)
// means: (n_components, n_parameters)
// covariances: (n_components, (n_parameters, n_parameters))
fn estimate_gaussian_parameters(
    data: &DMatrix<Float>,
    resp: &DMatrix<Float>,
    reg_covar: Float,
) -> (DVector<Float>, DMatrix<Float>, Vec<DMatrix<Float>>) {
    assert_eq!(data.nrows(), resp.nrows());

    let nk = resp.row_sum_tr().add_scalar(10.0 * Float::EPSILON);
    let mut means: DMatrix<Float> = resp.transpose() * data;
    means.column_iter_mut().for_each(|mut c| {
        c.component_div_assign(&nk);
    });
    let cov = (0..means.nrows())
        .map(|k| {
            let mean_k = means.row(k);
            let diff =
                DMatrix::from_rows(&data.row_iter().map(|row| row - mean_k).collect::<Vec<_>>());
            let weighted_diff_t = DMatrix::from_columns(
                &diff
                    .row_iter()
                    .zip(resp.column(k).iter())
                    .map(|(d, &r)| d.scale(r).transpose())
                    .collect::<Vec<_>>(),
            );
            let mut cov = (&weighted_diff_t * &diff).unscale(nk[k]);
            for i in 0..data.ncols() {
                cov[(i, i)] += reg_covar;
            }
            cov
        })
        .collect();
    (nk, means, cov)
}

// nk: (n_components,)
//
// # Returns
//
// dirichlet_0: (n_components,)
// dirichlet_1: (n_components,)
fn estimate_weights(
    nk: &DVector<Float>,
    weight_concentration_prior: Float,
) -> (DVector<Float>, DVector<Float>) {
    let n_components = nk.len();
    (nk.map(|x| x + 1.0), {
        let reversed: Vec<Float> = nk.iter().rev().copied().collect();
        let mut cumulative_sum = vec![0.0; n_components];
        let mut sum: Float = 0.0;
        for (i, &val) in reversed.iter().enumerate() {
            sum += val;
            cumulative_sum[i] = sum;
        }
        let mut tail = cumulative_sum[..n_components - 1]
            .iter()
            .rev()
            .copied()
            .collect::<Vec<Float>>();
        tail.push(0.0);
        DVector::from_iterator(
            n_components,
            tail.into_iter().map(|x| x + weight_concentration_prior),
        )
    })
}

// nk: (n_components,)
// xk: (n_components, n_parameters)
// mean_prior: (n_parameters,)
//
// # Returns:
//
// mean_precision: (n_components,)
// means: (n_components, n_parameters)
fn estimate_means(
    nk: &DVector<Float>,
    xk: &DMatrix<Float>,
    mean_prior: &DVector<Float>,
    mean_precision_prior: Float,
) -> (DVector<Float>, DMatrix<Float>) {
    assert_eq!(nk.len(), xk.nrows());
    assert_eq!(mean_prior.len(), xk.ncols());
    let mean_precision = nk.map(|x| x + mean_precision_prior);
    let mut means = DMatrix::zeros(xk.nrows(), xk.ncols());
    let nkxk: DMatrix<Float> = DMatrix::from_columns(
        &xk.column_iter()
            .map(|x| x.component_mul(nk))
            .collect::<Vec<_>>(),
    );
    means.row_iter_mut().for_each(|mut row| {
        row += mean_prior.transpose().scale(mean_precision_prior);
    });
    means += nkxk;
    means.column_iter_mut().for_each(|mut col| {
        col.component_div_assign(&mean_precision);
    });
    (mean_precision, means)
}

// nk: (n_components,)
// xk: (n_components, n_parameters)
// sk: (n_components, (n_parameters, n_parameters))
//
// covariance_prior: (n_parameters, n_parameters)
// mean_prior: (n_parameters,)
// mean_precision: (n_components,)
//
// # Returns
//
// degrees_of_freedom: (n_components,)
// covariances: (n_components, (n_parameters, n_parameters))
// precisions_cholesky: (n_components, (n_parameters, n_parameters))
#[allow(clippy::too_many_arguments)]
fn estimate_precisions(
    nk: &DVector<Float>,
    xk: &DMatrix<Float>,
    sk: &[DMatrix<Float>],
    degrees_of_freedom_prior: Float,
    covariance_prior: &DMatrix<Float>,
    mean_prior: &DVector<Float>,
    mean_precision_prior: Float,
    mean_precision: &DVector<Float>,
) -> (DVector<Float>, Vec<DMatrix<Float>>, Vec<DMatrix<Float>>) {
    let n_components = nk.len();
    let n_parameters = mean_prior.len();

    assert_eq!(xk.nrows(), n_components);
    assert_eq!(xk.ncols(), n_parameters);
    assert_eq!(covariance_prior.nrows(), n_parameters);
    assert_eq!(covariance_prior.ncols(), n_parameters);
    assert_eq!(mean_precision.len(), n_components);

    let degrees_of_freedom = nk.map(|x| x + degrees_of_freedom_prior);

    let mut covariances = Vec::with_capacity(n_components);
    let mut precisions_cholesky = Vec::with_capacity(n_components);

    for k in 0..n_components {
        let nk_k = nk[k];
        let xk_k = xk.row(k).transpose();
        let sk_k = &sk[k];
        let mean_precision_k = mean_precision[k];
        let degrees_of_freedom_k = degrees_of_freedom[k];
        let diff = &xk_k - mean_prior;
        let outer = &diff * diff.transpose();
        let covariance = (covariance_prior
            + (sk_k * nk_k)
            + outer * (nk_k * mean_precision_prior / mean_precision_k))
            .unscale(degrees_of_freedom_k);
        covariances.push(covariance.clone());
        #[allow(clippy::expect_used)]
        let cholesky = Cholesky::new(covariance).expect("Cholesky decomposition failed");
        let l = cholesky.l();
        let id = DMatrix::identity(n_parameters, n_parameters);
        #[allow(clippy::expect_used)]
        let solved = l
            .solve_lower_triangular(&id)
            .expect("Colesky solve_lower_triangular failed");
        precisions_cholesky.push(solved.transpose());
    }
    (degrees_of_freedom, covariances, precisions_cholesky)
}

// precisions_cholesky: (n_components, (n_parameters, n_parameters))
//
// # Returns
//
// log_det_cholesky: (n_components,)
fn log_det_cholesky(precisions_cholesky: &[DMatrix<Float>], n_parameters: usize) -> DVector<Float> {
    DVector::from_iterator(
        precisions_cholesky.len(),
        precisions_cholesky
            .iter()
            .map(|chol| (0..n_parameters).map(|i| chol[(i, i)].ln()).sum()),
    )
}

// data: (n_walkers, n_parameters)
// means: (n_components, n_parameters)
// precisions_cholesky: (n_components, (n_parameters, n_parameters))
//
// # Returns
//
// log_prob: (n_walkers, n_components)
fn log_gaussian_prob(
    data: &DMatrix<Float>,
    means: &DMatrix<Float>,
    precisions_cholesky: &[DMatrix<Float>],
) -> DMatrix<Float> {
    let n_walkers = data.nrows();
    let n_parameters = data.ncols();
    let n_components = means.nrows();

    let log_det = log_det_cholesky(precisions_cholesky, n_parameters);
    let mut log_prob = DMatrix::zeros(n_walkers, n_components);
    for k in 0..n_components {
        let mu_k = means.row(k);
        let prec_chol_k = &precisions_cholesky[k];

        for i in 0..n_walkers {
            let x_i = data.row(i);
            let centered = x_i - mu_k;
            let y = &centered * prec_chol_k;
            let sq_sum = y.map(|val| val * val).sum();
            log_prob[(i, k)] = (-0.5 as Float).mul_add(
                (n_parameters as Float).mul_add(Float::ln(2.0 * PI), sq_sum),
                log_det[k],
            );
        }
    }
    log_prob
}

// data: (n_walkers, n_parameters)
// means: (n_components, n_parameters)
// precisions_cholesky: (n_components, (n_parameters, n_parameters))
//
// # Returns
//
// log_prob_norm: Float
// log_resp: (n_walkers, n_components)
fn e_step(
    data: &DMatrix<Float>,
    means: &DMatrix<Float>,
    precisions_cholesky: &[DMatrix<Float>],
    mean_precision: &DVector<Float>,
    degrees_of_freedom: &DVector<Float>,
    weight_concentration: &(DVector<Float>, DVector<Float>),
) -> (Float, DMatrix<Float>) {
    let n_walkers = data.nrows();
    let n_parameters = data.ncols();
    let n_components = means.nrows();
    let estimated_log_prob = {
        let mut log_gauss = log_gaussian_prob(data, means, precisions_cholesky);
        log_gauss.row_iter_mut().for_each(|mut row| {
            row -= degrees_of_freedom
                .map(|x| 0.5 * (n_parameters as Float) * x.ln())
                .transpose()
        });
        let log_lambda = {
            let mut res: DVector<Float> = DVector::zeros(n_components);
            for j in 0..n_parameters {
                for k in 0..n_components {
                    res[k] += spec_math::Gamma::digamma(
                        &((0.5 * (degrees_of_freedom[k] - j as Float)) as f64),
                    ) as Float
                }
            }
            res.map(|r| (n_parameters as Float).mul_add(Float::ln(2.0), r))
        };
        log_gauss.row_iter_mut().for_each(|mut row| {
            row += (0.5 * (&log_lambda - mean_precision.map(|mu| n_parameters as Float / mu)))
                .transpose()
        });
        log_gauss
    };
    let estimated_log_weights = {
        let a = &weight_concentration.0;
        let b = &weight_concentration.1;
        let n = a.len();
        let digamma_sum = (a + b).map(|v| spec_math::Gamma::digamma(&(v as f64)) as Float);
        let digamma_a = a.map(|v| spec_math::Gamma::digamma(&(v as f64)) as Float);
        let digamma_b = b.map(|v| spec_math::Gamma::digamma(&(v as f64)) as Float);
        let mut cumulative = Vec::with_capacity(n);
        let mut acc = 0.0;
        cumulative.push(0.0);
        for i in 0..n - 1 {
            acc += digamma_b[i] - digamma_sum[i];
            cumulative.push(acc);
        }
        DVector::from_iterator(
            n,
            (0..n).map(|i| digamma_a[i] - digamma_sum[i] + cumulative[i]),
        )
    };
    let mut weighted_log_prob = estimated_log_prob;
    weighted_log_prob
        .row_iter_mut()
        .for_each(|mut row| row += &estimated_log_weights.transpose());
    let log_prob_norm = DVector::from_iterator(
        n_walkers,
        weighted_log_prob
            .row_iter()
            .map(|row| logsumexp::LogSumExp::ln_sum_exp(row.iter())),
    );
    let mut log_resp = weighted_log_prob;
    log_resp
        .column_iter_mut()
        .for_each(|mut col| col -= &log_prob_norm);
    (log_prob_norm.mean(), log_resp)
}

#[derive(Clone)]
struct DPGMResult {
    // labels: (n_walkers,)
    labels: Vec<usize>,
    // means: (n_components, (n_parameters,))
    means: Vec<DVector<Float>>,
    // covariances: (n_components, (n_parameters, n_parameters))
    covariances: Vec<DMatrix<Float>>,
}

// Dirichlet Process Gaussian Mixture
//
// Code is taken almost verbatim (converting numpy to nalgebra) from
// <https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_bayesian_mixture.py#L74>
// with some modifications to only use the "full"" covariance mode, the "kmeans" initialization
// method, and the "dirichlet_process" weight concentration prior. See the readme/crate
// documentation for the proper citation.
//
// n_components: usize, the number of Gaussian mixture components
// ensemble: &Ensemble
//
// # Returns
//
// DPGMResult
fn dpgm(n_components: usize, ensemble: &Ensemble, rng: &mut Rng) -> DPGMResult {
    let (n_walkers, _, n_parameters) = ensemble.dimension();
    let data = ensemble.get_latest_position_matrix();
    let weight_concentration_prior = 1.0 / n_components as Float;
    let mean_precision_prior = 1.0;
    let mean_prior = ensemble.mean();
    let degrees_of_freedom_prior = n_parameters as Float;
    let covariance_prior = cov(&data.transpose());

    let mut resp: DMatrix<Float> = DMatrix::zeros(n_walkers, n_components);
    let labels = kmeans(n_components, &data, rng);
    for (i, &cluster_id) in labels.iter().enumerate() {
        resp[(i, cluster_id)] = 1.0;
    }
    let (mut nk, mut xk, mut sk) = estimate_gaussian_parameters(&data, &resp, 1e-6);
    let mut weight_concentration = estimate_weights(&nk, weight_concentration_prior);
    let (mut mean_precision, mut means) =
        estimate_means(&nk, &xk, &mean_prior, mean_precision_prior);
    let (mut degrees_of_freedom, mut covariances, mut precisions_cholesky) = estimate_precisions(
        &nk,
        &xk,
        &sk,
        degrees_of_freedom_prior,
        &covariance_prior,
        &mean_prior,
        mean_precision_prior,
        &mean_precision,
    );
    let mut lower_bound = Float::NEG_INFINITY;
    for iiter in 1..=100 {
        let prev_lower_bound = lower_bound;
        let (log_prob_norm, log_resp) = e_step(
            &data,
            &means,
            &precisions_cholesky,
            &mean_precision,
            &degrees_of_freedom,
            &weight_concentration,
        );
        (nk, xk, sk) = estimate_gaussian_parameters(&data, &log_resp.map(Float::exp), 1e-6);
        weight_concentration = estimate_weights(&nk, weight_concentration_prior);
        (mean_precision, means) = estimate_means(&nk, &xk, &mean_prior, mean_precision_prior);
        (degrees_of_freedom, covariances, precisions_cholesky) = estimate_precisions(
            &nk,
            &xk,
            &sk,
            degrees_of_freedom_prior,
            &covariance_prior,
            &mean_prior,
            mean_precision_prior,
            &mean_precision,
        );
        lower_bound = {
            let log_det_precisions_cholesky = log_det_cholesky(&precisions_cholesky, n_parameters)
                - degrees_of_freedom
                    .map(Float::ln)
                    .scale(0.5 * n_parameters as Float);
            let log_wishart_norm = {
                let mut log_wishart_norm =
                    degrees_of_freedom.component_mul(&log_det_precisions_cholesky);
                log_wishart_norm +=
                    degrees_of_freedom.scale(0.5 * Float::ln(2.0) * n_parameters as Float);

                let gammaln_term: DVector<Float> = degrees_of_freedom.map(|dof| {
                    (0..n_parameters)
                        .map(|i| {
                            spec_math::Gamma::lgamma(&((0.5 * (dof - i as Float)) as f64)) as Float
                        })
                        .sum()
                });
                log_wishart_norm += gammaln_term;
                -log_wishart_norm
            };
            let log_norm_weight = -((0..weight_concentration.0.len())
                .map(|i| {
                    spec_math::Beta::lbeta(
                        &(weight_concentration.0[i] as f64),
                        weight_concentration.1[i] as f64,
                    )
                })
                .sum::<f64>()) as Float;
            (0.5 * (n_parameters as Float)).mul_add(
                -mean_precision.map(|mp| mp.ln()).sum(),
                -log_resp.map(|lr| lr.exp() * lr).sum() - log_wishart_norm.sum(),
            ) - log_norm_weight
        };
        let change = lower_bound - prev_lower_bound;
        if change.abs() < 1e-3 {
            break;
        }
    }
    let weight_dirichlet_sum = &weight_concentration.0 + &weight_concentration.1;
    let tmp0 = &weight_concentration.0.component_div(&weight_dirichlet_sum);
    let tmp1 = &weight_concentration.1.component_div(&weight_dirichlet_sum);
    let mut prod_vec = Vec::with_capacity(n_components);
    prod_vec.push(1.0);
    for i in 0..(n_components - 1) {
        prod_vec.push(prod_vec[i] * tmp1[i])
    }
    let mut weights = tmp0.component_mul(&DVector::from_vec(prod_vec));
    weights /= weights.sum();
    let precisions: Vec<DMatrix<Float>> = (0..n_components)
        .map(|k| &precisions_cholesky[k] * precisions_cholesky[k].transpose())
        .collect();
    let (_, log_resp) = e_step(
        &data,
        &means,
        &precisions_cholesky,
        &mean_precision,
        &degrees_of_freedom,
        &weight_concentration,
    );
    DPGMResult {
        labels: log_resp
            .row_iter()
            .map(|row| row.transpose().argmax().0)
            .collect(),
        means: means
            .row_iter()
            .map(|row| row.transpose())
            .collect::<Vec<DVector<Float>>>(),
        covariances,
    }
}
