use crate::{DVector, Float};
use nalgebra::Complex;
use rustfft::FftPlanner;
use serde::{Deserialize, Serialize};

/// Calculate the integrated autocorrelation time for each parameter according to Karamanis &
/// Beutler[^Karamanis]
///
/// `samples` should have the shape `(n_walkers, n_steps, n_parameters)`.
///
/// `c` is an optional window size (`7.0` if [`None`] provided), see Sokal[^Sokal].
///
/// This is a standalone function that can be used to bypass the ensemble types and calculate IATs
/// for custom inputs.
///
/// [^Karamanis]: Karamanis, M., & Beutler, F. (2020). Ensemble slice sampling: Parallel, black-box and gradient-free inference for correlated & multimodal distributions. arXiv Preprint arXiv: 2002.06212.
/// [^Sokal]: Sokal, A. (1997). Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms. In C. DeWitt-Morette, P. Cartier, & A. Folacci (Eds.), Functional Integration: Basics and Applications (pp. 131–192). doi:10.1007/978-1-4899-0319-8_6
pub fn integrated_autocorrelation_times(
    samples: Vec<Vec<DVector<Float>>>,
    c: Option<Float>,
) -> DVector<Float> {
    let c = c.unwrap_or(7.0);
    let n_parameters = samples[0][0].len();
    let samples: Vec<DVector<Float>> = samples.into_iter().flatten().collect();
    let mut n = 1usize;
    while n < samples.len() {
        n <<= 1;
    }
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(2 * n);
    let ifft = planner.plan_fft_inverse(2 * n);
    DVector::from_iterator(
        n_parameters,
        (0..n_parameters).map(|i_parameter| {
            let x: Vec<Float> = samples.iter().map(|sample| sample[i_parameter]).collect();
            let mean = x.iter().sum::<Float>() / x.len() as Float;
            let mut input: Vec<Complex<Float>> =
                x.iter().map(|&val| Complex::new(val - mean, 0.0)).collect();
            input.resize(2 * n, Complex::new(0.0, 0.0));

            fft.process(&mut input);

            for val in &mut input {
                *val *= val.conj();
            }

            ifft.process(&mut input);

            let mut acf: Vec<Float> = input
                .iter()
                .take(x.len())
                .map(|value| value.re / (4.0 * n as Float))
                .collect();

            if !acf.is_empty() && acf[0] != 0.0 {
                let norm_factor = acf[0];
                acf.iter_mut().for_each(|value| *value /= norm_factor);
            }

            let taus: Vec<Float> = acf
                .iter()
                .scan(0.0, |acc, &value| {
                    *acc += value;
                    Some(*acc)
                })
                .map(|value| Float::mul_add(2.0, value, -1.0))
                .collect();
            let ind = taus
                .iter()
                .enumerate()
                .position(|(idx, &tau)| (idx as Float) >= c * tau)
                .unwrap_or(taus.len() - 1);
            taus[ind]
        }),
    )
}

/// Diagnostics computed from retained MCMC chains.
///
/// Split-`R-hat` follows the modern recommendation in Vehtari et al.[^Vehtari].
///
/// [^Vehtari]: Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. Bayesian Analysis, 16(2), 667-718. https://doi.org/10.1214/20-BA1221
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MCMCDiagnostics {
    /// Split-`R-hat` for each parameter.
    pub r_hat: DVector<Float>,
    /// Effective sample size estimate for each parameter.
    pub ess: DVector<Float>,
    /// Acceptance rate inferred from retained transitions for each walker.
    pub acceptance_rates: DVector<Float>,
    /// Mean acceptance rate across walkers.
    pub mean_acceptance_rate: Float,
}

fn split_chains(chains: &[Vec<DVector<Float>>]) -> Vec<Vec<DVector<Float>>> {
    let mut split = Vec::new();
    for chain in chains {
        let half = chain.len() / 2;
        if half >= 2 {
            split.push(chain[..half].to_vec());
            split.push(chain[chain.len() - half..].to_vec());
        } else if chain.len() >= 2 {
            split.push(chain.clone());
        }
    }
    split
}

fn sample_variance(values: &[Float]) -> Float {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = values.iter().sum::<Float>() / values.len() as Float;
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<Float>()
        / (values.len() - 1) as Float
}

pub(crate) fn split_r_hat(chains: &[Vec<DVector<Float>>]) -> DVector<Float> {
    let split = split_chains(chains);
    if split.is_empty() || split[0].is_empty() {
        return DVector::zeros(0);
    }
    let n_chains = split.len();
    let n_samples = split[0].len();
    let n_params = split[0][0].len();

    DVector::from_iterator(
        n_params,
        (0..n_params).map(|param| {
            let means: Vec<Float> = split
                .iter()
                .map(|chain| {
                    chain.iter().map(|sample| sample[param]).sum::<Float>() / n_samples as Float
                })
                .collect();
            let variances: Vec<Float> = split
                .iter()
                .map(|chain| {
                    let vals: Vec<Float> = chain.iter().map(|sample| sample[param]).collect();
                    sample_variance(&vals)
                })
                .collect();
            let w = variances.iter().sum::<Float>() / n_chains as Float;
            let b = if n_chains > 1 {
                n_samples as Float * sample_variance(&means)
            } else {
                0.0
            };
            if w <= Float::EPSILON {
                1.0
            } else {
                let var_hat =
                    ((n_samples - 1) as Float / n_samples as Float) * w + b / n_samples as Float;
                (var_hat / w).sqrt().max(1.0)
            }
        }),
    )
}

pub(crate) fn effective_sample_size(chains: &[Vec<DVector<Float>>]) -> DVector<Float> {
    if chains.is_empty() || chains[0].is_empty() {
        return DVector::zeros(0);
    }
    let total_samples = chains.iter().map(Vec::len).sum::<usize>() as Float;
    integrated_autocorrelation_times(chains.to_vec(), None).map(|tau| {
        if !tau.is_finite() || tau <= 0.0 {
            total_samples
        } else {
            total_samples / tau
        }
    })
}

pub(crate) fn acceptance_rates(chains: &[Vec<DVector<Float>>]) -> DVector<Float> {
    DVector::from_iterator(
        chains.len(),
        chains.iter().map(|chain| {
            if chain.len() <= 1 {
                return 0.0;
            }
            let accepted = chain.windows(2).filter(|pair| pair[0] != pair[1]).count();
            accepted as Float / (chain.len() - 1) as Float
        }),
    )
}

pub(crate) fn diagnostics_from_chain(chains: &[Vec<DVector<Float>>]) -> MCMCDiagnostics {
    let acceptance_rates = acceptance_rates(chains);
    let mean_acceptance_rate = if acceptance_rates.is_empty() {
        0.0
    } else {
        acceptance_rates.iter().sum::<Float>() / acceptance_rates.len() as Float
    };
    MCMCDiagnostics {
        r_hat: split_r_hat(chains),
        ess: effective_sample_size(chains),
        acceptance_rates,
        mean_acceptance_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    #[test]
    fn test_split_r_hat_is_one_for_identical_split_chains() {
        let chains = vec![
            vec![dvector![0.0], dvector![1.0], dvector![0.0], dvector![1.0]],
            vec![dvector![0.0], dvector![1.0], dvector![0.0], dvector![1.0]],
        ];
        let r_hat = split_r_hat(&chains);
        assert_eq!(r_hat.len(), 1);
        assert!((r_hat[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_acceptance_rates_detect_repeated_samples() {
        let chains = vec![
            vec![dvector![0.0], dvector![0.0], dvector![1.0], dvector![1.0]],
            vec![dvector![0.0], dvector![1.0], dvector![2.0], dvector![3.0]],
        ];
        let rates = acceptance_rates(&chains);
        assert!((rates[0] - (1.0 / 3.0)).abs() < 1e-12);
        assert!((rates[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_effective_sample_size_is_positive() {
        let chains = vec![
            vec![dvector![0.0], dvector![1.0], dvector![0.0], dvector![1.0]],
            vec![dvector![1.0], dvector![0.0], dvector![1.0], dvector![0.0]],
        ];
        let ess = effective_sample_size(&chains);
        assert_eq!(ess.len(), 1);
        assert!(ess[0] > 0.0);
    }
}
