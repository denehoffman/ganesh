use crate::{DMatrix, DVector, Float};
use fastrand::Rng;
use fastrand_contrib::RngExt;
use nalgebra::Cholesky;
use parking_lot::Once;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) fn generate_random_vector(
    dimension: usize,
    lb: Float,
    ub: Float,
    rng: &mut Rng,
) -> DVector<Float> {
    DVector::from_vec((0..dimension).map(|_| rng.range(lb, ub)).collect())
}
pub(crate) fn generate_random_vector_in_limits(
    limits: &[(Float, Float)],
    rng: &mut Rng,
) -> DVector<Float> {
    DVector::from_vec(
        (0..limits.len())
            .map(|i| rng.range(limits[i].0, limits[i].1))
            .collect(),
    )
}

/// Computes the covariance matrix using a given hessian matrix.
pub fn hessian_to_covariance(hessian: &DMatrix<Float>) -> Option<DMatrix<Float>> {
    hessian.clone().try_inverse().or_else(|| {
        hessian
            .clone()
            .pseudo_inverse(Float::cbrt(Float::EPSILON))
            .ok()
    })
}

/// A helper trait to provide a weighted random choice method
pub trait RandChoice {
    /// Return an random index sampled with the given weights
    fn choice_weighted(&mut self, weights: &[Float]) -> Option<usize>;
}

impl RandChoice for Rng {
    fn choice_weighted(&mut self, weights: &[Float]) -> Option<usize> {
        let total_weight = weights.iter().sum();
        let u: Float = self.range(0.0, total_weight);
        let mut cumulative_weight = 0.0;
        for (index, &weight) in weights.iter().enumerate() {
            cumulative_weight += weight;
            if u <= cumulative_weight {
                return Some(index);
            }
        }
        None
    }
}

/// A helper trait to get feature-gated floating-point random values
pub trait SampleFloat {
    /// Get a random value in a range
    fn range(&mut self, lower: Float, upper: Float) -> Float;
    /// Get a random value in the range [0, 1]
    fn float(&mut self) -> Float;
    /// Get a random Normal value
    fn normal(&mut self, mu: Float, sigma: Float) -> Float;
    /// Get a random value from a multivariate Normal distribution
    #[allow(clippy::expect_used)]
    fn mv_normal(&mut self, mu: &DVector<Float>, cov: &DMatrix<Float>) -> DVector<Float> {
        let cholesky = Cholesky::new(cov.clone()).expect("Covariance matrix not positive definite");
        let a = cholesky.l();
        let z = DVector::from_iterator(mu.len(), (0..mu.len()).map(|_| self.normal(0.0, 1.0)));
        mu + a * z
    }
}
impl SampleFloat for Rng {
    #[cfg(not(feature = "f32"))]
    fn range(&mut self, lower: Float, upper: Float) -> Float {
        self.f64_range(lower..upper)
    }
    #[cfg(feature = "f32")]
    fn range(&mut self, lower: Float, upper: Float) -> Float {
        self.f32_range(lower..upper)
    }
    #[cfg(not(feature = "f32"))]
    fn float(&mut self) -> Float {
        self.f64()
    }
    #[cfg(feature = "f32")]
    fn float(&mut self) -> Float {
        self.f32()
    }
    #[cfg(not(feature = "f32"))]
    fn normal(&mut self, mu: Float, sigma: Float) -> Float {
        self.f64_normal(mu, sigma)
    }
    #[cfg(feature = "f32")]
    fn normal(&mut self, mu: Float, sigma: Float) -> Float {
        self.f32_normal(mu, sigma)
    }
}

static WARNINGS_ENABLED: AtomicBool = AtomicBool::new(true);
static WARNINGS_SET_BY_ENV: AtomicBool = AtomicBool::new(false);
static WARNINGS_OVERRIDE: AtomicBool = AtomicBool::new(false);
static INIT: Once = Once::new();

fn init_env_override() {
    INIT.call_once(|| {
        if let Ok(val) = std::env::var("GANESH_WARNINGS") {
            if val == "0" {
                WARNINGS_SET_BY_ENV.store(true, Ordering::Relaxed);
                WARNINGS_ENABLED.store(false, Ordering::Relaxed);
            }
            if val == "1" {
                WARNINGS_SET_BY_ENV.store(true, Ordering::Relaxed);
                WARNINGS_ENABLED.store(true, Ordering::Relaxed);
            }
        }
    });
}

fn try_set_warnings_override(value: bool) {
    init_env_override();
    if WARNINGS_SET_BY_ENV.load(Ordering::Relaxed) {
        return;
    }
    let already_set = WARNINGS_OVERRIDE.swap(true, Ordering::Relaxed);
    if !already_set {
        WARNINGS_ENABLED.store(value, Ordering::Relaxed);
    }
}

/// A method which can force-enable warnings which may be disabled by dependencies.
///
/// This method will still not enable warnings if the environment variable `GANESH_WARNINGS=0`.
pub fn enable_warnings() {
    try_set_warnings_override(true);
}

/// A method which can force-disable warnings which may be enabled by dependencies.
///
/// This method will still not disable warnings if the environment variable `GANESH_WARNINGS=1`.
pub fn disable_warnings() {
    try_set_warnings_override(false);
}

/// Returns `true` if warnings are enabled.
///
/// Warnings are enabled by default and can be disabled either by setting the environment variable
/// `GANESH_WARNINGS=0` or by calling [`disable_warnings`] first. The first call of
/// [`enable_warnings`] will ensure warnings are enabled, overriding any subsequent calls to
/// [`disable_warnings`]. Setting `GANESH_WARNINGS=1` will force-enable warnings regardless of any
/// calls to [`disable_warnings`]. In all cases, the environment variable takes precedence.
pub fn should_warn() -> bool {
    init_env_override();
    WARNINGS_ENABLED.load(Ordering::Relaxed)
}

/// Conditionally warns the user (warns by default).
///
/// See [`should_warn`] for details on how to conditionally enable and disable warnings.
pub fn maybe_warn(msg: &str) {
    if should_warn() {
        eprintln!("Warning: {msg}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastrand::Rng;

    #[test]
    fn test_pseudo_inverse() {
        let hessian = DMatrix::<Float>::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let cov = hessian_to_covariance(&hessian);
        assert!(cov.is_some());
        let cov = cov.unwrap();
        let expected = hessian.pseudo_inverse(Float::cbrt(Float::EPSILON)).unwrap();
        assert!(cov.relative_eq(&expected, Float::EPSILON, Float::EPSILON));
    }

    #[test]
    fn test_single_weight() {
        let mut rng = Rng::with_seed(0);
        let weights = vec![1.0];
        assert_eq!(rng.choice_weighted(&weights), Some(0));
    }

    #[test]
    fn test_two_equal_weights_deterministic() {
        let mut rng = Rng::with_seed(0);
        let weights = vec![1.0, 1.0];
        let first = rng.choice_weighted(&weights);
        let second = rng.choice_weighted(&weights);
        assert_eq!(first, Some(1));
        assert_eq!(second, Some(0));
    }

    #[test]
    fn test_weighted_three_choices_deterministic() {
        let mut rng = Rng::with_seed(0);
        let weights = vec![1.0, 2.0, 3.0];

        let first = rng.choice_weighted(&weights);
        let second = rng.choice_weighted(&weights);
        let third = rng.choice_weighted(&weights);

        assert_eq!(first, Some(2));
        assert_eq!(second, Some(0));
        assert_eq!(third, Some(0));
    }

    #[test]
    fn test_large_number_of_trials_reproducible_distribution() {
        let mut rng = Rng::with_seed(0);
        let weights = vec![1.0, 2.0, 3.0];
        let mut counts = [0; 3];
        for _ in 0..10_000 {
            counts[rng.choice_weighted(&weights).unwrap()] += 1;
        }
        assert_eq!(counts, [1705, 3244, 5051]);
    }

    #[test]
    fn test_empty_weights() {
        let mut rng = Rng::with_seed(0);
        let weights: Vec<Float> = vec![];
        assert_eq!(rng.choice_weighted(&weights), None);
    }

    #[test]
    fn test_zero_weights() {
        let mut rng = Rng::with_seed(0);
        let weights = vec![0.0, 0.0, 0.0];
        assert_eq!(rng.choice_weighted(&weights), Some(0));
    }

    #[test]
    fn test_output_dimension_matches_mu() {
        let mut rng = Rng::with_seed(0);
        let mu = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let cov = DMatrix::identity(3, 3);
        let sample = rng.mv_normal(&mu, &cov);
        assert_eq!(sample.len(), mu.len());
    }

    #[test]
    fn test_identity_covariance_zero_mean_is_standard_normal() {
        let mut rng = Rng::with_seed(0);
        let mu = DVector::from_vec(vec![0.0, 0.0]);
        let cov = DMatrix::identity(2, 2);

        let sample1 = rng.mv_normal(&mu, &cov);
        let sample2 = rng.mv_normal(&mu, &cov);

        assert_eq!(
            sample1,
            DVector::from_vec(vec![1.0059485396074146, -0.7239261169514642])
        );
        assert_eq!(
            sample2,
            DVector::from_vec(vec![-0.7517197959276235, -0.48053731558299817])
        );
    }

    #[test]
    fn test_mean_shift_applied() {
        let mut rng = Rng::with_seed(0);
        let mu = DVector::from_vec(vec![10.0, -5.0]);
        let cov = DMatrix::identity(2, 2);

        let sample = rng.mv_normal(&mu, &cov);
        assert_eq!(
            sample,
            DVector::from_vec(vec![11.005948539607415, -5.723926116951464])
        );
    }

    #[test]
    fn test_empirical_covariance_matches_target() {
        let mut rng = Rng::with_seed(0);
        let mu = DVector::from_vec(vec![0.0, 0.0]);
        let cov = DMatrix::from_row_slice(2, 2, &[1.0, 0.8, 0.8, 1.0]);

        let n_samples = 50_000;
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            samples.push(rng.mv_normal(&mu, &cov));
        }
        let mean: DVector<Float> = samples.iter().sum::<DVector<Float>>() / (n_samples as Float);
        let mut emp_cov = DMatrix::<Float>::zeros(2, 2);
        for x in &samples {
            let diff = x - &mean;
            emp_cov += &diff * diff.transpose();
        }
        emp_cov /= n_samples as Float;
        for i in 0..2 {
            for j in 0..2 {
                assert!((emp_cov[(i, j)] - cov[(i, j)]).abs() < 0.05);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Covariance matrix not positive definite")]
    fn test_non_positive_definite_triggers_expect() {
        let mut rng = Rng::with_seed(42);
        let mu = DVector::from_vec(vec![0.0, 0.0]);
        let cov = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]);
        let _ = rng.mv_normal(&mu, &cov);
    }

    fn reset_globals() {
        WARNINGS_ENABLED.store(true, Ordering::Relaxed);
        WARNINGS_SET_BY_ENV.store(false, Ordering::Relaxed);
        WARNINGS_OVERRIDE.store(false, Ordering::Relaxed);
    }

    #[test]
    fn test_default_should_warn_and_overrides() {
        reset_globals();
        assert!(should_warn());
        disable_warnings();
        assert!(!should_warn());
        enable_warnings();
        // this mimics a dependency trying to enable warnings after a user manually disables them
        assert!(!should_warn());

        reset_globals();
        enable_warnings();
        assert!(should_warn());
        disable_warnings();
        // this mimics a dependency trying to disable warnings after a user manually enables them
        assert!(should_warn());
    }

    // TODO: figure out how to get these tests to work with code coverage
    //
    // use std::env;
    //
    // #[test]
    // fn test_env_var_respected_disable() {
    //     reset_globals();
    //     env::set_var("GANESH_WARNINGS", "0");
    //     enable_warnings();
    //     assert!(!should_warn());
    //     env::remove_var("GANESH_WARNINGS");
    // }
    //
    // #[test]
    // fn test_env_var_respected_enable() {
    //     reset_globals();
    //     env::set_var("GANESH_WARNINGS", "1");
    //     disable_warnings();
    //     assert!(should_warn());
    //     env::remove_var("GANESH_WARNINGS");
    // }

    #[test]
    fn test_maybe_warn_branches() {
        reset_globals();
        maybe_warn("this should print");
        assert!(should_warn());

        reset_globals();
        disable_warnings();
        maybe_warn("this should not print");
        assert!(!should_warn());
    }
}
