use crate::core::{
    LinearAlgebra, LinearSolve, Matrix, PseudoInverse, RandomScalar, RealScalar, Vector,
};
use fastrand::Rng;
use parking_lot::Once;
use std::sync::atomic::{AtomicBool, Ordering};

/// Sample uniformly from `[lower, upper)` using a generic scalar representation.
pub fn sample_uniform<T: RandomScalar>(lower: T, upper: T, rng: &mut Rng) -> T {
    lower + (upper - lower) * T::random_unit(rng)
}

/// Generate a native vector with independent uniform entries.
pub fn generate_random_vector<T, B>(
    dimension: usize,
    lower: T,
    upper: T,
    rng: &mut Rng,
) -> Vector<T, B>
where
    T: RandomScalar,
    B: LinearAlgebra<T>,
{
    Vector::from_vec(
        (0..dimension)
            .map(|_| sample_uniform(lower, upper, rng))
            .collect(),
    )
}

/// Sample a standard normal variate using the Box-Muller transform.
pub fn sample_standard_normal<T: RandomScalar>(rng: &mut Rng) -> T {
    let mut u1 = T::random_unit(rng);
    while u1 <= T::zero() {
        u1 = T::random_unit(rng);
    }
    let u2 = T::random_unit(rng);
    (-T::literal(2.0) * u1.ln()).sqrt() * (T::literal(2.0 * std::f64::consts::PI) * u2).cos()
}

/// Choose an index proportionally to nonnegative generic scalar weights.
pub fn weighted_choice<T: RandomScalar>(weights: &[T], rng: &mut Rng) -> Option<usize> {
    let total = weights
        .iter()
        .copied()
        .fold(T::zero(), |sum, value| sum + value);
    if !total.is_finite() || total <= T::zero() {
        return None;
    }
    let target = T::random_unit(rng) * total;
    let mut cumulative = T::zero();
    for (index, weight) in weights.iter().copied().enumerate() {
        if weight < T::zero() || !weight.is_finite() {
            return None;
        }
        cumulative = cumulative + weight;
        if target <= cumulative {
            return Some(index);
        }
    }
    weights.len().checked_sub(1)
}

/// Convert a native Hessian to covariance using available solve capabilities.
pub fn hessian_to_covariance<T, B>(hessian: &Matrix<T, B>) -> Option<Matrix<T, B>>
where
    T: RealScalar,
    B: LinearSolve<T> + PseudoInverse<T>,
{
    hessian
        .lu_inverse()
        .or_else(|| hessian.pseudo_inverse(T::epsilon().cbrt()))
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
    use crate::core::{Matrix, NalgebraProvider};
    use fastrand::Rng;

    #[test]
    fn provider_generic_sampling_and_covariance_support_f32() {
        let mut rng = Rng::with_seed(11);
        let vector = generate_random_vector::<f32, NalgebraProvider>(8, -2.0, 3.0, &mut rng);
        assert!(vector
            .to_vec()
            .iter()
            .all(|value| (-2.0..3.0).contains(value)));
        let normal = sample_standard_normal::<f32>(&mut rng);
        assert!(normal.is_finite());
        assert_eq!(weighted_choice(&[0.0_f32, 1.0], &mut rng), Some(1));

        let hessian = Matrix::<f32>::identity(2).scale(2.0);
        let covariance = hessian_to_covariance(&hessian).unwrap();
        assert!((covariance.get(0, 0) - 0.5).abs() < 1e-5);
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
