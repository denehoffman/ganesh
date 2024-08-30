use std::fmt::Debug;

use nalgebra::{DVector, RealField};
use num::{Float, FromPrimitive};

use crate::{convert, Bound, Function, Status};

/// A trait which defines the methods for a line search algorithm.
///
/// Line searches are one-dimensional minimizers typically used to determine optimal step sizes for
/// [`Algorithm`](`crate::Algorithm`)s which only provide a direction for the next optimal step.
pub trait LineSearch<T, U, E> {
    /// The search method takes the current position of the minimizer, `x`, the search direction
    /// `p`, the objective function `func`, optional bounds `bounds`, and any arguments to the
    /// objective function `user_data`, and returns a [`Result`] containing the tuple,
    /// `(step_size, func(x + step_size * p), grad(x + step_size * p))`. Returns a [`None`]
    /// [`Result`] if the algorithm fails to find improvement.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn search(
        &mut self,
        x: &DVector<T>,
        p: &DVector<T>,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
        status: &mut Status<T>,
    ) -> Result<Option<(T, T, Vec<T>)>, E>;
}

/// A minimal line search algorithm which satisfies the Armijo condition. This is equivalent to
/// Algorithm 3.1 from Nocedal and Wright's book "Numerical Optimization"[^1] (page 37).
///
/// [^1]: [Numerical Optimization. Springer New York, 2006. doi: 10.1007/978-0-387-40065-5.](https://doi.org/10.1007/978-0-387-40065-5)
pub struct BacktrackingLineSearch<T> {
    alpha_0: T,
    rho: T,
    c: T,
}
impl<T> Default for BacktrackingLineSearch<T>
where
    T: Float,
{
    fn default() -> Self {
        Self {
            alpha_0: T::one(),
            rho: convert!(0.5, T),
            c: convert!(1e-4, T),
        }
    }
}

impl<T, U, E> LineSearch<T, U, E> for BacktrackingLineSearch<T>
where
    T: Float + RealField,
{
    fn search(
        &mut self,
        x: &DVector<T>,
        p: &DVector<T>,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
        status: &mut Status<T>,
    ) -> Result<Option<(T, T, Vec<T>)>, E> {
        let mut alpha_i = self.alpha_0;
        let phi = |alpha: T, ud: &mut U, st: &mut Status<T>| -> Result<T, E> {
            st.inc_n_f_evals();
            func.evaluate_bounded((x + p.scale(alpha)).as_slice(), bounds, ud)
        };
        let dphi = |alpha: T, ud: &mut U, st: &mut Status<T>| -> Result<T, E> {
            st.inc_n_g_evals();
            Ok(DVector::from_vec(func.gradient_bounded(
                (x + p.scale(alpha)).as_slice(),
                bounds,
                ud,
            )?)
            .dot(p))
        };
        let phi_0 = phi(T::zero(), user_data, status)?;
        let mut phi_alpha_i = phi(alpha_i, user_data, status)?;
        let dphi_0 = dphi(T::zero(), user_data, status)?;
        loop {
            let armijo = phi_alpha_i <= phi_0 + self.c * alpha_i * dphi_0;
            if armijo {
                let g_alpha_i =
                    func.gradient_bounded((x + p.scale(alpha_i)).as_slice(), bounds, user_data)?;
                return Ok(Some((alpha_i, phi_alpha_i, g_alpha_i)));
            }
            alpha_i = self.rho * alpha_i;
            phi_alpha_i = phi(alpha_i, user_data, status)?;
        }
    }
}

/// A line search which implements Algorithms 3.5 and 3.6 from Nocedal and Wright's book "Numerical
/// Optimization"[^1] (pages 60-61). This algorithm upholds the strong Wolfe conditions.
///
/// [^1]: [Numerical Optimization. Springer New York, 2006. doi: 10.1007/978-0-387-40065-5.](https://doi.org/10.1007/978-0-387-40065-5)
pub struct StrongWolfeLineSearch<T> {
    alpha_max: Option<T>,
    max_iters: usize,
    max_zoom: usize,
    c1: T,
    c2: T,
    old_phi_0: Option<T>,
}

impl<T> Default for StrongWolfeLineSearch<T>
where
    T: Float,
{
    fn default() -> Self {
        Self {
            alpha_max: None,
            max_iters: 100,
            max_zoom: 100,
            c1: convert!(1e-4, T),
            c2: convert!(0.9, T),
            old_phi_0: None,
        }
    }
}

impl<T> StrongWolfeLineSearch<T>
where
    T: Float,
{
    /// Set the maximum allowed step size for the search (defaults to [`None`]).
    pub const fn with_max_step(mut self, alpha_max: Option<T>) -> Self {
        self.alpha_max = alpha_max;
        self
    }
    /// Set the maximum allowed iterations of the algorithm (defaults to 10).
    pub const fn with_max_iterations(mut self, max_iters: usize) -> Self {
        self.max_iters = max_iters;
        self
    }
    /// Set the maximum allowed iterations of the internal zoom algorithm (defaults to 10).
    pub const fn with_max_zoom(mut self, max_zoom: usize) -> Self {
        self.max_zoom = max_zoom;
        self
    }
    /// Set the first control parameter, used in the Armijo condition evaluation (defaults to
    /// 1e-4).
    ///
    /// # Panics
    ///
    /// This method will panic if the condition $`0 < c_1 < c_2 < 1`$ is not met.
    pub fn with_c1(mut self, c1: T) -> Self {
        assert!(T::zero() < c1);
        assert!(c1 < self.c2);
        self.c1 = c1;
        self
    }
    /// Set the second control parameter, used in the second Wolfe condition (defaults to 0.9).
    ///
    /// # Panics
    ///
    /// This method will panic if the condition $`0 < c_1 < c_2 < 1`$ is not met.
    pub fn with_c2(mut self, c2: T) -> Self {
        assert!(T::one() > c2);
        assert!(c2 > self.c1);
        self.c2 = c2;
        self
    }
    /// Use the previous evaluation of the function to give a closer estimate for the initial step
    /// size.
    pub const fn with_previous_evaluation(mut self, fx: T) -> Self {
        self.old_phi_0 = Some(fx);
        self
    }
}

impl<T, U, E> LineSearch<T, U, E> for StrongWolfeLineSearch<T>
where
    T: Float + FromPrimitive + Debug + RealField + 'static,
{
    fn search(
        &mut self,
        x: &DVector<T>,
        p: &DVector<T>,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
        status: &mut Status<T>,
    ) -> Result<Option<(T, T, Vec<T>)>, E> {
        let phi = |alpha: T, ud: &mut U, st: &mut Status<T>| -> Result<T, E> {
            st.inc_n_f_evals();
            func.evaluate_bounded((x + p.scale(alpha)).as_slice(), bounds, ud)
        };
        let dphi = |alpha: T, ud: &mut U, st: &mut Status<T>| -> Result<(T, Vec<T>), E> {
            st.inc_n_g_evals();
            let gradient_vec =
                func.gradient_bounded((x + p.scale(alpha)).as_slice(), bounds, ud)?;
            Ok((DVector::from_vec(gradient_vec.clone()).dot(p), gradient_vec))
        };
        let phi_0 = phi(T::zero(), user_data, status)?;
        let (dphi_0, g_0) = dphi(T::zero(), user_data, status)?;
        let zoom = |alpha_lo: T,
                    alpha_hi: T,
                    ud: &mut U,
                    st: &mut Status<T>|
         -> Result<Option<(T, T, Vec<T>)>, E> {
            let mut alpha_lo = alpha_lo;
            let mut alpha_hi = alpha_hi;
            let mut alpha_j = (alpha_lo + alpha_hi) / (convert!(2, T));
            let mut phi_alpha_j = phi(alpha_j, ud, st)?;
            for _ in 0..self.max_zoom {
                let armijo = phi_alpha_j <= phi_0 + self.c1 * alpha_j * dphi_0;
                if !armijo || (phi_alpha_j >= phi(alpha_lo, ud, st)?) {
                    alpha_hi = alpha_j;
                } else {
                    let (dphi_alpha_j, g_alpha_j) = dphi(alpha_j, ud, st)?;
                    let wolfe2 = Float::abs(dphi_alpha_j) <= -self.c2 * dphi_0;
                    if wolfe2 {
                        return Ok(Some((alpha_j, phi_alpha_j, g_alpha_j)));
                    }
                    if dphi_alpha_j * (alpha_hi - alpha_lo) >= T::zero() {
                        alpha_hi = alpha_lo
                    }
                    alpha_lo = alpha_j;
                }
                (alpha_lo, alpha_hi) = if alpha_lo <= alpha_hi {
                    (alpha_lo, alpha_hi)
                } else {
                    (alpha_hi, alpha_lo)
                };
                alpha_j = (alpha_lo + alpha_hi) / (convert!(2, T));
                phi_alpha_j = phi(alpha_j, ud, st)?;
            }
            st.update_message("FAILED TO FIND LOCAL IMPROVEMENT");
            Ok(None)
        };
        let mut alpha_im1 = T::zero();
        let mut alpha_i = self.old_phi_0.map_or_else(T::one, |old_phi_0| {
            if dphi_0 != T::zero() {
                Float::min(T::one(), convert!(2.02, T) * (phi_0 - old_phi_0) / dphi_0)
            } else {
                T::one()
            }
        });
        if alpha_i < T::zero() {
            alpha_i = T::one();
        }
        if let Some(alpha_max) = self.alpha_max {
            alpha_i = Float::min(alpha_i, alpha_max)
        }
        let mut phi_alpha_im1 = phi_0;
        #[allow(unused_assignments)]
        let mut phi_alpha_i = phi_0;
        #[allow(unused_assignments)]
        let mut dphi_alpha_i = dphi_0;
        #[allow(unused_assignments)]
        let mut g_alpha_i = g_0;
        for i in 0..self.max_iters {
            phi_alpha_i = phi(alpha_i, user_data, status)?;
            let armijo = phi_alpha_i <= phi_0 + self.c1 * alpha_i * dphi_0;
            if !armijo || (phi_alpha_i >= phi_alpha_im1 && i > 0) {
                return zoom(alpha_im1, alpha_i, user_data, status);
            }
            (dphi_alpha_i, g_alpha_i) = dphi(alpha_i, user_data, status)?;
            let wolfe2 = Float::abs(dphi_alpha_i) <= -self.c2 * dphi_0;
            if wolfe2 {
                return Ok(Some((alpha_i, phi_alpha_i, g_alpha_i)));
            }
            if dphi_alpha_i >= T::zero() {
                return zoom(alpha_i, alpha_im1, user_data, status);
            }
            alpha_im1 = alpha_i;
            phi_alpha_im1 = phi_alpha_i;
            if let Some(alpha_max) = self.alpha_max {
                alpha_i = (alpha_i + alpha_max) / convert!(2, T);
            } else {
                alpha_i *= convert!(2, T);
            }
        }
        status.update_message("FAILED TO FIND LOCAL IMPROVEMENT");
        Ok(None)
    }
}
