use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::{traits::Status, Float};

/// A status message struct containing all information about a minimization result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GradientStatus {
    /// The initial parameters of the minimization.
    pub x0: DVector<Float>,
    /// A [`String`] message that can be set by [`Solver`](crate::traits::Solver)s.
    pub message: String,
    /// The current parameters of the minimization.
    pub x: DVector<Float>,
    /// The current value of the minimization problem function at [`GradientStatus::x`].
    pub fx: Float,
    /// The number of function evaluations (approximately, this is left up to individual
    /// [`Solver`](crate::traits::Solver)s to correctly compute and may not be exact).
    pub n_f_evals: usize,
    /// The number of gradient evaluations (approximately, this is left up to individual
    /// [`Solver`](crate::traits::Solver)s to correctly compute and may not be exact).
    pub n_g_evals: usize,
    /// Flag that says whether or not the fit is in a converged state.
    pub converged: bool,
    /// The Hessian matrix at the end of the fit ([`None`] if not computed yet)
    pub hess: Option<DMatrix<Float>>,
    /// Covariance matrix at the end of the fit ([`None`] if not computed yet)
    pub cov: Option<DMatrix<Float>>,
    /// Errors on parameters at the end of the fit ([`None`] if not computed yet)
    pub err: Option<DVector<Float>>,
}

impl Status for GradientStatus {
    fn reset(&mut self) {
        self.message = String::new();
        self.x = DVector::zeros(self.x.len());
        self.fx = Float::default();
        self.n_f_evals = 0;
        self.n_g_evals = 0;
        self.converged = false;
        self.hess = None;
        self.cov = None;
        self.err = None;
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn message(&self) -> &str {
        &self.message
    }
    fn update_message(&mut self, message: &str) {
        self.message = message.to_string();
    }
}

impl GradientStatus {
    /// Updates the [`GradientStatus::x0`] field.
    pub fn with_x0<I: IntoIterator<Item = Float>>(&mut self, x0: I) -> &mut Self {
        let x0 = x0.into_iter().collect::<Vec<Float>>();
        self.x0 = DVector::from_column_slice(&x0);
        self
    }
    /// Updates the [`GradientStatus::message`] field.
    pub fn with_message(&mut self, message: &str) {
        self.message = message.to_string();
    }
    /// Updates the [`GradientStatus::x`] and [`GradientStatus::fx`] fields.
    pub fn with_position(&mut self, pos: (DVector<Float>, Float)) {
        self.x = pos.0;
        self.fx = pos.1;
    }
    /// Sets [`GradientStatus::converged`] to be `true`.
    pub fn set_converged(&mut self) {
        self.converged = true;
    }
    /// Increments [`GradientStatus::n_f_evals`] by `1`.
    pub fn inc_n_f_evals(&mut self) {
        self.n_f_evals += 1;
    }
    /// Increments [`GradientStatus::n_g_evals`] by `1`.
    pub fn inc_n_g_evals(&mut self) {
        self.n_g_evals += 1;
    }
    /// Updates the [`GradientStatus::err`] field.
    pub fn with_cov(&mut self, covariance: Option<DMatrix<Float>>) {
        if let Some(cov_mat) = &covariance {
            self.err = Some(cov_mat.diagonal().map(Float::sqrt));
        }
        self.cov = covariance;
    }
    /// Updates the [`GradientStatus::hess`] field and computes [`GradientStatus::cov`] and [`GradientStatus::err`].
    pub fn with_hess(&mut self, hessian: &DMatrix<Float>) {
        self.hess = Some(hessian.clone());
        let mut covariance = hessian.clone().try_inverse();
        if covariance.is_none() {
            covariance = hessian
                .clone()
                .pseudo_inverse(Float::cbrt(Float::EPSILON))
                .ok();
        }
        if let Some(cov_mat) = &covariance {
            self.err = Some(cov_mat.diagonal().map(Float::sqrt));
        }
        self.cov = covariance;
    }
}
