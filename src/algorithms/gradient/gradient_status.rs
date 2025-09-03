use crate::{traits::Status, DMatrix, DVector, Float};
use serde::{Deserialize, Serialize};

/// A status message struct containing all information about a minimization result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GradientStatus {
    /// A [`String`] message that can be set by [`Algorithm`](crate::traits::Algorithm)s.
    pub message: String,
    /// The current parameters of the minimization.
    pub x: DVector<Float>,
    /// The current value of the minimization problem function at [`GradientStatus::x`].
    pub fx: Float,
    /// The number of function evaluations (approximately, this is left up to individual
    /// [`Algorithm`](crate::traits::Algorithm)s to correctly compute and may not be exact).
    pub n_f_evals: usize,
    /// The number of gradient evaluations (approximately, this is left up to individual
    /// [`Algorithm`](crate::traits::Algorithm)s to correctly compute and may not be exact).
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
        use crate::core::utils::hessian_to_covariance;
        self.hess = Some(hessian.clone());
        let covariance = hessian_to_covariance(hessian);
        if let Some(cov_mat) = &covariance {
            self.err = Some(cov_mat.diagonal().map(Float::sqrt));
        }
        self.cov = covariance;
    }
}
