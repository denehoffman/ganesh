use crate::{
    DMatrix, DVector, Float,
    traits::{Status, StatusMessage},
};
use serde::{Deserialize, Serialize};

/// A status message struct containing all information about a minimization result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GradientFreeStatus {
    /// A [`String`] message that can be set by [`Algorithm`](crate::traits::Algorithm)s.
    pub message: StatusMessage,
    /// The current parameters of the minimization.
    pub x: DVector<Float>,
    /// The current value of the minimization problem function at [`GradientFreeStatus::x`].
    pub fx: Float,
    /// The number of function evaluations (approximately, this is left up to individual
    /// [`Algorithm`](crate::traits::Algorithm)s to correctly compute and may not be exact).
    pub n_f_evals: usize,
    /// The Hessian matrix at the end of the fit ([`None`] if not computed yet)
    pub hess: Option<DMatrix<Float>>,
    /// Covariance matrix at the end of the fit ([`None`] if not computed yet)
    pub cov: Option<DMatrix<Float>>,
    /// Errors on parameters at the end of the fit ([`None`] if not computed yet)
    pub err: Option<DVector<Float>>,
}

impl Status for GradientFreeStatus {
    fn reset(&mut self) {
        self.message = Default::default();
        self.x = DVector::zeros(self.x.len());
        self.fx = Default::default();
        self.n_f_evals = Default::default();
        self.hess = Default::default();
        self.cov = Default::default();
        self.err = Default::default();
    }

    fn message(&self) -> &StatusMessage {
        &self.message
    }

    fn set_message(&mut self) -> &mut StatusMessage {
        &mut self.message
    }
}

impl GradientFreeStatus {
    /// Updates the [`GradientFreeStatus::x`] and [`GradientFreeStatus::fx`] fields and sets the
    /// status message to an initialized state.
    pub fn initialize(&mut self, pos: (DVector<Float>, Float)) {
        self.set_message()
            .succeed_with_message(&format!("f(x) = {}", pos.1));
        self.x = pos.0;
        self.fx = pos.1;
    }
    /// Updates the [`GradientFreeStatus::x`] and [`GradientFreeStatus::fx`] fields and sets the
    /// status message to a step state.
    pub fn set_position(&mut self, pos: (DVector<Float>, Float)) {
        self.set_message()
            .step_with_message(&format!("f(x) = {}", pos.1));
        self.x = pos.0;
        self.fx = pos.1;
    }
    /// Increments [`GradientFreeStatus::n_f_evals`] by `1`.
    pub fn inc_n_f_evals(&mut self) {
        self.n_f_evals += 1;
    }
    /// Updates the [`GradientFreeStatus::err`] field.
    pub fn set_cov(&mut self, covariance: Option<DMatrix<Float>>) {
        if let Some(cov_mat) = &covariance {
            self.err = Some(cov_mat.diagonal().map(Float::sqrt));
        }
        self.cov = covariance;
    }
    /// Updates the [`GradientFreeStatus::hess`] field and computes [`GradientFreeStatus::cov`] and [`GradientFreeStatus::err`].
    pub fn set_hess(&mut self, hessian: &DMatrix<Float>) {
        use crate::core::utils::hessian_to_covariance;
        self.hess = Some(hessian.clone());
        let covariance = hessian_to_covariance(hessian);
        if let Some(cov_mat) = &covariance {
            self.err = Some(cov_mat.diagonal().map(Float::sqrt));
        }
        self.cov = covariance;
    }
}
