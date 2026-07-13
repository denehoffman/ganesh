use crate::{
    core::{EvalCounts, LinearAlgebra, NalgebraBackend, RealScalar, Vector},
    traits::{ProgressStatus, Status, StatusMessage},
    DMatrix, DVector, Float,
};
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

/// A status message struct containing all information about a minimization result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GradientFreeStatus {
    /// A [`String`] message that can be set by [`Algorithm`](crate::traits::Algorithm)s.
    pub message: StatusMessage,
    /// The current parameters of the minimization.
    pub x: DVector<Float>,
    /// The current value of the minimization problem function at [`GradientFreeStatus::x`].
    pub fx: Float,
    /// Evaluation counts requested by the algorithm API.
    #[serde(flatten)]
    pub evals: EvalCounts,
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
        self.evals = Default::default();
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

    fn check_invariants(&mut self) -> ControlFlow<()> {
        if !self.fx.is_finite() {
            self.set_message().fail_with_message("f(x) is not finite");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

impl GradientFreeStatus {
    /// Updates the [`GradientFreeStatus::x`] and [`GradientFreeStatus::fx`] fields and sets the
    /// status message to an initialized state.
    pub fn initialize(&mut self, pos: (DVector<Float>, Float)) {
        self.set_message().initialize();
        self.x = pos.0;
        self.fx = pos.1;
    }
    /// Updates the [`GradientFreeStatus::x`] and [`GradientFreeStatus::fx`] fields and sets the
    /// status message to a step state.
    pub fn set_position(&mut self, pos: (DVector<Float>, Float)) {
        self.set_message().step();
        self.x = pos.0;
        self.fx = pos.1;
    }
    /// Updates the [`GradientFreeStatus::x`] and [`GradientFreeStatus::fx`] fields without
    /// touching the status message.
    pub fn set_position_silent(&mut self, pos: (DVector<Float>, Float)) {
        self.set_position(pos);
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

impl ProgressStatus for GradientFreeStatus {
    fn write_progress(&self, out: &mut String) -> std::fmt::Result {
        use std::fmt::Write;
        write!(out, "status={} fx={}", self.message, self.fx)
    }
}

/// Scalar- and backend-generic status for derivative-free minimizers.
#[derive(Debug, Clone)]
pub struct BackendGradientFreeStatus<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraBackend> {
    /// Current status message.
    pub message: StatusMessage,
    /// Current best parameters.
    pub x: Vector<T, B>,
    /// Current best objective value.
    pub fx: T,
    /// Evaluation counts requested by the algorithm.
    pub evals: EvalCounts,
}

impl<T, B> Default for BackendGradientFreeStatus<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            message: StatusMessage::default(),
            x: Vector::zeros(0),
            fx: T::zero(),
            evals: EvalCounts::default(),
        }
    }
}

impl<T, B> Status for BackendGradientFreeStatus<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn reset(&mut self) {
        let len = self.x.len();
        *self = Self::default();
        self.x = Vector::zeros(len);
    }

    fn message(&self) -> &StatusMessage {
        &self.message
    }

    fn set_message(&mut self) -> &mut StatusMessage {
        &mut self.message
    }

    fn check_invariants(&mut self) -> ControlFlow<()> {
        if self.message.success() {
            return ControlFlow::Break(());
        }
        if !self.fx.is_finite() {
            self.message.fail_with_message("f(x) is not finite");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

impl<T, B> BackendGradientFreeStatus<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Initialize the best position.
    pub fn initialize(&mut self, x: Vector<T, B>, fx: T) {
        self.message.initialize();
        self.x = x;
        self.fx = fx;
    }

    /// Update the best position.
    pub fn set_position(&mut self, x: Vector<T, B>, fx: T) {
        self.message.step();
        self.x = x;
        self.fx = fx;
    }
}

impl<T, B> ProgressStatus for BackendGradientFreeStatus<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn write_progress(&self, out: &mut String) -> std::fmt::Result {
        use std::fmt::Write;
        write!(out, "status={} fx={}", self.message, self.fx)
    }
}
