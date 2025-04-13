use std::fmt::Display;

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::{traits::Status, Float};

use super::Bound;

/// A status message struct containing all information about a minimization result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GradientStatus {
    /// The initial parameters of the minimization.
    pub x0: DVector<Float>,
    /// A [`String`] message that can be set by minimization [`Algorithm`]s.
    pub message: String,
    /// The current parameters of the minimization.
    pub x: DVector<Float>,
    /// The current value of the minimization problem function at [`Status::x`].
    pub fx: Float,
    /// The number of function evaluations (approximately, this is left up to individual
    /// [`Algorithm`]s to correctly compute and may not be exact).
    pub n_f_evals: usize,
    /// The number of gradient evaluations (approximately, this is left up to individual
    /// [`Algorithm`]s to correctly compute and may not be exact).
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

impl Display for GradientStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let title = format!(
          "╒══════════════════════════════════════════════════════════════════════════════════════════════╕
│{:^94}│",
          "FIT RESULTS",
      );
        let status = format!(
          "╞════════════════════════════════════════════╤════════════════════╤═════════════╤══════════════╡
│ Status: {}                    │ fval: {:+12.3E} │ #fcn: {:>5} │ #grad: {:>5} │",
          if self.converged {
              "Converged      "
          } else {
              "Invalid Minimum"
          },
          self.fx,
          self.n_f_evals,
          self.n_g_evals,
      );
        let message = format!(
          "├────────────────────────────────────────────┴────────────────────┴─────────────┴──────────────┤
│ Message: {:<83} │",
          self.message,
      );
        let header = "├───────╥──────────────┬──────────────╥──────────────┬──────────────┬──────────────┬───────────┤
│ Par # ║        Value │  Uncertainty ║      Initial │       -Bound │       +Bound │ At Limit? │
├───────╫──────────────┼──────────────╫──────────────┼──────────────┼──────────────┼───────────┤"
          .to_string();
        let mut res_list: Vec<String> = vec![];
        let errs = self
            .err
            .clone()
            .unwrap_or_else(|| DVector::from_element(self.x.len(), Float::NAN));
        // TODO: move the summary somewhere else
        let bounds = vec![Bound::NoBound; self.x.len()];
        for i in 0..self.x.len() {
            let row = format!(
              "│ {:>5} ║ {:>+12.3E} │ {:>+12.3E} ║ {:>+12.3E} │ {:>+12.3E} │ {:>+12.3E} │ {:^9} │",
              i,
              self.x[i],
              errs[i],
              self.x0[i],
              bounds[i].lower(),
              bounds[i].upper(),
              if bounds[i].at_bound(self.x[i]) { "yes" } else { "" }
          );
            res_list.push(row);
        }
        let bottom = "└───────╨──────────────┴──────────────╨──────────────┴──────────────┴──────────────┴───────────┘".to_string();
        let out = [title, status, message, header, res_list.join("\n"), bottom].join("\n");
        write!(f, "{}", out)
    }
}

impl GradientStatus {
    /// Sets the initial parameters of the minimization.
    pub fn with_x0<I: IntoIterator<Item = Float>>(mut self, x0: I) -> Self {
        let x0 = x0.into_iter().collect::<Vec<Float>>();
        self.x0 = DVector::from_column_slice(&x0);
        self
    }
    /// Updates the [`Status::message`] field.
    pub fn update_message(&mut self, message: &str) {
        self.message = message.to_string();
    }
    /// Updates the [`Status::x`] and [`Status::fx`] fields.
    pub fn update_position(&mut self, pos: (DVector<Float>, Float)) {
        self.x = pos.0;
        self.fx = pos.1;
    }
    /// Sets [`Status::converged`] to be `true`.
    pub fn set_converged(&mut self) {
        self.converged = true;
    }
    /// Increments [`Status::n_f_evals`] by `1`.
    pub fn inc_n_f_evals(&mut self) {
        self.n_f_evals += 1;
    }
    /// Increments [`Status::n_g_evals`] by `1`.
    pub fn inc_n_g_evals(&mut self) {
        self.n_g_evals += 1;
    }
    /// Sets the covariance matrix and updates parameter errors.
    pub fn set_cov(&mut self, covariance: Option<DMatrix<Float>>) {
        if let Some(cov_mat) = &covariance {
            self.err = Some(cov_mat.diagonal().map(Float::sqrt));
        }
        self.cov = covariance;
    }
    /// Sets the Hessian matrix, computes the covariance matrix, and updates parameter errors.
    pub fn set_hess(&mut self, hessian: &DMatrix<Float>) {
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
