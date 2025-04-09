/// Module containing the Nelder-Mead minimization algorithm.
pub mod nelder_mead;
use std::{fmt::Display, sync::Arc};

use nalgebra::{DMatrix, DVector};
pub use nelder_mead::NelderMead;

/// Module containing various line-search methods.
pub mod line_search;

/// Module containing the L-BFGS-B method.
pub mod lbfgsb;
pub use lbfgsb::LBFGSB;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::{
    init_ctrl_c_handler, is_ctrl_c_pressed, reset_ctrl_c_handler, traits::Observer, Bound, Float,
    Function,
};

/// A status message struct containing all information about a minimization result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Status {
    /// A [`String`] message that can be set by minimization [`Algorithm`]s.
    pub message: String,
    /// The current position of the minimization.
    pub x: DVector<Float>,
    /// The initial position of the minimization.
    pub x0: DVector<Float>,
    /// The bounds used for the minimization.
    pub bounds: Option<Vec<Bound>>,
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
    /// Optional parameter names
    pub parameters: Option<Vec<String>>,
}

impl Status {
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
    /// Sets parameter names.
    pub fn set_parameter_names<L: AsRef<str>>(&mut self, names: &[L]) {
        self.parameters = Some(names.iter().map(|name| name.as_ref().to_string()).collect());
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
impl Display for Status {
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
        let bounds = self
            .bounds
            .clone()
            .unwrap_or_else(|| vec![Bound::NoBound; self.x.len()]);
        for i in 0..self.x.len() {
            let row =
                format!(
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

/// A trait representing a minimization algorithm.
///
/// This trait is implemented for the algorithms found in the [`algorithms`](super) module, and contains
/// all the methods needed to be run by a [`Minimizer`].
pub trait Algorithm<U, E> {
    /// Any setup work done before the main steps of the algorithm should be done here.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
        x0: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E>;
    /// The main "step" of an algorithm, which is repeated until termination conditions are met or
    /// the max number of steps have been taken.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn step(
        &mut self,
        i_step: usize,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E>;
    /// Runs any termination/convergence checks and returns true if the algorithm has converged.
    /// Developers should also update the internal [`Status`] of the algorithm here if converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    fn check_for_termination(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<bool, E>;
    /// Runs any steps needed by the [`Algorithm`] after termination or convergence. This will run
    /// regardless of whether the [`Algorithm`] converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    #[allow(unused_variables)]
    fn postprocessing(
        &mut self,
        func: &dyn Function<U, E>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        Ok(())
    }
}

/// The main struct used for running [`Algorithm`]s on [`Function`]s.
pub struct Minimizer<U, E> {
    /// The [`Status`] of the [`Minimizer`], usually read after minimization.
    pub status: Status,
    algorithm: Box<dyn Algorithm<U, E>>,
    max_steps: usize,
    observers: Vec<Arc<RwLock<dyn Observer<U>>>>,
    dimension: usize,
    bounds: Option<Vec<Bound>>,
}
impl<U, E> Display for Minimizer<U, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.status)
    }
}
impl<U, E> Minimizer<U, E> {
    const DEFAULT_MAX_STEPS: usize = 4000;
    /// Creates a new [`Minimizer`] with the given (boxed) [`Algorithm`] and `dimension` set to the number
    /// of free parameters in the minimization problem.
    pub fn new(algorithm: Box<dyn Algorithm<U, E>>, dimension: usize) -> Self {
        Self {
            status: Status::default(),
            algorithm,
            max_steps: Self::DEFAULT_MAX_STEPS,
            observers: Vec::default(),
            dimension,
            bounds: None,
        }
    }
    fn reset_status(&mut self) {
        let new_status = Status {
            bounds: self.status.bounds.clone(),
            ..Default::default()
        };
        self.status = new_status;
    }
    /// Sets all [`Bound`]s of the [`Minimizer`]. This can be [`None`] for an unbounded problem, or
    /// [`Some`] [`Vec<(T, T)>`] with length equal to the number of free parameters. Individual
    /// upper or lower bounds can be unbounded by setting them equal to `T::infinity()` or
    /// `T::neg_infinity()` (e.g. `f64::INFINITY` and `f64::NEG_INFINITY`).
    ///
    /// # Panics
    ///
    /// This function will panic if the number of bounds is not equal to the number of free
    /// parameters.
    pub fn with_bounds<I: IntoIterator<Item = B>, B: Into<Bound>>(mut self, bounds: I) -> Self {
        let bounds = bounds.into_iter().map(Into::into).collect::<Vec<Bound>>();
        assert!(bounds.len() == self.dimension);
        self.bounds = Some(bounds);
        self
    }

    /// Set the maximum number of steps to perform before failure (default: 4000).
    pub const fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
    /// Adds a single [`Observer`] to the [`Minimizer`].
    pub fn with_observer(mut self, observer: Arc<RwLock<dyn Observer<U>>>) -> Self {
        self.observers.push(observer);
        self
    }
    /// Minimize the given [`Function`] starting at the point `x0`.
    ///
    /// This method first runs [`Algorithm::initialize`], then runs [`Algorithm::step`] in a loop,
    /// terminating if [`Algorithm::check_for_termination`] returns `true` or if
    /// the maximum number of allowed steps is exceeded. Each step will be followed by a sequential
    /// call to all given [`Observer`]s' callback functions. Finally, regardless of convergence,
    /// [`Algorithm::postprocessing`] is called. If the algorithm did not converge in the given
    /// step limit, the [`Status::message`] will be set to `"MAX EVALS"` at termination.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if the evaluation fails. See [`Function::evaluate`] for more
    /// information.
    ///
    /// # Panics
    ///
    /// This method will panic if the length of `x0` is not equal to the dimension of the problem
    /// (number of free parameters) or if any values of `x0` are outside the [`Bound`]s given to the
    /// [`Minimizer`].
    pub fn minimize(
        &mut self,
        func: &dyn Function<U, E>,
        x0: &[Float],
        user_data: &mut U,
    ) -> Result<(), E> {
        assert!(x0.len() == self.dimension);
        init_ctrl_c_handler();
        reset_ctrl_c_handler();
        self.reset_status();
        if let Some(bounds) = &self.bounds {
            for (i, (x_i, bound_i)) in x0.iter().zip(bounds).enumerate() {
                assert!(
                    bound_i.contains(*x_i),
                    "Parameter #{} = {} is outside of the given bound: {}",
                    i,
                    x_i,
                    bound_i
                )
            }
        }
        self.status.x0 = DVector::from_column_slice(x0);
        self.algorithm
            .initialize(func, x0, self.bounds.as_ref(), user_data, &mut self.status)?;
        let mut current_step = 0;
        let mut observer_termination = false;
        while current_step <= self.max_steps
            && !observer_termination
            && !self
                .algorithm
                .check_for_termination(func, user_data, &mut self.status)?
            && !is_ctrl_c_pressed()
        {
            self.algorithm
                .step(current_step, func, user_data, &mut self.status)?;
            current_step += 1;
            if !self.observers.is_empty() {
                for observer in self.observers.iter_mut() {
                    observer_termination =
                        observer
                            .write()
                            .callback(current_step, &mut self.status, user_data)
                            || observer_termination;
                }
            }
        }
        self.algorithm
            .postprocessing(func, user_data, &mut self.status)?;
        if current_step > self.max_steps && !self.status.converged {
            self.status.update_message("MAX EVALS");
        }
        if is_ctrl_c_pressed() {
            self.status.update_message("Ctrl-C Pressed");
        }
        Ok(())
    }
}
