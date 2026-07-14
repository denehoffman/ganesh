use crate::{
    core::{
        EvalCounts, LinearAlgebra, Matrix, NalgebraProvider, PseudoInverse, RealScalar, Vector,
    },
    traits::{ProgressStatus, Status, StatusMessage},
};
use std::ops::ControlFlow;

/// Scalar- and linear-algebra-generic status for gradient-based minimizers.
#[derive(Debug, Clone)]
pub struct GradientStatus<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Current status message.
    pub message: StatusMessage,
    /// Current parameters.
    pub x: Vector<T, B>,
    /// Current objective value.
    pub fx: T,
    /// Evaluation counts requested by the algorithm.
    pub evals: EvalCounts,
    /// Final Hessian, when computed.
    pub hess: Option<Matrix<T, B>>,
    /// Final covariance, when available from the provider.
    pub cov: Option<Matrix<T, B>>,
    /// Parameter standard deviations.
    pub err: Option<Vector<T, B>>,
}

impl<T, B> Default for GradientStatus<T, B>
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
            hess: None,
            cov: None,
            err: None,
        }
    }
}

impl<T, B> Status for GradientStatus<T, B>
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

impl<T, B> GradientStatus<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Initialize the current position and objective value.
    pub fn initialize(&mut self, x: Vector<T, B>, fx: T) {
        self.message.initialize();
        self.x = x;
        self.fx = fx;
    }

    /// Update the current position and objective value.
    pub fn set_position(&mut self, x: Vector<T, B>, fx: T) {
        self.message.step();
        self.x = x;
        self.fx = fx;
    }

    /// Store a covariance matrix and derive its diagonal standard deviations.
    pub fn set_cov(&mut self, covariance: Option<Matrix<T, B>>) {
        self.err = covariance.as_ref().map(|matrix| {
            Vector::from_vec(
                (0..matrix.rows())
                    .map(|index| matrix.get(index, index).sqrt())
                    .collect(),
            )
        });
        self.cov = covariance;
    }

    /// Store a Hessian and compute covariance through the selected provider.
    pub fn set_hess(&mut self, hessian: Matrix<T, B>)
    where
        B: PseudoInverse<T>,
    {
        let covariance = hessian.pseudo_inverse(T::epsilon().cbrt());
        self.hess = Some(hessian);
        self.set_cov(covariance);
    }
}

impl<T, B> ProgressStatus for GradientStatus<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn write_progress(&self, out: &mut String) -> std::fmt::Result {
        use std::fmt::Write;
        write!(out, "status={} fx={}", self.message, self.fx)
    }
}
