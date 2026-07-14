use crate::{
    core::{EvalCounts, LinearAlgebra, NalgebraProvider, RealScalar, Vector},
    traits::{ProgressStatus, Status, StatusMessage},
};
use std::ops::ControlFlow;

/// Scalar- and linear-algebra-generic status for derivative-free minimizers.
#[derive(Debug, Clone)]
pub struct GradientFreeStatus<T: RealScalar = f64, B: LinearAlgebra<T> = NalgebraProvider> {
    /// Current status message.
    pub message: StatusMessage,
    /// Current best parameters.
    pub x: Vector<T, B>,
    /// Current best objective value.
    pub fx: T,
    /// Evaluation counts requested by the algorithm.
    pub evals: EvalCounts,
}

impl<T, B> Default for GradientFreeStatus<T, B>
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

impl<T, B> Status for GradientFreeStatus<T, B>
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

impl<T, B> GradientFreeStatus<T, B>
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

impl<T, B> ProgressStatus for GradientFreeStatus<T, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    fn write_progress(&self, out: &mut String) -> std::fmt::Result {
        use std::fmt::Write;
        write!(out, "status={} fx={}", self.message, self.fx)
    }
}
