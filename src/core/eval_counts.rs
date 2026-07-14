use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign};

/// Evaluation counts for objective, gradient, and Hessian quantities requested through the
/// algorithm API.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct EvalCounts {
    /// Function evaluation count.
    #[serde(default, rename = "n_f_evals")]
    f: usize,
    /// Gradient evaluation count.
    #[serde(default, rename = "n_g_evals")]
    g: usize,
    /// Hessian evaluation count.
    #[serde(default, rename = "n_h_evals")]
    h: usize,
}

impl EvalCounts {
    /// Create evaluation counts from raw component counts.
    pub const fn new(f: usize, g: usize, h: usize) -> Self {
        Self { f, g, h }
    }

    /// Return the function evaluation count.
    pub const fn f(&self) -> usize {
        self.f
    }

    /// Return the gradient evaluation count.
    pub const fn g(&self) -> usize {
        self.g
    }

    /// Return the Hessian evaluation count.
    pub const fn h(&self) -> usize {
        self.h
    }

    /// Record one function evaluation result.
    pub const fn record_f(&mut self) {
        self.f += 1;
    }

    /// Record `count` function evaluation results.
    pub const fn record_many_f(&mut self, count: usize) {
        self.f += count;
    }

    /// Record one gradient evaluation result.
    pub const fn record_g(&mut self) {
        self.g += 1;
    }

    /// Record `count` gradient evaluation results.
    pub const fn record_many_g(&mut self, count: usize) {
        self.g += count;
    }

    /// Record one Hessian evaluation result.
    pub const fn record_h(&mut self) {
        self.h += 1;
    }

    /// Record `count` Hessian evaluation results.
    pub const fn record_many_h(&mut self, count: usize) {
        self.h += count;
    }

    /// Record one combined function and gradient evaluation result.
    pub const fn record_fg(&mut self) {
        self.record_f();
        self.record_g();
    }

    /// Record one combined function and Hessian evaluation result.
    pub const fn record_fh(&mut self) {
        self.record_f();
        self.record_h();
    }

    /// Record one combined gradient and Hessian evaluation result.
    pub const fn record_gh(&mut self) {
        self.record_g();
        self.record_h();
    }

    /// Record one combined function, gradient, and Hessian evaluation result.
    pub const fn record_fgh(&mut self) {
        self.record_f();
        self.record_g();
        self.record_h();
    }
}

impl AddAssign for EvalCounts {
    fn add_assign(&mut self, rhs: Self) {
        self.f += rhs.f;
        self.g += rhs.g;
        self.h += rhs.h;
    }
}

impl Add for EvalCounts {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_primitive_and_combined_evaluations() {
        let mut counts = EvalCounts::default();
        counts.record_f();
        counts.record_g();
        counts.record_h();
        counts.record_fg();
        counts.record_fh();
        counts.record_gh();
        counts.record_fgh();

        assert_eq!(counts, EvalCounts::new(4, 4, 4));
    }

    #[test]
    fn serializes_with_compatibility_field_names() {
        let json = serde_json::to_string(&EvalCounts::new(1, 2, 3)).unwrap();
        assert_eq!(json, r#"{"n_f_evals":1,"n_g_evals":2,"n_h_evals":3}"#);

        let counts: EvalCounts = serde_json::from_str(&json).unwrap();
        assert_eq!(counts.f(), 1);
        assert_eq!(counts.g(), 2);
        assert_eq!(counts.h(), 3);
    }

    #[test]
    fn defaults_missing_compatibility_fields() {
        let counts: EvalCounts = serde_json::from_str(r#"{"n_f_evals":5}"#).unwrap();
        assert_eq!(counts, EvalCounts::new(5, 0, 0));
    }
}
