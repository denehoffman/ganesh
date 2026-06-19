use pyo3::{pyclass, pymethods};

use crate::core::EvalCounts;

/// Python-facing read-only wrapper for evaluation counts.
#[pyclass(skip_from_py_object, module = "ganesh", name = "EvalCounts")]
#[derive(Clone, Copy)]
pub struct PyEvalCounts {
    counts: EvalCounts,
}

#[pymethods]
impl PyEvalCounts {
    /// Function evaluation count.
    #[getter]
    pub const fn f(&self) -> usize {
        self.counts.f()
    }

    /// Gradient evaluation count.
    #[getter]
    pub const fn g(&self) -> usize {
        self.counts.g()
    }

    /// Hessian evaluation count.
    #[getter]
    pub const fn h(&self) -> usize {
        self.counts.h()
    }
}

impl From<EvalCounts> for PyEvalCounts {
    fn from(counts: EvalCounts) -> Self {
        Self { counts }
    }
}
