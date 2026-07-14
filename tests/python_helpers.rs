#![cfg(feature = "python")]

#[pyo3::pymodule]
mod downstream_extension {
    #[pymodule_export]
    use ganesh::python::ganesh;
}

#[test]
fn declarative_submodule_can_be_reexported_downstream() {
    // Compilation of the module above is the contract under test. Keeping this as an
    // integration test also catches accidental reliance on crate-private items.
}
