//! Python-facing status wrapper classes for built-in algorithms.
#![allow(clippy::doc_markdown, clippy::missing_errors_doc, missing_docs)]

use pyo3::{
    pyclass, pymethods,
    types::{PyDict, PyDictMethods, PyModule, PyModuleMethods},
    Bound, IntoPyObject, Py, PyAny, PyResult, Python,
};

use crate::{
    algorithms::{
        gradient::GradientStatus,
        gradient_free::{GradientFreeStatus, SimulatedAnnealingStatus},
        mcmc::EnsembleStatus,
        particles::{
            SwarmBoundaryMethod, SwarmParticle, SwarmPositionInitializer, SwarmStatus,
            SwarmTopology, SwarmUpdateMethod, SwarmVelocityInitializer,
        },
    },
    core::Point,
    python::numeric::{matrix_to_python, tensor3_to_python, vector_to_python},
    traits::StatusMessage,
    DMatrix, DVector, Float,
};

/// Register the built-in Python status wrapper classes in a native module.
pub fn register_status_types(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStatusMessage>()?;
    module.add_class::<PyGradientStatus>()?;
    module.add_class::<PyGradientFreeStatus>()?;
    module.add_class::<PyEnsembleStatus>()?;
    module.add_class::<PySwarmStatus>()?;
    module.add_class::<PySimulatedAnnealingStatus>()?;
    Ok(())
}

fn message_to_python<'py>(
    py: Python<'py>,
    message: &StatusMessage,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("status_type", message.status_type.to_string())?;
    dict.set_item("text", message.text.clone())?;
    dict.set_item("success", message.success())?;
    Ok(dict)
}

fn point_to_python<'py>(
    py: Python<'py>,
    point: &Point<DVector<Float>>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("x", vector_to_python(py, point.x.as_slice())?)?;
    dict.set_item("fx", point.fx)?;
    Ok(dict)
}

fn matrix_ref_to_python<'py>(
    py: Python<'py>,
    matrix: &DMatrix<Float>,
) -> PyResult<Bound<'py, PyAny>> {
    let rows = matrix
        .row_iter()
        .map(|row| row.iter().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    matrix_to_python(py, &rows)
}

fn optional_matrix_to_python<'py>(
    py: Python<'py>,
    matrix: &Option<DMatrix<Float>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    matrix
        .as_ref()
        .map(|matrix| matrix_ref_to_python(py, matrix))
        .transpose()
}

fn optional_vector_to_python<'py>(
    py: Python<'py>,
    vector: &Option<DVector<Float>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    vector
        .as_ref()
        .map(|vector| vector_to_python(py, vector.as_slice()))
        .transpose()
}

fn chain_to_python(chain: &[Vec<DVector<Float>>]) -> Vec<Vec<Vec<Float>>> {
    chain
        .iter()
        .map(|walker| {
            walker
                .iter()
                .map(|position| position.as_slice().to_vec())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn flat_chain_to_python(chain: &[DVector<Float>]) -> Vec<Vec<Float>> {
    chain
        .iter()
        .map(|position| position.as_slice().to_vec())
        .collect()
}

fn particle_to_python<'py>(
    py: Python<'py>,
    particle: &SwarmParticle,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("position", point_to_python(py, &particle.position)?)?;
    dict.set_item(
        "velocity",
        vector_to_python(py, particle.velocity.as_slice())?,
    )?;
    dict.set_item("best", point_to_python(py, &particle.best)?)?;
    Ok(dict)
}

fn swarm_to_python<'py>(
    py: Python<'py>,
    swarm: &crate::algorithms::particles::Swarm,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let particles = swarm
        .particles
        .iter()
        .map(|particle| particle_to_python(py, particle).map(|value| value.unbind()))
        .collect::<PyResult<Vec<_>>>()?;
    dict.set_item("particles", particles)?;
    dict.set_item("topology", topology_to_python(swarm.topology))?;
    dict.set_item(
        "update_method",
        update_method_to_python(swarm.update_method),
    )?;
    dict.set_item(
        "boundary_method",
        boundary_method_to_python(swarm.boundary_method),
    )?;
    dict.set_item(
        "position_initializer",
        position_initializer_to_python(py, &swarm.position_initializer)?,
    )?;
    dict.set_item(
        "velocity_initializer",
        velocity_initializer_to_python(py, &swarm.velocity_initializer)?,
    )?;
    Ok(dict)
}

const fn topology_to_python(topology: SwarmTopology) -> &'static str {
    match topology {
        SwarmTopology::Global => "Global",
        SwarmTopology::Ring => "Ring",
    }
}

const fn update_method_to_python(update_method: SwarmUpdateMethod) -> &'static str {
    match update_method {
        SwarmUpdateMethod::Synchronous => "Synchronous",
        SwarmUpdateMethod::Asynchronous => "Asynchronous",
    }
}

const fn boundary_method_to_python(boundary_method: SwarmBoundaryMethod) -> &'static str {
    match boundary_method {
        SwarmBoundaryMethod::Inf => "Inf",
        SwarmBoundaryMethod::Shr => "Shr",
    }
}

fn position_initializer_to_python<'py>(
    py: Python<'py>,
    initializer: &SwarmPositionInitializer,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match initializer {
        SwarmPositionInitializer::RandomInLimits {
            bounds,
            n_particles,
        } => {
            dict.set_item("kind", "RandomInLimits")?;
            dict.set_item("bounds", bounds.clone())?;
            dict.set_item("n_particles", *n_particles)?;
        }
        SwarmPositionInitializer::Custom(positions) => {
            let positions = positions
                .iter()
                .map(|position| {
                    vector_to_python(py, position.as_slice()).map(|value| value.unbind())
                })
                .collect::<PyResult<Vec<_>>>()?;
            dict.set_item("kind", "Custom")?;
            dict.set_item("positions", positions)?;
        }
        SwarmPositionInitializer::LatinHypercube {
            bounds,
            n_particles,
        } => {
            dict.set_item("kind", "LatinHypercube")?;
            dict.set_item("bounds", bounds.clone())?;
            dict.set_item("n_particles", *n_particles)?;
        }
    }
    Ok(dict)
}

fn velocity_initializer_to_python<'py>(
    py: Python<'py>,
    initializer: &SwarmVelocityInitializer,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match initializer {
        SwarmVelocityInitializer::Zero => {
            dict.set_item("kind", "Zero")?;
        }
        SwarmVelocityInitializer::RandomInLimits(limits) => {
            dict.set_item("kind", "RandomInLimits")?;
            dict.set_item("limits", limits.clone())?;
        }
    }
    Ok(dict)
}

#[pyclass(skip_from_py_object, module = "ganesh", name = "StatusMessage")]
#[derive(Clone)]
pub struct PyStatusMessage {
    message: StatusMessage,
}

#[pymethods]
impl PyStatusMessage {
    #[getter]
    pub fn status_type(&self) -> String {
        self.message.status_type.to_string()
    }

    #[getter]
    pub fn text(&self) -> String {
        self.message.text.clone()
    }

    #[getter]
    pub const fn success(&self) -> bool {
        self.message.success()
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        message_to_python(py, &self.message)
    }
}

impl From<StatusMessage> for PyStatusMessage {
    fn from(message: StatusMessage) -> Self {
        Self { message }
    }
}

impl From<PyStatusMessage> for StatusMessage {
    fn from(message: PyStatusMessage) -> Self {
        message.message
    }
}

impl From<&PyStatusMessage> for StatusMessage {
    fn from(message: &PyStatusMessage) -> Self {
        message.message.clone()
    }
}

#[pyclass(skip_from_py_object, module = "ganesh", name = "GradientStatus")]
#[derive(Clone)]
pub struct PyGradientStatus {
    status: GradientStatus,
}

#[pymethods]
impl PyGradientStatus {
    #[getter]
    pub fn message<'py>(&self, py: Python<'py>) -> PyResult<Py<PyStatusMessage>> {
        Py::new(py, PyStatusMessage::from(self.status.message.clone()))
    }

    #[getter]
    pub fn status_type(&self) -> String {
        self.status.message.status_type.to_string()
    }

    #[getter]
    pub fn message_text(&self) -> String {
        self.status.message.text.clone()
    }

    #[getter]
    pub const fn success(&self) -> bool {
        self.status.message.success()
    }

    #[getter]
    pub fn x<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.status.x.as_slice())
    }

    #[getter]
    pub const fn fx(&self) -> Float {
        self.status.fx
    }

    #[getter]
    pub const fn n_f_evals(&self) -> usize {
        self.status.n_f_evals
    }

    #[getter]
    pub const fn n_g_evals(&self) -> usize {
        self.status.n_g_evals
    }

    #[getter]
    pub const fn n_h_evals(&self) -> usize {
        self.status.n_h_evals
    }

    #[getter]
    pub fn hess<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        optional_matrix_to_python(py, &self.status.hess)
    }

    #[getter]
    pub fn cov<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        optional_matrix_to_python(py, &self.status.cov)
    }

    #[getter]
    pub fn err<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        optional_vector_to_python(py, &self.status.err)
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("message", message_to_python(py, &self.status.message)?)?;
        dict.set_item("x", vector_to_python(py, self.status.x.as_slice())?)?;
        dict.set_item("fx", self.status.fx)?;
        dict.set_item("n_f_evals", self.status.n_f_evals)?;
        dict.set_item("n_g_evals", self.status.n_g_evals)?;
        dict.set_item("n_h_evals", self.status.n_h_evals)?;
        dict.set_item("hess", optional_matrix_to_python(py, &self.status.hess)?)?;
        dict.set_item("cov", optional_matrix_to_python(py, &self.status.cov)?)?;
        dict.set_item("err", optional_vector_to_python(py, &self.status.err)?)?;
        Ok(dict)
    }
}

impl From<GradientStatus> for PyGradientStatus {
    fn from(status: GradientStatus) -> Self {
        Self { status }
    }
}

impl From<PyGradientStatus> for GradientStatus {
    fn from(status: PyGradientStatus) -> Self {
        status.status
    }
}

impl From<&PyGradientStatus> for GradientStatus {
    fn from(status: &PyGradientStatus) -> Self {
        status.status.clone()
    }
}

#[pyclass(skip_from_py_object, module = "ganesh", name = "GradientFreeStatus")]
#[derive(Clone)]
pub struct PyGradientFreeStatus {
    status: GradientFreeStatus,
}

#[pymethods]
impl PyGradientFreeStatus {
    #[getter]
    pub fn message<'py>(&self, py: Python<'py>) -> PyResult<Py<PyStatusMessage>> {
        Py::new(py, PyStatusMessage::from(self.status.message.clone()))
    }

    #[getter]
    pub fn status_type(&self) -> String {
        self.status.message.status_type.to_string()
    }

    #[getter]
    pub fn message_text(&self) -> String {
        self.status.message.text.clone()
    }

    #[getter]
    pub const fn success(&self) -> bool {
        self.status.message.success()
    }

    #[getter]
    pub fn x<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        vector_to_python(py, self.status.x.as_slice())
    }

    #[getter]
    pub const fn fx(&self) -> Float {
        self.status.fx
    }

    #[getter]
    pub const fn n_f_evals(&self) -> usize {
        self.status.n_f_evals
    }

    #[getter]
    pub fn hess<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        optional_matrix_to_python(py, &self.status.hess)
    }

    #[getter]
    pub fn cov<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        optional_matrix_to_python(py, &self.status.cov)
    }

    #[getter]
    pub fn err<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        optional_vector_to_python(py, &self.status.err)
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("message", message_to_python(py, &self.status.message)?)?;
        dict.set_item("x", vector_to_python(py, self.status.x.as_slice())?)?;
        dict.set_item("fx", self.status.fx)?;
        dict.set_item("n_f_evals", self.status.n_f_evals)?;
        dict.set_item("hess", optional_matrix_to_python(py, &self.status.hess)?)?;
        dict.set_item("cov", optional_matrix_to_python(py, &self.status.cov)?)?;
        dict.set_item("err", optional_vector_to_python(py, &self.status.err)?)?;
        Ok(dict)
    }
}

impl From<GradientFreeStatus> for PyGradientFreeStatus {
    fn from(status: GradientFreeStatus) -> Self {
        Self { status }
    }
}

impl From<PyGradientFreeStatus> for GradientFreeStatus {
    fn from(status: PyGradientFreeStatus) -> Self {
        status.status
    }
}

impl From<&PyGradientFreeStatus> for GradientFreeStatus {
    fn from(status: &PyGradientFreeStatus) -> Self {
        status.status.clone()
    }
}

#[pyclass(skip_from_py_object, module = "ganesh", name = "EnsembleStatus")]
#[derive(Clone)]
pub struct PyEnsembleStatus {
    status: EnsembleStatus,
}

#[pymethods]
impl PyEnsembleStatus {
    #[getter]
    pub fn message<'py>(&self, py: Python<'py>) -> PyResult<Py<PyStatusMessage>> {
        Py::new(py, PyStatusMessage::from(self.status.message.clone()))
    }

    #[getter]
    pub fn status_type(&self) -> String {
        self.status.message.status_type.to_string()
    }

    #[getter]
    pub fn message_text(&self) -> String {
        self.status.message.text.clone()
    }

    #[getter]
    pub const fn success(&self) -> bool {
        self.status.message.success()
    }

    #[getter]
    pub const fn n_f_evals(&self) -> usize {
        self.status.n_f_evals
    }

    #[getter]
    pub const fn n_g_evals(&self) -> usize {
        self.status.n_g_evals
    }

    #[getter]
    pub fn dimension(&self) -> (usize, usize, usize) {
        self.status.dimension()
    }

    #[pyo3(signature = (*, burn=None, thin=None))]
    pub fn get_chain<'py>(
        &self,
        py: Python<'py>,
        burn: Option<usize>,
        thin: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        tensor3_to_python(py, &chain_to_python(&self.status.get_chain(burn, thin)))
    }

    #[pyo3(signature = (*, burn=None, thin=None))]
    pub fn get_flat_chain<'py>(
        &self,
        py: Python<'py>,
        burn: Option<usize>,
        thin: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        matrix_to_python(
            py,
            &flat_chain_to_python(&self.status.get_flat_chain(burn, thin)),
        )
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("message", message_to_python(py, &self.status.message)?)?;
        dict.set_item(
            "chain",
            tensor3_to_python(py, &chain_to_python(&self.status.get_chain(None, None)))?,
        )?;
        dict.set_item("n_f_evals", self.status.n_f_evals)?;
        dict.set_item("n_g_evals", self.status.n_g_evals)?;
        dict.set_item("dimension", self.status.dimension())?;
        Ok(dict)
    }
}

impl From<EnsembleStatus> for PyEnsembleStatus {
    fn from(status: EnsembleStatus) -> Self {
        Self { status }
    }
}

impl From<PyEnsembleStatus> for EnsembleStatus {
    fn from(status: PyEnsembleStatus) -> Self {
        status.status
    }
}

impl From<&PyEnsembleStatus> for EnsembleStatus {
    fn from(status: &PyEnsembleStatus) -> Self {
        status.status.clone()
    }
}

#[pyclass(skip_from_py_object, module = "ganesh", name = "SwarmStatus")]
#[derive(Clone)]
pub struct PySwarmStatus {
    status: SwarmStatus,
}

#[pymethods]
impl PySwarmStatus {
    #[getter]
    pub fn message<'py>(&self, py: Python<'py>) -> PyResult<Py<PyStatusMessage>> {
        Py::new(py, PyStatusMessage::from(self.status.message.clone()))
    }

    #[getter]
    pub fn status_type(&self) -> String {
        self.status.message.status_type.to_string()
    }

    #[getter]
    pub fn message_text(&self) -> String {
        self.status.message.text.clone()
    }

    #[getter]
    pub const fn success(&self) -> bool {
        self.status.message.success()
    }

    #[getter]
    pub fn gbest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        point_to_python(py, &self.status.gbest)
    }

    #[getter]
    pub fn initial_gbest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        point_to_python(py, &self.status.initial_gbest)
    }

    #[getter]
    pub const fn n_f_evals(&self) -> usize {
        self.status.n_f_evals
    }

    #[getter]
    pub fn swarm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        swarm_to_python(py, &self.status.swarm)
    }

    pub fn get_best<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        point_to_python(py, &self.status.get_best())
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("message", message_to_python(py, &self.status.message)?)?;
        dict.set_item("gbest", point_to_python(py, &self.status.gbest)?)?;
        dict.set_item(
            "initial_gbest",
            point_to_python(py, &self.status.initial_gbest)?,
        )?;
        dict.set_item("n_f_evals", self.status.n_f_evals)?;
        dict.set_item("swarm", swarm_to_python(py, &self.status.swarm)?)?;
        Ok(dict)
    }
}

impl From<SwarmStatus> for PySwarmStatus {
    fn from(status: SwarmStatus) -> Self {
        Self { status }
    }
}

impl From<PySwarmStatus> for SwarmStatus {
    fn from(status: PySwarmStatus) -> Self {
        status.status
    }
}

impl From<&PySwarmStatus> for SwarmStatus {
    fn from(status: &PySwarmStatus) -> Self {
        status.status.clone()
    }
}

#[pyclass(
    skip_from_py_object,
    module = "ganesh",
    name = "SimulatedAnnealingStatus"
)]
#[derive(Clone)]
pub struct PySimulatedAnnealingStatus {
    status: SimulatedAnnealingStatus<DVector<Float>>,
}

#[pymethods]
impl PySimulatedAnnealingStatus {
    #[getter]
    pub fn message<'py>(&self, py: Python<'py>) -> PyResult<Py<PyStatusMessage>> {
        Py::new(py, PyStatusMessage::from(self.status.message.clone()))
    }

    #[getter]
    pub fn status_type(&self) -> String {
        self.status.message.status_type.to_string()
    }

    #[getter]
    pub fn message_text(&self) -> String {
        self.status.message.text.clone()
    }

    #[getter]
    pub const fn success(&self) -> bool {
        self.status.message.success()
    }

    #[getter]
    pub const fn temperature(&self) -> Float {
        self.status.temperature
    }

    #[getter]
    pub fn initial<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        point_to_python(py, &self.status.initial)
    }

    #[getter]
    pub fn best<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        point_to_python(py, &self.status.best)
    }

    #[getter]
    pub fn current<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        point_to_python(py, &self.status.current)
    }

    #[getter]
    pub const fn iteration(&self) -> usize {
        self.status.iteration
    }

    #[getter]
    pub const fn converged(&self) -> bool {
        self.status.converged
    }

    #[getter]
    pub const fn n_f_evals(&self) -> usize {
        self.status.n_f_evals
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("message", message_to_python(py, &self.status.message)?)?;
        dict.set_item("temperature", self.status.temperature)?;
        dict.set_item("initial", point_to_python(py, &self.status.initial)?)?;
        dict.set_item("best", point_to_python(py, &self.status.best)?)?;
        dict.set_item("current", point_to_python(py, &self.status.current)?)?;
        dict.set_item("iteration", self.status.iteration)?;
        dict.set_item("converged", self.status.converged)?;
        dict.set_item("n_f_evals", self.status.n_f_evals)?;
        Ok(dict)
    }
}

impl From<SimulatedAnnealingStatus<DVector<Float>>> for PySimulatedAnnealingStatus {
    fn from(status: SimulatedAnnealingStatus<DVector<Float>>) -> Self {
        Self { status }
    }
}

impl From<PySimulatedAnnealingStatus> for SimulatedAnnealingStatus<DVector<Float>> {
    fn from(status: PySimulatedAnnealingStatus) -> Self {
        status.status
    }
}

impl From<&PySimulatedAnnealingStatus> for SimulatedAnnealingStatus<DVector<Float>> {
    fn from(status: &PySimulatedAnnealingStatus) -> Self {
        status.status.clone()
    }
}

impl<'py> IntoPyObject<'py> for StatusMessage {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(py, PyStatusMessage::from(self))?
            .into_bound(py)
            .into_any())
    }
}

impl<'py> IntoPyObject<'py> for GradientStatus {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(py, PyGradientStatus::from(self))?
            .into_bound(py)
            .into_any())
    }
}

impl<'py> IntoPyObject<'py> for GradientFreeStatus {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(py, PyGradientFreeStatus::from(self))?
            .into_bound(py)
            .into_any())
    }
}

impl<'py> IntoPyObject<'py> for EnsembleStatus {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(py, PyEnsembleStatus::from(self))?
            .into_bound(py)
            .into_any())
    }
}

impl<'py> IntoPyObject<'py> for SwarmStatus {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(py, PySwarmStatus::from(self))?
            .into_bound(py)
            .into_any())
    }
}

impl<'py> IntoPyObject<'py> for SimulatedAnnealingStatus<DVector<Float>> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Py::new(py, PySimulatedAnnealingStatus::from(self))?
            .into_bound(py)
            .into_any())
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{types::PyAnyMethods, Py};

    use super::*;
    use crate::{
        algorithms::particles::{
            Swarm, SwarmBoundaryMethod, SwarmPositionInitializer, SwarmTopology, SwarmUpdateMethod,
            SwarmVelocityInitializer,
        },
        core::Point,
    };

    fn sample_gradient_status() -> GradientStatus {
        GradientStatus {
            message: StatusMessage::default().set_step_with_message("iterating"),
            x: DVector::from_vec(vec![0.25, 0.75]),
            fx: 0.5,
            n_f_evals: 12,
            n_g_evals: 7,
            n_h_evals: 2,
            hess: Some(DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0])),
            cov: Some(DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.25])),
            err: Some(DVector::from_vec(vec![
                std::f64::consts::FRAC_1_SQRT_2,
                0.5,
            ])),
        }
    }

    fn sample_swarm_status() -> SwarmStatus {
        let best = Point {
            x: DVector::from_vec(vec![0.25, -0.25]),
            fx: Some(0.125),
        };
        let other = Point {
            x: DVector::from_vec(vec![1.0, 1.5]),
            fx: Some(2.0),
        };
        let mut swarm = Swarm::new(SwarmPositionInitializer::RandomInLimits {
            bounds: vec![(-1.0, 1.0), (-2.0, 2.0)],
            n_particles: 0,
        });
        swarm.topology = SwarmTopology::Ring;
        swarm.update_method = SwarmUpdateMethod::Asynchronous;
        swarm.boundary_method = SwarmBoundaryMethod::Shr;
        swarm.velocity_initializer =
            SwarmVelocityInitializer::RandomInLimits(vec![(-0.5, 0.5), (-0.5, 0.5)]);
        SwarmStatus {
            gbest: best,
            initial_gbest: other,
            message: StatusMessage::default().set_step_with_message("swarm moved"),
            swarm,
            n_f_evals: 22,
        }
    }

    #[test]
    fn status_wrapper_roundtrip_converts_back_to_native() {
        let native = sample_gradient_status();
        let wrapper = PyGradientStatus::from(native.clone());
        let roundtrip = GradientStatus::from(wrapper);
        assert_eq!(roundtrip.fx, native.fx);
        assert_eq!(roundtrip.n_h_evals, native.n_h_evals);
        assert_eq!(roundtrip.message.text, native.message.text);
    }

    #[test]
    fn borrowed_status_wrapper_converts_back_to_native() {
        let wrapper = PySwarmStatus::from(sample_swarm_status());
        let native = SwarmStatus::from(&wrapper);
        assert_eq!(native.n_f_evals, 22);
        assert_eq!(native.message.text, "swarm moved");
        assert_eq!(native.gbest.fx, Some(0.125));
    }

    #[test]
    fn native_status_into_pyobject_returns_typed_wrapper() {
        crate::python::attach_for_tests(|py| {
            let wrapper = sample_gradient_status().into_pyobject(py).unwrap();
            let wrapper = wrapper.extract::<Py<PyGradientStatus>>().unwrap();
            let wrapper = wrapper.bind(py).borrow();
            assert_eq!(wrapper.fx(), 0.5);
            assert_eq!(wrapper.n_f_evals(), 12);
        });
    }
}
