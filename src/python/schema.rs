//! Machine-readable schema helpers for typed Python config wrappers.

use serde::Serialize;

use crate::python::config::{
    PyAIESConfig, PyCMAESConfig, PyDifferentialEvolutionConfig, PyESSConfig, PyLBFGSBConfig,
    PyNelderMeadConfig, PyPSOConfig, PySimulatedAnnealingConfig,
};

/// A machine-readable schema for a typed Python-facing config wrapper.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConfigSchema {
    /// The exported Python class name.
    pub name: &'static str,
    /// The schema version for downstream compatibility checks.
    pub version: usize,
    /// The fields exposed by the wrapper.
    pub fields: Vec<ConfigFieldSchema>,
}

impl ConfigSchema {
    /// Serialize the schema as pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// A machine-readable schema entry for one config field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConfigFieldSchema {
    /// The exported field name.
    pub name: &'static str,
    /// The expected Python-side field type.
    pub kind: ConfigFieldKind,
    /// Whether the field must be provided by the caller.
    pub required: bool,
    /// The default value rendered as a string when one exists.
    pub default: Option<&'static str>,
    /// Short human-facing field description.
    pub description: &'static str,
}

/// A simplified logical type for Python config fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ConfigFieldKind {
    /// A scalar floating-point field.
    Float,
    /// A required integer field.
    Integer,
    /// An optional integer field.
    OptionalInteger,
    /// A one-dimensional float array field.
    VectorFloat,
    /// A two-dimensional float array field.
    MatrixFloat,
    /// An optional per-parameter bounds field.
    Bounds,
    /// An optional list of parameter-name strings.
    ParameterNames,
}

/// Schema/introspection support for typed Python config wrappers.
pub trait HasPyConfigSchema {
    /// Return the stable wrapper schema.
    fn schema() -> ConfigSchema;

    /// Return the schema as pretty JSON.
    fn schema_json_pretty() -> Result<String, serde_json::Error> {
        Self::schema().to_json_pretty()
    }
}

fn field(
    name: &'static str,
    kind: ConfigFieldKind,
    required: bool,
    default: Option<&'static str>,
    description: &'static str,
) -> ConfigFieldSchema {
    ConfigFieldSchema {
        name,
        kind,
        required,
        default,
        description,
    }
}

impl HasPyConfigSchema for PyLBFGSBConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "LBFGSBConfig",
            version: 1,
            fields: vec![
                field("x0", ConfigFieldKind::VectorFloat, true, None, "Initial parameter vector."),
                field(
                    "memory_limit",
                    ConfigFieldKind::Integer,
                    false,
                    Some("10"),
                    "Number of stored L-BFGS-B correction pairs.",
                ),
                field(
                    "bounds",
                    ConfigFieldKind::Bounds,
                    false,
                    Some("None"),
                    "Optional lower/upper bounds per parameter.",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for PyNelderMeadConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "NelderMeadConfig",
            version: 1,
            fields: vec![
                field("x0", ConfigFieldKind::VectorFloat, true, None, "Initial simplex anchor."),
                field(
                    "bounds",
                    ConfigFieldKind::Bounds,
                    false,
                    Some("None"),
                    "Optional lower/upper bounds per parameter.",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for PyPSOConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "PSOConfig",
            version: 1,
            fields: vec![
                field(
                    "positions",
                    ConfigFieldKind::MatrixFloat,
                    true,
                    None,
                    "Initial swarm positions with shape (n_particles, n_parameters).",
                ),
                field(
                    "bounds",
                    ConfigFieldKind::Bounds,
                    false,
                    Some("None"),
                    "Optional lower/upper bounds per parameter.",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for PyAIESConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "AIESConfig",
            version: 1,
            fields: vec![
                field(
                    "walkers",
                    ConfigFieldKind::MatrixFloat,
                    true,
                    None,
                    "Initial walker positions with shape (n_walkers, n_parameters).",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for PyESSConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "ESSConfig",
            version: 1,
            fields: vec![
                field(
                    "walkers",
                    ConfigFieldKind::MatrixFloat,
                    true,
                    None,
                    "Initial walker positions with shape (n_walkers, n_parameters).",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
                field(
                    "n_adaptive",
                    ConfigFieldKind::Integer,
                    false,
                    Some("0"),
                    "Number of adaptive warmup steps.",
                ),
                field(
                    "max_steps",
                    ConfigFieldKind::Integer,
                    false,
                    Some("10000"),
                    "Maximum internal ESS proposal steps.",
                ),
                field(
                    "mu",
                    ConfigFieldKind::Float,
                    false,
                    Some("1.0"),
                    "Differential-move scaling parameter.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for PyDifferentialEvolutionConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "DifferentialEvolutionConfig",
            version: 1,
            fields: vec![
                field("x0", ConfigFieldKind::VectorFloat, true, None, "Initial parameter vector."),
                field(
                    "population_size",
                    ConfigFieldKind::OptionalInteger,
                    false,
                    Some("None"),
                    "Optional population size override.",
                ),
                field(
                    "differential_weight",
                    ConfigFieldKind::Float,
                    false,
                    Some("0.8"),
                    "Mutation differential weight F.",
                ),
                field(
                    "crossover_probability",
                    ConfigFieldKind::Float,
                    false,
                    Some("0.9"),
                    "Binomial crossover probability CR.",
                ),
                field(
                    "initial_scale",
                    ConfigFieldKind::Float,
                    false,
                    Some("1.0"),
                    "External-space initialization half-width around x0.",
                ),
                field(
                    "bounds",
                    ConfigFieldKind::Bounds,
                    false,
                    Some("None"),
                    "Optional lower/upper bounds per parameter.",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for PyCMAESConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "CMAESConfig",
            version: 1,
            fields: vec![
                field("x0", ConfigFieldKind::VectorFloat, true, None, "Initial mean vector."),
                field(
                    "sigma",
                    ConfigFieldKind::Float,
                    true,
                    None,
                    "Initial global step size.",
                ),
                field(
                    "population_size",
                    ConfigFieldKind::OptionalInteger,
                    false,
                    Some("None"),
                    "Optional offspring population size.",
                ),
                field(
                    "bounds",
                    ConfigFieldKind::Bounds,
                    false,
                    Some("None"),
                    "Optional lower/upper bounds per parameter.",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for PySimulatedAnnealingConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "SimulatedAnnealingConfig",
            version: 1,
            fields: vec![
                field(
                    "initial_temperature",
                    ConfigFieldKind::Float,
                    true,
                    None,
                    "Initial annealing temperature.",
                ),
                field(
                    "cooling_rate",
                    ConfigFieldKind::Float,
                    true,
                    None,
                    "Multiplicative cooling rate in (0, 1).",
                ),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lbfgsb_schema_has_expected_fields() {
        let schema = PyLBFGSBConfig::schema();
        assert_eq!(schema.name, "LBFGSBConfig");
        assert_eq!(schema.version, 1);
        assert_eq!(schema.fields[0].name, "x0");
        assert_eq!(schema.fields[1].name, "memory_limit");
    }

    #[test]
    fn cmaes_schema_json_contains_sigma() {
        let json = PyCMAESConfig::schema_json_pretty().unwrap();
        assert!(json.contains("\"name\": \"CMAESConfig\""));
        assert!(json.contains("\"sigma\""));
    }
}
