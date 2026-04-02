//! Machine-readable schema helpers for the pure Python config contract.

use serde::Serialize;

use crate::algorithms::{
    gradient::{AdamConfig, ConjugateGradientConfig, LBFGSBConfig, TrustRegionConfig},
    gradient_free::{
        CMAESConfig, DifferentialEvolutionConfig, NelderMeadConfig, SimulatedAnnealingConfig,
    },
    mcmc::{AIESConfig, ESSConfig},
    particles::PSOConfig,
};

/// A machine-readable schema for a Python-facing config class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConfigSchema {
    /// The exported Python class name.
    pub name: &'static str,
    /// The schema version for downstream compatibility checks.
    pub version: usize,
    /// The fields exposed by the Python class.
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
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ConfigFieldKind {
    Float,
    Integer,
    OptionalInteger,
    String,
    VectorFloat,
    MatrixFloat,
    Bounds,
    ParameterNames,
    OptionalObject,
    ObjectSequence,
}

/// Schema/introspection support for Python-facing config classes that extract into native configs.
pub trait HasPyConfigSchema {
    /// Return the stable Python config schema.
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

impl HasPyConfigSchema for LBFGSBConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "LBFGSBConfig",
            version: 1,
            fields: vec![
                field(
                    "x0",
                    ConfigFieldKind::VectorFloat,
                    true,
                    None,
                    "Initial parameter vector.",
                ),
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
                    "Optional lower and upper bounds per parameter.",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
                field(
                    "bounds_handling",
                    ConfigFieldKind::String,
                    false,
                    Some("None"),
                    "Optional bounds handling mode: auto, native_bounds, or transform_bounds.",
                ),
                field(
                    "line_search",
                    ConfigFieldKind::OptionalObject,
                    false,
                    Some("None"),
                    "Optional strong-Wolfe line-search helper object.",
                ),
                field(
                    "error_mode",
                    ConfigFieldKind::String,
                    false,
                    Some("None"),
                    "Optional error mode: exact_hessian or skip.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for NelderMeadConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "NelderMeadConfig",
            version: 1,
            fields: vec![
                field(
                    "x0",
                    ConfigFieldKind::VectorFloat,
                    false,
                    Some("None"),
                    "Initial simplex anchor when construction_method is not supplied.",
                ),
                field(
                    "construction_method",
                    ConfigFieldKind::OptionalObject,
                    false,
                    Some("None"),
                    "Optional simplex construction helper object.",
                ),
                field(
                    "bounds",
                    ConfigFieldKind::Bounds,
                    false,
                    Some("None"),
                    "Optional lower and upper bounds per parameter.",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
                field(
                    "alpha",
                    ConfigFieldKind::Float,
                    false,
                    Some("None"),
                    "Optional reflection coefficient.",
                ),
                field(
                    "beta",
                    ConfigFieldKind::Float,
                    false,
                    Some("None"),
                    "Optional expansion coefficient.",
                ),
                field(
                    "gamma",
                    ConfigFieldKind::Float,
                    false,
                    Some("None"),
                    "Optional contraction coefficient.",
                ),
                field(
                    "delta",
                    ConfigFieldKind::Float,
                    false,
                    Some("None"),
                    "Optional shrink coefficient.",
                ),
                field(
                    "adaptive_dimension",
                    ConfigFieldKind::Integer,
                    false,
                    Some("None"),
                    "Optional dimension used to enable adaptive Nelder-Mead parameters.",
                ),
                field(
                    "expansion_method",
                    ConfigFieldKind::String,
                    false,
                    Some("None"),
                    "Optional simplex expansion method.",
                ),
                field(
                    "bounds_handling",
                    ConfigFieldKind::String,
                    false,
                    Some("None"),
                    "Optional bounds handling mode: auto, native_bounds, or transform_bounds.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for PSOConfig {
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
                    "Optional lower and upper bounds per parameter.",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
                field(
                    "omega",
                    ConfigFieldKind::Float,
                    false,
                    Some("0.8"),
                    "Inertial weight.",
                ),
                field(
                    "c1",
                    ConfigFieldKind::Float,
                    false,
                    Some("0.1"),
                    "Cognitive weight.",
                ),
                field(
                    "c2",
                    ConfigFieldKind::Float,
                    false,
                    Some("0.1"),
                    "Social weight.",
                ),
                field(
                    "bounds_handling",
                    ConfigFieldKind::String,
                    false,
                    Some("None"),
                    "Optional bounds handling mode: auto, native_bounds, or transform_bounds.",
                ),
                field(
                    "topology",
                    ConfigFieldKind::String,
                    false,
                    Some("None"),
                    "Optional swarm topology.",
                ),
                field(
                    "update_method",
                    ConfigFieldKind::String,
                    false,
                    Some("None"),
                    "Optional swarm update method.",
                ),
                field(
                    "boundary_method",
                    ConfigFieldKind::String,
                    false,
                    Some("None"),
                    "Optional swarm boundary handling method.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for AIESConfig {
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
                field(
                    "moves",
                    ConfigFieldKind::ObjectSequence,
                    false,
                    Some("None"),
                    "Optional list of AIES move helper objects.",
                ),
                field(
                    "chain_storage",
                    ConfigFieldKind::OptionalObject,
                    false,
                    Some("None"),
                    "Optional chain storage helper object.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for ESSConfig {
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
                    "moves",
                    ConfigFieldKind::ObjectSequence,
                    false,
                    Some("None"),
                    "Optional list of ESS move helper objects.",
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
                field(
                    "chain_storage",
                    ConfigFieldKind::OptionalObject,
                    false,
                    Some("None"),
                    "Optional chain storage helper object.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for DifferentialEvolutionConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "DifferentialEvolutionConfig",
            version: 1,
            fields: vec![
                field(
                    "x0",
                    ConfigFieldKind::VectorFloat,
                    true,
                    None,
                    "Initial parameter vector.",
                ),
                field(
                    "population_size",
                    ConfigFieldKind::OptionalInteger,
                    false,
                    Some("None"),
                    "Optional explicit population size.",
                ),
                field(
                    "differential_weight",
                    ConfigFieldKind::Float,
                    false,
                    Some("0.8"),
                    "Mutation differential weight.",
                ),
                field(
                    "crossover_probability",
                    ConfigFieldKind::Float,
                    false,
                    Some("0.9"),
                    "Binomial crossover probability.",
                ),
                field(
                    "initial_scale",
                    ConfigFieldKind::Float,
                    false,
                    Some("1.0"),
                    "External-coordinate initialization half-width.",
                ),
                field(
                    "bounds",
                    ConfigFieldKind::Bounds,
                    false,
                    Some("None"),
                    "Optional lower and upper bounds per parameter.",
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

impl HasPyConfigSchema for CMAESConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "CMAESConfig",
            version: 1,
            fields: vec![
                field(
                    "x0",
                    ConfigFieldKind::VectorFloat,
                    true,
                    None,
                    "Initial mean vector.",
                ),
                field(
                    "sigma",
                    ConfigFieldKind::Float,
                    true,
                    None,
                    "Global step size.",
                ),
                field(
                    "population_size",
                    ConfigFieldKind::OptionalInteger,
                    false,
                    Some("None"),
                    "Optional explicit offspring population size.",
                ),
                field(
                    "bounds",
                    ConfigFieldKind::Bounds,
                    false,
                    Some("None"),
                    "Optional lower and upper bounds per parameter.",
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

impl HasPyConfigSchema for SimulatedAnnealingConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "SimulatedAnnealingConfig",
            version: 1,
            fields: vec![
                field(
                    "initial_temperature",
                    ConfigFieldKind::Float,
                    false,
                    Some("1.0"),
                    "Initial simulated annealing temperature.",
                ),
                field(
                    "cooling_rate",
                    ConfigFieldKind::Float,
                    false,
                    Some("0.999"),
                    "Multiplicative cooling factor per step.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for AdamConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "AdamConfig",
            version: 1,
            fields: vec![
                field(
                    "x0",
                    ConfigFieldKind::VectorFloat,
                    true,
                    None,
                    "Initial parameter vector.",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
                field(
                    "alpha",
                    ConfigFieldKind::Float,
                    false,
                    Some("0.001"),
                    "Initial learning rate.",
                ),
                field(
                    "beta_1",
                    ConfigFieldKind::Float,
                    false,
                    Some("0.9"),
                    "First-moment decay rate.",
                ),
                field(
                    "beta_2",
                    ConfigFieldKind::Float,
                    false,
                    Some("0.999"),
                    "Second-moment decay rate.",
                ),
                field(
                    "epsilon",
                    ConfigFieldKind::Float,
                    false,
                    Some("1e-8"),
                    "Divide-by-zero tolerance.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for ConjugateGradientConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "ConjugateGradientConfig",
            version: 1,
            fields: vec![
                field(
                    "x0",
                    ConfigFieldKind::VectorFloat,
                    true,
                    None,
                    "Initial parameter vector.",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
                field(
                    "line_search",
                    ConfigFieldKind::OptionalObject,
                    false,
                    Some("None"),
                    "Optional strong-Wolfe line-search helper object.",
                ),
                field(
                    "update",
                    ConfigFieldKind::String,
                    false,
                    Some("None"),
                    "Optional conjugate-gradient update rule.",
                ),
            ],
        }
    }
}

impl HasPyConfigSchema for TrustRegionConfig {
    fn schema() -> ConfigSchema {
        ConfigSchema {
            name: "TrustRegionConfig",
            version: 1,
            fields: vec![
                field(
                    "x0",
                    ConfigFieldKind::VectorFloat,
                    true,
                    None,
                    "Initial parameter vector.",
                ),
                field(
                    "parameter_names",
                    ConfigFieldKind::ParameterNames,
                    false,
                    Some("None"),
                    "Optional parameter names propagated into summaries.",
                ),
                field(
                    "subproblem",
                    ConfigFieldKind::String,
                    false,
                    Some("None"),
                    "Optional trust-region subproblem solver.",
                ),
                field(
                    "initial_radius",
                    ConfigFieldKind::Float,
                    false,
                    Some("1.0"),
                    "Initial trust-region radius.",
                ),
                field(
                    "max_radius",
                    ConfigFieldKind::Float,
                    false,
                    Some("1000.0"),
                    "Maximum trust-region radius.",
                ),
                field(
                    "eta",
                    ConfigFieldKind::Float,
                    false,
                    Some("1e-4"),
                    "Acceptance threshold.",
                ),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lbfgsb_schema_contains_python_fields() {
        let schema = LBFGSBConfig::schema();
        assert_eq!(schema.name, "LBFGSBConfig");
        assert!(schema.fields.iter().any(|field| field.name == "x0"));
        assert!(schema
            .fields
            .iter()
            .any(|field| field.name == "memory_limit"));
    }

    #[test]
    fn cmaes_schema_exports_to_json() {
        let json = CMAESConfig::schema_json_pretty().unwrap();
        assert!(json.contains("\"name\": \"CMAESConfig\""));
        assert!(json.contains("\"sigma\""));
    }
}
