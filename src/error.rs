use thiserror::Error;

/// A general result type for the `ganesh` crate
pub type GaneshResult<T> = Result<T, GaneshError>;

/// A general error type for the `ganesh` crate
#[derive(Error, Debug)]
pub enum GaneshError {
    /// Variant for errors which may occur while configuring an [`Algorithm`](crate::traits::algorithm::Algorithm)
    #[error("Configuration error: {0}")]
    ConfigError(String),
    /// Variant for numerical failures such as singular solves or invalid covariance factorizations
    #[error("Numerical error: {0}")]
    NumericalError(String),
}
