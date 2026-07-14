use std::{
    borrow::Cow,
    fmt::{Display, Write},
    ops::ControlFlow,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An enum indicating the status of an [`Algorithm`](crate::traits::Algorithm)
#[non_exhaustive]
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub enum StatusType {
    /// Indicates the algorithm has not yet been initialized
    #[default]
    Uninitialized,
    /// Indicates the algorithm has been initialized
    Initialized,
    /// Indicates the algorithm has completed a step
    StepType,
    /// Indicates the algorithm has succeeded
    Success,
    /// Indicates the algorithm has failed
    Failed,
    /// Custom message
    Custom,
}

impl Display for StatusType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Uninitialized => "Uninitialized",
                Self::Initialized => "Initialized",
                Self::StepType => "Step",
                Self::Success => "Success",
                Self::Failed => "Failed",
                Self::Custom => "Message",
            }
        )
    }
}

/// A status message for an [`Algorithm`](crate::traits::Algorithm)
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct StatusMessage {
    /// The status type
    pub status_type: StatusType,
    /// The internal custom message
    #[serde(default, with = "optional_cow_static_str")]
    pub text: Option<Cow<'static, str>>,
}
impl StatusMessage {
    /// Reset the status message to uninitialized
    pub fn reset(&mut self) {
        self.status_type = StatusType::Uninitialized;
        self.text = None;
    }
    /// Set the status message to uninitialized
    pub fn uninitialize(&mut self) {
        self.reset()
    }
    /// Set the status message to initialized
    pub fn initialize(&mut self) {
        self.reset();
        self.status_type = StatusType::Initialized;
    }
    /// Set the status message to initialized
    pub fn set_initialized(mut self) -> Self {
        self.initialize();
        self
    }
    /// Set the status message to initialized with a custom message
    pub fn initialize_with_message(&mut self, message: impl Into<Cow<'static, str>>) {
        self.initialize();
        self.text = Some(message.into());
    }
    /// Set the status message to initialized with a custom message
    pub fn set_initialized_with_message(mut self, message: impl Into<Cow<'static, str>>) -> Self {
        self.initialize_with_message(message);
        self
    }
    /// Set the status message to a step
    pub fn step(&mut self) {
        self.reset();
        self.status_type = StatusType::StepType;
    }
    /// Set the status message to a step
    pub fn set_step(mut self) -> Self {
        self.step();
        self
    }
    /// Set the status message to a step with a custom message
    pub fn step_with_message(&mut self, message: impl Into<Cow<'static, str>>) {
        self.step();
        self.text = Some(message.into());
    }
    /// Set the status message to a step with a custom message
    pub fn set_step_with_message(mut self, message: impl Into<Cow<'static, str>>) -> Self {
        self.step_with_message(message);
        self
    }
    /// Set the status message to success
    pub fn succeed(&mut self) {
        self.reset();
        self.status_type = StatusType::Success;
    }
    /// Set the status message to success
    pub fn set_success(mut self) -> Self {
        self.succeed();
        self
    }
    /// Set the status message to converged with a custom message
    pub fn succeed_with_message(&mut self, message: impl Into<Cow<'static, str>>) {
        self.succeed();
        self.text = Some(message.into());
    }
    /// Set the status message to converged with a custom message
    pub fn set_success_with_message(mut self, message: impl Into<Cow<'static, str>>) -> Self {
        self.succeed_with_message(message);
        self
    }
    /// Set the status message to failed
    pub fn fail(&mut self) {
        self.reset();
        self.status_type = StatusType::Failed;
    }
    /// Set the status message to failed
    pub fn set_failed(mut self) -> Self {
        self.fail();
        self
    }
    /// Set the status message to failed with a custom message
    pub fn fail_with_message(&mut self, message: impl Into<Cow<'static, str>>) {
        self.fail();
        self.text = Some(message.into());
    }
    /// Set the status message to failed with a custom message
    pub fn set_failed_with_message(mut self, message: impl Into<Cow<'static, str>>) -> Self {
        self.fail_with_message(message);
        self
    }
    /// Set the status message to a custom message
    pub fn custom(&mut self, message: impl Into<Cow<'static, str>>) {
        self.reset();
        self.status_type = StatusType::Custom;
        self.text = Some(message.into());
    }
    /// Set the status message to a custom message.
    pub fn custom_with_message(&mut self, message: impl Into<Cow<'static, str>>) {
        self.custom(message);
    }
    /// Set the status message to a custom message
    pub fn set_custom(mut self, message: impl Into<Cow<'static, str>>) -> Self {
        self.custom(message);
        self
    }
    /// Returns the message text payload, if one exists.
    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }
    /// Returns the message text payload, or an empty string for message-less status transitions.
    pub fn text_or_empty(&self) -> &str {
        self.text().unwrap_or("")
    }
    /// Returns `true` if the status message is [`StatusType::Success`]
    pub const fn success(&self) -> bool {
        matches!(self.status_type, StatusType::Success)
    }
}

impl Display for StatusMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.text() {
            Some(text) if !text.is_empty() => write!(f, "{}: {}", self.status_type, text),
            _ => write!(f, "{}", self.status_type),
        }
    }
}

mod optional_cow_static_str {
    use super::*;

    pub fn serialize<S>(text: &Option<Cow<'static, str>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        text.as_deref().unwrap_or("").serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Cow<'static, str>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<String>::deserialize(deserializer).map(|text| text.map(Cow::Owned))
    }
}

/// A trait which holds the status of a [`Algorithm`](crate::traits::Algorithm)
///
/// This must be implemented for own [`Algorithm`](crate::traits::Algorithm)s that need
/// different status information than the ones implemented in this crate.
pub trait Status: Clone + Default {
    /// Resets the status to its default state. This is called at the beginning of every
    /// [`Algorithm::process`](crate::traits::Algorithm::process) run. Only members that are
    /// not persistent between runs should be reset. For example, the initial parameters of
    /// a minimization should not be reset.
    fn reset(&mut self);
    /// Returns true if the algorithm has terminated successfully
    fn success(&self) -> bool {
        self.message().success()
    }
    /// Returns the message of the minimization.
    fn message(&self) -> &StatusMessage;
    /// Sets the message of the minimization.
    fn set_message(&mut self) -> &mut StatusMessage;
    /// Checks invariants that must hold for this status after initialization and each step.
    fn check_invariants(&mut self) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// A trait for statuses that can render a concise progress line without forcing per-step
/// allocations in the algorithm implementation.
pub trait ProgressStatus: Status {
    /// Write a status-specific progress payload into `out`.
    ///
    /// # Errors
    ///
    /// Returns a formatting error if writing into `out` fails.
    fn write_progress(&self, out: &mut String) -> std::fmt::Result {
        write!(out, "status={}", self.message())
    }
}
