use crate::traits::{Algorithm, Observer, ProgressStatus, Status, Terminator};
use std::{
    fmt::{Debug, Write},
    ops::ControlFlow,
};

enum CallbackLike<A, P, S, U, E, C> {
    Terminator(Box<dyn Terminator<A, P, S, U, E, C>>),
    Observer(Box<dyn Observer<A, P, S, U, E, C>>),
}
impl<A, P, S, U, E, C> Clone for CallbackLike<A, P, S, U, E, C> {
    fn clone(&self) -> Self {
        match self {
            Self::Terminator(t) => Self::Terminator(dyn_clone::clone_box(&**t)),
            Self::Observer(o) => Self::Observer(dyn_clone::clone_box(&**o)),
        }
    }
}
impl<A, P, S, U, E, C> CallbackLike<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn callback(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        args: &U,
        config: &C,
    ) -> ControlFlow<()> {
        match self {
            Self::Terminator(terminator) => terminator.check_for_termination(
                current_step,
                algorithm,
                problem,
                status,
                args,
                config,
            ),
            Self::Observer(observer) => {
                observer.observe(current_step, algorithm, problem, status, args, config);
                ControlFlow::Continue(())
            }
        }
    }
}

/// A set of [`Terminator`]s and/or [`Observer`]s which can be used as an input to [`Algorithm::process`].
#[derive(Clone)]
pub struct Callbacks<A, P, S, U, E, C>(Vec<CallbackLike<A, P, S, U, E, C>>);
impl<A, P, S, U, E, C> Callbacks<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    /// Create an empty set of callbacks.
    pub const fn empty() -> Self {
        Self(Vec::new())
    }

    /// Return the set of [`Callbacks`] with an additional [`Terminator`] added.
    pub fn with_terminator<T>(mut self, terminator: T) -> Self
    where
        T: Terminator<A, P, S, U, E, C> + 'static,
    {
        self.0.push(CallbackLike::Terminator(Box::new(terminator)));
        self
    }

    /// Return the set of [`Callbacks`] with an additional [`Observer`] added.
    pub fn with_observer<O>(mut self, observer: O) -> Self
    where
        O: Observer<A, P, S, U, E, C> + 'static,
    {
        self.0.push(CallbackLike::Observer(Box::new(observer)));
        self
    }
    /// Runs all of the contained [`Terminator`]s and [`Observer`]s and returns [`ControlFlow::Break`] if any of the terminators return [`ControlFlow::Break`].
    pub fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        args: &U,
        config: &C,
    ) -> ControlFlow<()> {
        if self.0.iter_mut().any(|callback| {
            callback
                .callback(current_step, algorithm, problem, status, args, config)
                .is_break()
        }) {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// A [`Terminator`] which terminates the algorithm after a number of steps.
#[derive(Copy, Clone)]
pub struct MaxSteps(pub usize);
impl Default for MaxSteps {
    fn default() -> Self {
        Self(4000)
    }
}
impl<A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for MaxSteps
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        status: &mut S,
        _args: &U,
        _config: &C,
    ) -> ControlFlow<()> {
        if current_step >= self.0.saturating_sub(1) {
            status
                .set_message()
                .custom(&format!("Maximum number of steps reached ({})", self.0));
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

/// A debugging callback which prints out the step, status, and any user data at the current step
/// in an algorithm.
///
/// # Usage:
///
/// ```rust
/// use ganesh::traits::*;
/// use ganesh::algorithms::gradient_free::{nelder_mead::NelderMeadInit, NelderMead, NelderMeadConfig};
/// use ganesh::test_functions::Rosenbrock;
/// use ganesh::core::DebugObserver;
///
/// let problem = Rosenbrock { n: 2 };
/// let mut nm = NelderMead::default();
/// let init = NelderMeadInit::new([2.3, 3.4]);
/// let config = NelderMeadConfig::default();
/// let result = nm
///     .process(
///         &problem,
///         &(),
///         init,
///         config,
///         NelderMead::default_callbacks().with_observer(DebugObserver),
///     )
///     .unwrap();
/// // ^ This will print debug messages for each step
/// assert!(result.message.success());
/// ```
#[derive(Copy, Clone)]
pub struct DebugObserver;
impl<A, P, S, U, E, C> Observer<A, P, S, U, E, C> for DebugObserver
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status + Debug,
    U: Debug,
{
    fn observe(
        &mut self,
        current_step: usize,
        _algorithm: &A,
        _problem: &P,
        status: &S,
        _args: &U,
        _config: &C,
    ) {
        println!("Step: {}\n{:#?}", current_step, status);
    }
}

/// A lightweight observer which prints a concise one-line progress update every `interval` steps.
///
/// The emitted line uses the format
/// `step=<n> status=<status message>`.
#[derive(Clone)]
pub struct ProgressObserver {
    interval: usize,
    emitted_lines: usize,
    line_buffer: String,
}

impl ProgressObserver {
    /// Create a new [`ProgressObserver`] with the given reporting interval.
    pub fn new(interval: usize) -> Self {
        Self {
            interval: interval.max(1),
            emitted_lines: 0,
            line_buffer: String::new(),
        }
    }

    /// Return the configured reporting interval.
    pub const fn interval(&self) -> usize {
        self.interval
    }

    /// Return the number of progress lines emitted so far.
    pub const fn emitted_lines(&self) -> usize {
        self.emitted_lines
    }

    const fn should_emit(&self, current_step: usize) -> bool {
        current_step % self.interval == 0
    }
}

impl Default for ProgressObserver {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<A, P, S, U, E, C> Observer<A, P, S, U, E, C> for ProgressObserver
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: ProgressStatus,
{
    fn observe(
        &mut self,
        current_step: usize,
        _algorithm: &A,
        _problem: &P,
        status: &S,
        _args: &U,
        _config: &C,
    ) {
        if self.should_emit(current_step) {
            self.line_buffer.clear();
            let _ = write!(&mut self.line_buffer, "step={} ", current_step);
            let _ = status.write_progress(&mut self.line_buffer);
            println!("{}", self.line_buffer);
            self.emitted_lines += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{ProgressStatus, StatusMessage, StatusType};
    use serde::{Deserialize, Serialize};
    use std::{cell::RefCell, convert::Infallible, rc::Rc};

    #[derive(Clone, Default, Serialize, Deserialize)]
    struct DummyStatus {
        message: StatusMessage,
    }

    impl Status for DummyStatus {
        fn reset(&mut self) {
            self.message.reset();
        }

        fn message(&self) -> &StatusMessage {
            &self.message
        }

        fn set_message(&mut self) -> &mut StatusMessage {
            &mut self.message
        }
    }

    impl ProgressStatus for DummyStatus {}

    #[derive(Default)]
    struct DummyAlgorithm;

    impl Algorithm<(), DummyStatus, (), Infallible> for DummyAlgorithm {
        type Summary = ();
        type Config = ();
        type Init = ();

        fn initialize(
            &mut self,
            _problem: &(),
            status: &mut DummyStatus,
            _args: &(),
            _init: &Self::Init,
            _config: &Self::Config,
        ) -> Result<(), Infallible> {
            status.set_message().initialize();
            Ok(())
        }

        fn step(
            &mut self,
            current_step: usize,
            _problem: &(),
            status: &mut DummyStatus,
            _args: &(),
            _config: &Self::Config,
        ) -> Result<(), Infallible> {
            status
                .set_message()
                .step_with_message(&format!("step {}", current_step));
            Ok(())
        }

        fn summarize(
            &self,
            _current_step: usize,
            _problem: &(),
            _status: &DummyStatus,
            _args: &(),
            _init: &Self::Init,
            _config: &Self::Config,
        ) -> Result<Self::Summary, Infallible> {
            Ok(())
        }
    }

    #[test]
    fn progress_observer_formats_one_line_status() {
        let mut status = DummyStatus {
            message: StatusMessage {
                status_type: StatusType::StepType,
                text: "EXPAND".to_string(),
            },
        };
        status.set_message().step_with_message("EXPAND");
        let mut line = String::new();
        write!(&mut line, "step={} ", 7).unwrap();
        status.write_progress(&mut line).unwrap();

        assert_eq!(line, "step=7 status=Step: EXPAND");
    }

    #[test]
    fn progress_observer_respects_interval() {
        let observer = Rc::new(RefCell::new(ProgressObserver::new(2)));
        let mut algorithm = DummyAlgorithm;
        let callbacks = Callbacks::empty()
            .with_terminator(MaxSteps(5))
            .with_observer(observer.clone());

        algorithm.process(&(), &(), (), (), callbacks).unwrap();

        assert_eq!(observer.borrow().interval(), 2);
        assert_eq!(observer.borrow().emitted_lines(), 2);
    }
}
