use crate::{core::Callbacks, traits::Status};
use std::convert::Infallible;

/// A trait representing an [`Algorithm`] which can be used to solve a problem `P`.
///
/// This trait is implemented for the algorithms found in the [`algorithms`](`crate::algorithms`) module and contains
/// all the methods needed to [`process`](`Algorithm::process`) a problem.
pub trait Algorithm<P, S: Status, U = (), E = Infallible>: Send + Sync {
    /// A type which holds a summary of the algorithm's ending state.
    type Summary;
    /// The configuration struct for the algorithm.
    type Config;
    /// The initialization payload for a single run.
    type Init;

    /// Any setup work done before the main steps of the algorithm should be done here.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    fn initialize(
        &mut self,
        problem: &P,
        status: &mut S,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E>;
    /// The main "step" of an algorithm, which is repeated until termination conditions are met or
    /// the max number of steps have been taken.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    fn step(
        &mut self,
        current_step: usize,
        problem: &P,
        status: &mut S,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E>;

    /// Runs any steps needed by the [`Algorithm`] after termination or convergence. This will run
    /// regardless of whether the [`Algorithm`] converged.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    #[allow(unused_variables)]
    fn postprocessing(
        &mut self,
        problem: &P,
        status: &mut S,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Generates a new [`Algorithm::Summary`] from the current state of the [`Algorithm`], which can be displayed or used elsewhere.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    #[allow(unused_variables)]
    fn summarize(
        &self,
        current_step: usize,
        problem: &P,
        status: &S,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<Self::Summary, E>;

    /// Reset the algorithm to its initial state.
    fn reset(&mut self) {}

    /// Process the given problem using this [`Algorithm`].
    ///
    /// This method first runs [`Algorithm::initialize`], then runs [`Algorithm::step`] in a loop,
    /// terminating if any supplied [`Terminator`](crate::traits::Terminator)s return
    /// [`ControlFlow::Break`](`std::ops::ControlFlow::Break`). Finally, regardless of convergence,
    /// [`Algorithm::postprocessing`] is called. [`Algorithm::summarize`] is called to create a
    /// summary of the [`Algorithm`]'s state.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    fn process<I, C>(
        &mut self,
        problem: &P,
        args: &U,
        init: I,
        config: Self::Config,
        callbacks: C,
    ) -> Result<Self::Summary, E>
    where
        I: Into<Self::Init>,
        C: Into<Callbacks<Self, P, S, U, E, Self::Config>>,
        Self: Sized,
    {
        let init = init.into();
        let mut status = S::default();
        let mut cbs: Callbacks<Self, P, S, U, E, Self::Config> = callbacks.into();
        self.initialize(problem, &mut status, args, &init, &config)?;
        if status.check_invariants().is_break() {
            self.postprocessing(problem, &mut status, args, &config)?;
            return self.summarize(0, problem, &status, args, &init, &config);
        }
        let mut current_step = 0;
        loop {
            self.step(current_step, problem, &mut status, args, &config)?;

            if status.check_invariants().is_break() {
                break;
            }

            if cbs
                .check_for_termination(current_step, self, problem, &mut status, args, &config)
                .is_break()
            {
                break;
            }
            current_step += 1;
        }
        self.postprocessing(problem, &mut status, args, &config)?;
        self.summarize(current_step, problem, &status, args, &init, &config)
    }

    /// Process the given problem using this [`Algorithm`] and the algorithm's default callbacks.
    ///
    /// This method first runs [`Algorithm::initialize`], then runs [`Algorithm::step`] in a loop,
    /// terminating if any of the [`Algorithm::default_callbacks`] return
    /// [`ControlFlow::Break`](`std::ops::ControlFlow::Break`). Finally, regardless of convergence,
    /// [`Algorithm::postprocessing`] is called. [`Algorithm::summarize`] is called to create a
    /// summary of the [`Algorithm`]'s state.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    fn process_with_default_callbacks<I>(
        &mut self,
        problem: &P,
        user_data: &U,
        init: I,
        config: Self::Config,
    ) -> Result<Self::Summary, E>
    where
        I: Into<Self::Init>,
        Self: Sized,
    {
        self.process(problem, user_data, init, config, Self::default_callbacks())
    }

    /// Process the given problem using this [`Algorithm`] with default config and default callbacks.
    ///
    /// This method is similar to [`Algorithm::process`], except it uses
    /// [`Default::default`] for the algorithm configuration and
    /// [`Algorithm::default_callbacks`] for the callback set.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    fn process_default<I>(
        &mut self,
        problem: &P,
        user_data: &U,
        init: I,
    ) -> Result<Self::Summary, E>
    where
        I: Into<Self::Init>,
        Self: Sized,
        Self::Config: Default,
    {
        self.process(
            problem,
            user_data,
            init,
            Self::Config::default(),
            Self::default_callbacks(),
        )
    }

    /// Provides a set of reasonable default callbacks specific to this [`Algorithm`].
    fn default_callbacks() -> Callbacks<Self, P, S, U, E, Self::Config>
    where
        Self: Sized,
    {
        Callbacks::empty()
    }
}

/// A trait for algorithm configs which can propagate parameter names into summaries.
pub trait SupportsParameterNames
where
    Self: Sized,
{
    /// A helper method to get the mutable internal parameter name storage.
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>>;
    /// Set the names associated with each parameter.
    fn with_parameter_names<I, S>(mut self, parameter_names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        *self.get_parameter_names_mut() = Some(
            parameter_names
                .into_iter()
                .map(|name| name.as_ref().to_string())
                .collect(),
        );
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{StatusMessage, StatusType};
    use serde::{Deserialize, Serialize};
    use std::{convert::Infallible, ops::ControlFlow};

    #[derive(Clone, Default, Serialize, Deserialize)]
    struct InvariantStatus {
        message: StatusMessage,
        invalid: bool,
    }

    impl Status for InvariantStatus {
        fn reset(&mut self) {
            self.message.reset();
            self.invalid = false;
        }

        fn message(&self) -> &StatusMessage {
            &self.message
        }

        fn set_message(&mut self) -> &mut StatusMessage {
            &mut self.message
        }

        fn check_invariants(&mut self) -> ControlFlow<()> {
            if self.invalid {
                self.message.fail_with_message("invariant failed");
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        }
    }

    #[derive(Default)]
    struct InvariantAlgorithm {
        steps: usize,
    }

    #[derive(Clone, Copy)]
    struct InvariantConfig {
        fail_after_initialize: bool,
        fail_after_step: bool,
    }

    impl Algorithm<(), InvariantStatus, (), Infallible> for InvariantAlgorithm {
        type Summary = (usize, StatusMessage);
        type Config = InvariantConfig;
        type Init = ();

        fn initialize(
            &mut self,
            _problem: &(),
            status: &mut InvariantStatus,
            _args: &(),
            _init: &Self::Init,
            config: &Self::Config,
        ) -> Result<(), Infallible> {
            status.message.initialize();
            status.invalid = config.fail_after_initialize;
            Ok(())
        }

        fn step(
            &mut self,
            _current_step: usize,
            _problem: &(),
            status: &mut InvariantStatus,
            _args: &(),
            config: &Self::Config,
        ) -> Result<(), Infallible> {
            self.steps += 1;
            status.message.step();
            status.invalid = config.fail_after_step;
            Ok(())
        }

        fn summarize(
            &self,
            _current_step: usize,
            _problem: &(),
            status: &InvariantStatus,
            _args: &(),
            _init: &Self::Init,
            _config: &Self::Config,
        ) -> Result<Self::Summary, Infallible> {
            Ok((self.steps, status.message.clone()))
        }
    }

    #[test]
    fn process_checks_status_invariants_after_initialize() {
        let mut algorithm = InvariantAlgorithm::default();
        let config = InvariantConfig {
            fail_after_initialize: true,
            fail_after_step: false,
        };

        let (steps, message) = algorithm
            .process(&(), &(), (), config, Callbacks::empty())
            .unwrap();

        assert_eq!(steps, 0);
        assert!(matches!(message.status_type, StatusType::Failed));
        assert_eq!(message.text(), Some("invariant failed"));
    }

    #[test]
    fn process_checks_status_invariants_after_step() {
        let mut algorithm = InvariantAlgorithm::default();
        let config = InvariantConfig {
            fail_after_initialize: false,
            fail_after_step: true,
        };

        let (steps, message) = algorithm
            .process(&(), &(), (), config, Callbacks::empty())
            .unwrap();

        assert_eq!(steps, 1);
        assert!(matches!(message.status_type, StatusType::Failed));
        assert_eq!(message.text(), Some("invariant failed"));
    }
}
