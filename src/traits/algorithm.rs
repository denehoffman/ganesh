use crate::{
    core::{Bounds, Callbacks},
    traits::{Bound, Status, Transform, TransformExt},
};
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
    /// terminating if any supplied [`Terminator`]s return
    /// [`ControlFlow::Break`](`std::ops::ControlFlow::Break`). Finally, regardless of convergence,
    /// [`Algorithm::postprocessing`] is called. [`Algorithm::summarize`] is called to create a
    /// summary of the [`Algorithm`]'s state.
    ///
    /// # Errors
    ///
    /// Returns an `Err(E)` if any internal evaluation of the problem `P` fails.
    fn process<C>(
        &mut self,
        problem: &P,
        args: &U,
        init: Self::Init,
        config: Self::Config,
        callbacks: C,
    ) -> Result<Self::Summary, E>
    where
        C: Into<Callbacks<Self, P, S, U, E, Self::Config>>,
        Self: Sized,
    {
        let mut status = S::default();
        let mut cbs: Callbacks<Self, P, S, U, E, Self::Config> = callbacks.into();
        self.initialize(problem, &mut status, args, &init, &config)?;
        let mut current_step = 0;
        loop {
            self.step(current_step, problem, &mut status, args, &config)?;

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
    fn process_with_default_callbacks(
        &mut self,
        problem: &P,
        user_data: &U,
        init: Self::Init,
        config: Self::Config,
    ) -> Result<Self::Summary, E>
    where
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
    fn process_default(
        &mut self,
        problem: &P,
        user_data: &U,
        init: Self::Init,
    ) -> Result<Self::Summary, E>
    where
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

/// A trait which can be implemented on the configuration structs of [`Algorithm`](`crate::traits::Algorithm`)s to imply that the algorithm can be run with parameter bounds.
pub trait SupportsBounds
where
    Self: Sized,
{
    /// A helper method to get the mutable internal [`Bounds`] object.
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds>;
    /// Sets all [`Bound`]s used by the [`Algorithm`]. This can be [`None`] for an unbounded problem, or
    /// [`Some`] [`Vec<(T, T)>`] with length equal to the number of free parameters. Individual
    /// upper or lower bounds can be unbounded by setting them equal to `T::infinity()` or
    /// `T::neg_infinity()` (e.g. `f64::INFINITY` and `f64::NEG_INFINITY`).
    fn with_bounds<I: IntoIterator<Item = B>, B: Into<Bound>>(mut self, bounds: I) -> Self {
        let bounds = bounds
            .into_iter()
            .map(Into::into)
            .collect::<Vec<_>>()
            .into();
        *self.get_bounds_mut() = Some(bounds);
        self
    }
}

/// Explicit policy for how an algorithm should apply configured bounds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BoundsHandlingMode {
    /// Use the algorithm's default behavior.
    ///
    /// For algorithms with native bound support, this keeps bounds native. For algorithms without
    /// native bound support, callers should provide an explicit bounds transform instead.
    #[default]
    Auto,
    /// Keep bounds separate from any configured transform and use the algorithm's native
    /// bounded-space machinery.
    NativeBounds,
    /// Convert configured bounds into an explicit transform and run the algorithm without native
    /// bounds.
    ///
    /// If both a transform and bounds are configured, the transform is applied first and the
    /// bounds transform is applied second.
    TransformBounds,
}

pub(crate) fn resolve_bounds_and_transform(
    bounds: &Option<Bounds>,
    transform: &Option<Box<dyn Transform>>,
    mode: BoundsHandlingMode,
) -> (Option<Bounds>, Option<Box<dyn Transform>>) {
    match mode {
        BoundsHandlingMode::Auto | BoundsHandlingMode::NativeBounds => (
            bounds.clone(),
            transform
                .as_ref()
                .map(|transform| dyn_clone::clone_box(transform.as_ref())),
        ),
        BoundsHandlingMode::TransformBounds => {
            let resolved_transform = match (bounds, transform) {
                (Some(bounds), Some(transform)) => Some(Box::new(
                    dyn_clone::clone_box(transform.as_ref()).compose(bounds.clone()),
                ) as Box<dyn Transform>),
                (Some(bounds), None) => Some(Box::new(bounds.clone()) as Box<dyn Transform>),
                (None, Some(transform)) => Some(dyn_clone::clone_box(transform.as_ref())),
                (None, None) => None,
            };
            (None, resolved_transform)
        }
    }
}

/// A trait which can be implemented on the configuration structs of [`Algorithm`](`crate::traits::Algorithm`)s to imply that the algorithm can be run with parameter transformations.
pub trait SupportsTransform
where
    Self: Sized,
{
    /// A helper method to get the mutable internal [`Bounds`] object.
    fn get_transform_mut(&mut self) -> &mut Option<Box<dyn Transform>>;
    /// Set the transformation to apply to the parameter space.
    fn with_transform<T: Transform + 'static>(mut self, transform: &T) -> Self {
        *self.get_transform_mut() = Some(dyn_clone::clone_box(transform));
        self
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
    use crate::{DMatrix, DVector, Float};
    use std::borrow::Cow;

    #[derive(Clone)]
    struct Scale(Float);

    impl Transform for Scale {
        fn to_external<'a>(&'a self, z: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(z.scale(self.0))
        }

        fn to_internal<'a>(&'a self, x: &'a DVector<Float>) -> Cow<'a, DVector<Float>> {
            Cow::Owned(x.unscale(self.0))
        }

        fn to_external_jacobian(&self, z: &DVector<Float>) -> DMatrix<Float> {
            DMatrix::identity(z.len(), z.len()).scale(self.0)
        }

        fn to_external_component_hessian(&self, _a: usize, z: &DVector<Float>) -> DMatrix<Float> {
            DMatrix::zeros(z.len(), z.len())
        }
    }

    #[test]
    fn transform_bounds_mode_moves_bounds_into_transform() {
        let bounds = Some(Bounds::from([(0.0, 1.0)]));
        let transform: Option<Box<dyn Transform>> = Some(Box::new(Scale(2.0)));

        let (resolved_bounds, resolved_transform) =
            resolve_bounds_and_transform(&bounds, &transform, BoundsHandlingMode::TransformBounds);

        assert!(resolved_bounds.is_none());
        let resolved_transform = resolved_transform.expect("transform should be composed");
        let x = resolved_transform.to_owned_external(&DVector::from_row_slice(&[10.0]));
        assert!(x[0] >= 0.0 && x[0] <= 1.0);
    }

    #[test]
    fn native_bounds_mode_preserves_bounds_and_transform() {
        let bounds = Some(Bounds::from([(0.0, 1.0)]));
        let transform: Option<Box<dyn Transform>> = Some(Box::new(Scale(2.0)));

        let (resolved_bounds, resolved_transform) =
            resolve_bounds_and_transform(&bounds, &transform, BoundsHandlingMode::NativeBounds);

        assert!(resolved_bounds.is_some());
        assert!(resolved_transform.is_some());
    }
}
