use core::f32;

use nalgebra::{DVector, RealField};
use typed_builder::TypedBuilder;

use crate::{
    algorithms::line_search::TwoWayBacktrackingLineSearch,
    core::{Function, LineSearch, Minimizer},
};

/// Used to set options for the [`GradientDescent`] optimizer.
///
/// See also: [`GradientDescentOptions::builder()`]
#[derive(TypedBuilder, Debug)]
pub struct GradientDescentOptions<F, A, E>
where
    F: From<f32> + RealField + Copy,
{
    /// The line search algorithm to use for gradient descent. See
    /// [`algorithms::line_search`](`crate::algorithms::line_search`)
    /// for options. (default [`TwoWayBacktrackingLineSearch`]).
    #[builder(default = Box::new(TwoWayBacktrackingLineSearch::builder().build()))]
    pub line_search_method: Box<dyn LineSearch<F, A, E>>,
    /// The minimum absolute difference between evaluations that will terminate the
    /// algorithm (default = 1e-8)
    #[builder(default = F::from(1e-8))]
    pub tolerance: F,
}

/// Gradient Descent
///
/// This method uses the gradient of a function to determine the minimizing step direction:
///
/// ```math
/// \vec{x}_{i+1} = \vec{x}_i - \gamma g_f(\vec{x}_i)
/// ```
/// where $`g_f`$ is the gradient vector, and $`0 \lt \gamma`$ is the learning rate.
///
/// This method will terminate if $`|f(\vec{x}_{i}) - f(\vec{x}_{i-1})|`$ is smaller than
/// [`GradientDescentOptions::tolerance`].
pub struct GradientDescent<F, A, E>
where
    F: From<f32> + RealField + Copy,
{
    function: Box<dyn Function<F, A, E>>,
    options: GradientDescentOptions<F, A, E>,
    x: DVector<F>,
    x_old: Option<DVector<F>>,
    fx: F,
    fx_old: F,
    x_best: DVector<F>,
    fx_best: F,
    current_step: usize,
    learning_rate: Option<F>,
    g_old: Option<DVector<F>>,
}
impl<F, A, E> GradientDescent<F, A, E>
where
    F: From<f32> + RealField + Copy,
{
    /// Create a new Gradient Descent optimizer from a struct which implements [`Function`], an initial
    /// starting point `x0`, and some options.
    pub fn new<Func: Function<F, A, E> + 'static>(
        function: Func,
        x0: &[F],
        options: Option<GradientDescentOptions<F, A, E>>,
    ) -> Self {
        Self {
            function: Box::new(function),
            options: options.unwrap_or_else(|| GradientDescentOptions::builder().build()),
            x: DVector::from_row_slice(x0),
            x_old: None,
            fx: F::from(f32::NAN),
            fx_old: F::from(f32::NAN),
            x_best: DVector::from_element(x0.len(), F::from(f32::NAN)),
            fx_best: F::from(f32::INFINITY),
            current_step: 0,
            learning_rate: None,
            g_old: None,
        }
    }
}

impl<F, A, E> Minimizer<F, A, E> for GradientDescent<F, A, E>
where
    F: From<f32> + RealField + Copy,
{
    fn initialize(&mut self, _args: Option<&A>) -> Result<(), E> {
        self.learning_rate = Some(self.options.line_search_method.get_base_learning_rate());
        Ok(())
    }
    fn step(&mut self, args: Option<&A>) -> Result<(), E> {
        self.current_step += 1;
        self.fx_old = self.fx;
        let p = -self.function.gradient(&self.x, args)?;
        let step_and_rate = self.options.line_search_method.search(
            &*self.function,
            args,
            &self.x,
            &self.x_old,
            &p,
            &self.g_old,
            self.learning_rate,
        )?;
        self.x_old = Some(self.x.clone_owned()); // TODO: memory swap here?
        self.g_old = Some(p);
        self.x = step_and_rate.0;
        self.fx = step_and_rate.1;
        self.learning_rate = Some(step_and_rate.2);
        Ok(())
    }

    fn check_for_termination(&self) -> bool {
        (self.fx - self.fx_old).abs() <= self.options.tolerance.abs()
    }

    fn best(&self) -> (&DVector<F>, &F) {
        (&self.x_best, &self.fx_best)
    }

    fn update_best(&mut self) {
        if self.fx < self.fx_best {
            self.x_best = self.x.clone();
            self.fx_best = self.fx;
        }
    }
}
