use nalgebra::{ComplexField, DVector};
use num::Float;
use typed_builder::TypedBuilder;

use crate::core::{convert, Field, Function, Minimizer};

/// Used to set options for the [`Newton`] optimizer.
///
/// See also: [`NewtonOptions::builder()`]
#[derive(TypedBuilder)]
pub struct NewtonOptions<F>
where
    F: Field,
{
    /// The distance to move in the Newton direction (default = 1.0)
    #[builder(default = F::one())]
    pub step_size: F,
    /// The minimum absolute difference between evaluations that will terminate the
    /// algorithm (default = 1e-8)
    #[builder(default = convert!(1e-8, F))]
    pub tolerance: F,
}

/// Newton's method (Newton-Raphson).
///
/// Newton's method uses both the gradient and Hessian of a function to determine the minimizing
/// step direction:
///
/// ```math
/// \vec{x}_{i+1} = \vec{x}_i - \gamma [H_f(\vec{x}_i)]^{-1}\cdot g_f(\vec{x}_i)
/// ```
/// where $`H_f`$ and $`g_f`$ are the Hessian matrix and gradient vector, respectively, and
/// $`0 \lt \gamma \leq 1`$ is the step size (for $`\gamma \neq 1`$, this method is referred to as
/// the relaxed or damped Newton's method).
///
/// This method will terminate if either the maximum requested steps are performed or if
/// $`|f(\vec{x}_{i}) - f(\vec{x}_{i-1})|`$ is smaller than [`NewtonOptions::tolerance`].
pub struct Newton<F, A, E>
where
    F: Field,
{
    function: Box<dyn Function<F, A, E>>,
    options: NewtonOptions<F>,
    x: DVector<F>,
    fx: F,
    fx_old: F,
    x_best: DVector<F>,
    fx_best: F,
    current_step: usize,
    singular_hessian: bool,
}
impl<F, A, E> Newton<F, A, E>
where
    F: Field + 'static,
{
    /// Create a new Newton optimizer from a struct which implements [`Function`], an initial
    /// starting point `x0`, and some options.
    pub fn new<Func: Function<F, A, E> + 'static>(
        function: Func,
        x0: &[F],
        options: Option<NewtonOptions<F>>,
    ) -> Self {
        Self {
            function: Box::new(function),
            options: options.unwrap_or_else(|| NewtonOptions::builder().build()),
            x: DVector::from_row_slice(x0),
            fx: F::nan(),
            fx_old: F::nan(),
            x_best: DVector::from_element(x0.len(), F::nan()),
            fx_best: F::infinity(),
            current_step: 0,
            singular_hessian: false,
        }
    }
}

impl<F, A, E> Minimizer<F, A, E> for Newton<F, A, E>
where
    F: Field + ComplexField,
{
    fn step(&mut self, args: Option<&A>) -> Result<(), E> {
        self.current_step += 1;
        self.fx_old = self.fx;
        let (gradient, hessian) = self.function.gradient_and_hessian(&self.x, args)?;
        match hessian.lu().solve(&-gradient) {
            Some(sol) => {
                self.x
                    .iter_mut()
                    .zip(sol.as_slice())
                    .for_each(|(x, dx)| *x += self.options.step_size * *dx);
                self.fx = self.function.evaluate(&self.x, args)?;
            }
            None => self.singular_hessian = true,
        };
        Ok(())
    }

    fn check_for_termination(&self) -> bool {
        (<F as Float>::abs(self.fx - self.fx_old) <= <F as Float>::abs(self.options.tolerance))
            || self.singular_hessian
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
