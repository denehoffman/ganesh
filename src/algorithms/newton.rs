use nalgebra::ComplexField;
use typed_builder::TypedBuilder;

use crate::{Field, Function, Minimizer};

/// Used to set options for the [`Newton`] optimizer.
///
/// See also: [`NewtonOptions::builder()`]
#[derive(TypedBuilder, Debug)]
pub struct NewtonOptions<F: Field> {
    /// The distance to move in the Newton direction (default = 1.0)
    #[builder(default = F::one())]
    pub step_size: F,
    /// The maximum number of steps to compute (default = 1000)
    #[builder(default = 1000)]
    pub max_iters: usize,
    /// The minimum absolute difference between evaluations that will terminate the
    /// algorithm (default = 1e-8)
    #[builder(default = F::convert(1e-8))]
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
/// This method will terminate if either [`NewtonOptions::max_iters`] steps are performed or if
/// $`|f(\vec{x}_{i}) - f(\vec{x}_{i-1})|`$ is smaller than [`NewtonOptions::tolerance`].
pub struct Newton<F, E>
where
    F: Field,
{
    function: Box<dyn Function<F, E>>,
    options: NewtonOptions<F>,
    x: Vec<F>,
    fx: F,
    fx_old: F,
    x_best: Vec<F>,
    fx_best: F,
}
impl<F, E> Newton<F, E>
where
    F: Field,
{
    /// Create a new Newton optimizer from a struct which implements [`Function`], an initial
    /// starting point `x0`, and some options.
    pub fn new<Func: Function<F, E> + 'static>(
        function: Func,
        x0: &[F],
        options: Option<NewtonOptions<F>>,
    ) -> Self {
        Self {
            function: Box::new(function),
            options: options.unwrap_or_else(|| NewtonOptions::builder().build()),
            x: x0.to_vec(),
            fx: F::nan(),
            fx_old: F::nan(),
            x_best: vec![F::nan(); x0.len()],
            fx_best: F::nan(),
        }
    }
}

/// A message passed into the [`Newton::minimize`] callback.
#[derive(Clone)]
pub struct NewtonMessage<F> {
    /// The current step number.
    pub step: usize,
    /// The current position of the minimizer.
    pub x: Vec<F>,
    /// The current value of the minimizer function.
    pub fx: F,
    /// Whether the Hessian was invertable or not
    pub singular_hessian: bool,
}

impl<F, E> Minimizer<F, NewtonMessage<F>, E> for Newton<F, E>
where
    F: Field,
{
    fn step(&mut self, i: usize) -> Result<NewtonMessage<F>, E> {
        self.fx_old = self.fx;
        let (gradient, hessian) = self.function.gradient_and_hessian(&self.x)?;
        match hessian.lu().solve(&-gradient) {
            Some(sol) => {
                self.x
                    .iter_mut()
                    .zip(sol.as_slice())
                    .for_each(|(x, dx)| *x += self.options.step_size * *dx);
                self.fx = self.function.evaluate(&self.x)?;
                Ok(NewtonMessage {
                    step: i,
                    x: self.x.clone(),
                    fx: self.fx,
                    singular_hessian: false,
                })
            }
            None => Ok(NewtonMessage {
                step: i,
                x: self.x.clone(),
                fx: self.fx,
                singular_hessian: true,
            }),
        }
    }

    fn terminate(&self) -> bool {
        ComplexField::abs(self.fx - self.fx_old) <= ComplexField::abs(self.options.tolerance)
    }

    fn minimize<Func: Fn(&NewtonMessage<F>)>(
        &mut self,
        callback: Func,
    ) -> Result<NewtonMessage<F>, E> {
        let mut m = NewtonMessage {
            step: 0,
            x: self.x.clone(),
            fx: self.fx,
            singular_hessian: false,
        };
        for i in 0..self.options.max_iters {
            m = self.step(i)?;
            if self.fx < self.fx_best {
                self.x_best = self.x.clone();
                self.fx_best = self.fx;
            }
            callback(&m);
            if self.terminate() {
                return Ok(m);
            }
        }
        Ok(m)
    }

    fn best(&self) -> (Vec<F>, F) {
        (self.x_best.clone(), self.fx_best)
    }
}
