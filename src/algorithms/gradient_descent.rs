use nalgebra::ComplexField;
use typed_builder::TypedBuilder;

use crate::core::{Field, Function, Minimizer};

/// Used to set options for the [`GradientDescent`] optimizer.
///
/// See also: [`GradientDescentOptions::builder()`]
#[derive(TypedBuilder, Debug)]
pub struct GradientDescentOptions<F: Field> {
    /// The distance to move in the descent direction (default = 1.0)
    #[builder(default = F::one())]
    pub learning_rate: F,
    /// The maximum number of steps to compute (default = 1000)
    #[builder(default = 1000)]
    pub max_iters: usize,
    /// The minimum absolute difference between evaluations that will terminate the
    /// algorithm (default = 1e-8)
    #[builder(default = F::convert(1e-8))]
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
/// This method will terminate if either [`GradientDescentOptions::max_iters`] steps are performed or if
/// $`|f(\vec{x}_{i}) - f(\vec{x}_{i-1})|`$ is smaller than [`GradientDescentOptions::tolerance`].
pub struct GradientDescent<F, A, E>
where
    F: Field,
{
    function: Box<dyn Function<F, A, E>>,
    options: GradientDescentOptions<F>,
    x: Vec<F>,
    fx: F,
    fx_old: F,
    x_best: Vec<F>,
    fx_best: F,
}
impl<F, A, E> GradientDescent<F, A, E>
where
    F: Field,
{
    /// Create a new Gradient Descent optimizer from a struct which implements [`Function`], an initial
    /// starting point `x0`, and some options.
    pub fn new<Func: Function<F, A, E> + 'static>(
        function: Func,
        x0: &[F],
        options: Option<GradientDescentOptions<F>>,
    ) -> Self {
        Self {
            function: Box::new(function),
            options: options.unwrap_or_else(|| GradientDescentOptions::builder().build()),
            x: x0.to_vec(),
            fx: F::nan(),
            fx_old: F::nan(),
            x_best: vec![F::nan(); x0.len()],
            fx_best: F::nan(),
        }
    }
}
/// A message passed into the [`GradientDescent::minimize`] callback.
pub struct GradientDescentMessage<F> {
    /// The current step number.
    pub step: usize,
    /// The current position of the minimizer.
    pub x: Vec<F>,
    /// The current value of the minimizer function.
    pub fx: F,
}

impl<F, A, E> Minimizer<F, A, GradientDescentMessage<F>, E> for GradientDescent<F, A, E>
where
    F: Field,
{
    fn step(&mut self, i: usize, args: &Option<A>) -> Result<GradientDescentMessage<F>, E> {
        self.fx_old = self.fx;
        let gradient = self.function.gradient(&self.x, args)?;
        self.x
            .iter_mut()
            .zip(gradient.as_slice())
            .for_each(|(x, dx)| *x -= self.options.learning_rate * *dx);
        self.fx = self.function.evaluate(&self.x, args)?;
        Ok(GradientDescentMessage {
            step: i,
            x: self.x.clone(),
            fx: self.fx,
        })
    }

    fn terminate(&self) -> bool {
        ComplexField::abs(self.fx - self.fx_old) <= ComplexField::abs(self.options.tolerance)
    }

    fn minimize<Func: Fn(&GradientDescentMessage<F>)>(
        &mut self,
        args: &Option<A>,
        callback: Func,
    ) -> Result<GradientDescentMessage<F>, E> {
        let mut m = GradientDescentMessage {
            step: 0,
            x: self.x.clone(),
            fx: self.fx,
        };
        for i in 0..self.options.max_iters {
            m = self.step(i, args)?;
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
