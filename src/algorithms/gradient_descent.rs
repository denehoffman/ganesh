use nalgebra::ComplexField;
use typed_builder::TypedBuilder;

use crate::{Field, Function, Minimizer};

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
pub struct GradientDescent<F, E>
where
    F: Field,
{
    function: Box<dyn Function<F, E>>,
    options: GradientDescentOptions<F>,
    x: Vec<F>,
    fx: F,
    x_best: Vec<F>,
    fx_best: F,
}
impl<F, E> GradientDescent<F, E>
where
    F: Field,
{
    pub fn new<Func: Function<F, E> + 'static>(
        function: Func,
        x0: &[F],
        options: GradientDescentOptions<F>,
    ) -> Self {
        Self {
            function: Box::new(function),
            options,
            x: x0.to_vec(),
            fx: F::infinity(),
            x_best: vec![F::infinity(); x0.len()],
            fx_best: F::infinity(),
        }
    }
}
pub enum GradientDescentMessage<F> {
    Continue { x: Vec<F>, fx: F },
}
impl<F, E> Minimizer<F, GradientDescentMessage<F>, E> for GradientDescent<F, E>
where
    F: Field,
{
    fn step(&mut self) -> Result<GradientDescentMessage<F>, E> {
        let gradient = self.function.gradient(&self.x)?;
        self.x
            .iter_mut()
            .zip(gradient.as_slice())
            .for_each(|(x, dx)| *x -= self.options.learning_rate * *dx);
        self.fx = self.function.evaluate(&self.x)?;
        Ok(GradientDescentMessage::Continue {
            x: self.x.clone(),
            fx: self.fx,
        })
    }

    fn terminate(&self) -> bool {
        false
    }

    fn minimize<Func: Fn(GradientDescentMessage<F>)>(&mut self, callback: Func) -> Result<(), E> {
        for _ in 0..self.options.max_iters {
            let m = self.step()?;
            if self.fx < self.fx_best {
                self.x_best = self.x.clone();
                self.fx_best = self.fx;
            }
            callback(m);
            if self.terminate() {
                return Ok(());
            }
        }
        Ok(())
    }

    fn best(&self) -> (Vec<F>, F) {
        (self.x_best.clone(), self.fx_best)
    }
}
