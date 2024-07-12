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
pub struct Newton<F, E>
where
    F: Field,
{
    function: Box<dyn Function<F, E>>,
    options: NewtonOptions<F>,
    x: Vec<F>,
    fx: F,
    x_best: Vec<F>,
    fx_best: F,
}
impl<F, E> Newton<F, E>
where
    F: Field,
{
    pub fn new<Func: Function<F, E> + 'static>(
        function: Func,
        x0: &[F],
        options: NewtonOptions<F>,
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
pub enum NewtonMessage<F> {
    Continue { x: Vec<F>, fx: F },
    SingularHessian { x: Vec<F>, fx: F },
}
impl<F, E> Minimizer<F, NewtonMessage<F>, E> for Newton<F, E>
where
    F: Field,
{
    fn step(&mut self) -> Result<NewtonMessage<F>, E> {
        let (gradient, hessian) = self.function.gradient_and_hessian(&self.x)?;
        match hessian.lu().solve(&-gradient) {
            Some(sol) => {
                self.x
                    .iter_mut()
                    .zip(sol.as_slice())
                    .for_each(|(x, dx)| *x += self.options.step_size * *dx);
                self.fx = self.function.evaluate(&self.x)?;
                Ok(NewtonMessage::Continue {
                    x: self.x.clone(),
                    fx: self.fx,
                })
            }
            None => Ok(NewtonMessage::SingularHessian {
                x: self.x.clone(),
                fx: self.fx,
            }),
        }
    }

    fn terminate(&self) -> bool {
        false
    }

    fn minimize<Func: Fn(NewtonMessage<F>)>(&mut self, callback: Func) -> Result<(), E> {
        for _ in 0..self.options.max_iters {
            let m = self.step()?;
            match m {
                NewtonMessage::Continue { x: _, fx: _ } => {}
                NewtonMessage::SingularHessian { x: _, fx: _ } => {
                    println!("Hessian is Singular!");
                    return Ok(());
                }
            }
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
