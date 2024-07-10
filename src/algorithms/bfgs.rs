// TODO: incomplete
use derive_builder::Builder;
use nalgebra::DMatrix;
use num_traits::Zero;

use crate::{Field, Function, Minimizer};

#[derive(Builder)]
pub struct BFGSOptions<F: Field> {
    #[builder(default = "F::one()")]
    step_size: F,
    #[builder(default = "1000")]
    max_iters: usize,
}

pub struct BFGS<F, E>
where
    F: Field,
{
    function: Box<dyn Function<F, E>>,
    options: BFGSOptions<F>,
    x: Vec<F>,
    fx: F,
    b_mat: DMatrix<F>,
    x_best: Vec<F>,
    fx_best: F,
}
impl<F, E> BFGS<F, E>
where
    F: Field,
{
    pub fn new<Func: Function<F, E> + 'static>(
        function: Func,
        x0: &[F],
        options: BFGSOptions<F>,
    ) -> Self {
        Self {
            function: Box::new(function),
            options,
            x: x0.to_vec(),
            fx: F::infinity(),
            b_mat: DMatrix::zeros(x0.len(), x0.len()),
            x_best: vec![F::infinity(); x0.len()],
            fx_best: F::infinity(),
        }
    }
}
pub enum BFGSMessage<F> {
    Continue { x: Vec<F>, fx: F },
    SingularHessian { x: Vec<F>, fx: F },
}
impl<F, E> Minimizer<F, BFGSMessage<F>, E> for BFGS<F, E>
where
    F: Field,
{
    fn step(&mut self) -> Result<BFGSMessage<F>, E> {
        let (gradient, hessian) = self.function.gradient_and_hessian(&self.x)?;
        match hessian.lu().solve(&-gradient) {
            Some(sol) => {
                self.x
                    .iter_mut()
                    .zip(sol.as_slice())
                    .for_each(|(x, dx)| *x += self.options.step_size * *dx);
                self.fx = self.function.evaluate(&self.x)?;
                Ok(BFGSMessage::Continue {
                    x: self.x.clone(),
                    fx: self.fx,
                })
            }
            None => Ok(BFGSMessage::SingularHessian {
                x: self.x.clone(),
                fx: self.fx,
            }),
        }
    }

    fn terminate(&self) -> bool {
        false
    }

    fn minimize<Func: Fn(BFGSMessage<F>)>(&mut self, callback: Func) -> Result<(), E> {
        for _ in 0..self.options.max_iters {
            let m = self.step()?;
            match m {
                BFGSMessage::Continue { x: _, fx: _ } => {}
                BFGSMessage::SingularHessian { x: _, fx: _ } => {
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
