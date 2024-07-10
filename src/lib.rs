use nalgebra::{ComplexField, DMatrix, DVector};
use num_traits::Float;

pub trait Field: Float + ComplexField + std::iter::Sum {}
impl Field for f32 {}
impl Field for f64 {}

pub trait Function<F, Error>
where
    F: Field + 'static,
{
    fn evaluate(&self, x: &[F]) -> Result<F, Error>;

    fn gradient(&self, x: &[F]) -> Result<DVector<F>, Error> {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        let h = ComplexField::cbrt(F::epsilon());

        for i in 0..n {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            x_plus[i] += h;
            x_minus[i] -= h;
            let f_plus = self.evaluate(&x_plus)?;
            let f_minus = self.evaluate(&x_minus)?;
            grad[i] = (f_plus - f_minus) / (F::from(2.0).unwrap() * h);
        }

        Ok(grad)
    }

    fn gradient_and_hessian(&self, x: &[F]) -> Result<(DVector<F>, DMatrix<F>), Error> {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        let mut hess = DMatrix::zeros(n, n);
        let h = ComplexField::cbrt(F::epsilon());
        let two = F::from(2.0).unwrap();
        let four = two * two;

        // Compute Hessian
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal element using central difference
                    let mut x_plus = x.to_vec();
                    let mut x_minus = x.to_vec();
                    x_plus[i] += h;
                    x_minus[i] -= h;

                    let f_plus = self.evaluate(&x_plus)?;
                    let f_minus = self.evaluate(&x_minus)?;
                    let f_center = self.evaluate(x)?;

                    grad[i] = (f_plus - f_minus) / (two * h);
                    hess[(i, i)] = (f_plus - two * f_center + f_minus) / (h * h);
                } else {
                    // Off-diagonal element
                    let mut x_plus_plus = x.to_vec();
                    let mut x_plus_minus = x.to_vec();
                    let mut x_minus_plus = x.to_vec();
                    let mut x_minus_minus = x.to_vec();

                    x_plus_plus[i] += h;
                    x_plus_plus[j] += h;
                    x_plus_minus[i] += h;
                    x_plus_minus[j] -= h;
                    x_minus_plus[i] -= h;
                    x_minus_plus[j] += h;
                    x_minus_minus[i] -= h;
                    x_minus_minus[j] -= h;

                    let f_plus_plus = self.evaluate(&x_plus_plus)?;
                    let f_plus_minus = self.evaluate(&x_plus_minus)?;
                    let f_minus_plus = self.evaluate(&x_minus_plus)?;
                    let f_minus_minus = self.evaluate(&x_minus_minus)?;

                    hess[(i, j)] = (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus)
                        / (four * h * h);
                    hess[(j, i)] = hess[(i, j)];
                }
            }
        }

        Ok((grad, hess))
    }
}

pub trait Minimizer<F, M, E>
where
    F: Field,
{
    fn step(&mut self) -> Result<M, E>;
    fn callback(&self) -> Result<(), E> {
        Ok(())
    }
    fn terminate(&self) -> bool;
    fn minimize<Func: Fn(M)>(&mut self, callback: Func) -> Result<(), E>;
    fn best(&self) -> (Vec<F>, F);
}

pub mod algorithms;
pub mod test_functions;

pub mod prelude {
    pub use crate::Field;
    pub use crate::{Function, Minimizer};
}
