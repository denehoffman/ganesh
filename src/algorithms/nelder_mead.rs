use nalgebra::ComplexField;
use typed_builder::TypedBuilder;

use crate::{Field, Function, Minimizer};

/// Used to set options for the [`NelderMead`] optimizer.
///
/// See also: [`NelderMeadOptions::builder()`]
#[derive(TypedBuilder, Debug)]
pub struct NelderMeadOptions<F: Field> {
    // TODO: validate coeffs, alpha > 0, gamma > 1, 0 < rho <= 0.5, sigma (0 < sigma < 1?)
    /// The step size from the starting point to each other point in the simplex (default = 1.0)
    #[builder(default = F::convert(1.0))]
    pub simplex_size: F,
    /// The coefficient $`\alpha > 0`$ to use in the reflection step (default = 1.0)
    #[builder(default = F::convert(1.0))]
    pub reflection_coeff: F,
    /// The coefficient $`\gamma > 1`$ to use in the expansion step (default = 2.0)
    #[builder(default = F::convert(2.0))]
    pub expansion_coeff: F,
    /// The coefficient $`0 < \rho_o \leq 0.5`$ to use in the contraction step
    /// in the case where the reflected point is better than the worst point in the simplex (default = 0.5)
    #[builder(default = F::convert(0.5))]
    pub outside_contraction_coeff: F,
    /// The coefficient $`0 < \rho_i \leq 0.5`$ to use in the contraction step
    /// in the case where the reflected point is worse than the worst point in the simplex (default = 0.5)
    #[builder(default = F::convert(0.5))]
    pub inside_contraction_coeff: F,
    /// The coefficient $`\sigma > 0`$ to use in the shrink step (default = 0.5)
    #[builder(default = F::convert(0.5))]
    pub shrink_coeff: F,
    /// If the standard deviation of the function at all points in the
    /// simplex falls below this value, the algorithm will terminate/converge (default=1e-8)
    #[builder(default = F::convert(1e-8))]
    pub min_simplex_standard_deviation: F,
    /// The maximum number of steps to compute (default = 1000)
    #[builder(default = 1000)]
    pub max_iters: usize,
}
pub struct NelderMead<F, E>
where
    F: Field,
{
    function: Box<dyn Function<F, E>>,
    options: NelderMeadOptions<F>,
    simplex_x: Vec<Vec<F>>,
    simplex_fx: Vec<F>,
    centroid_x: Vec<F>,
    centroid_fx: F,
    x_best: Vec<F>,
    fx_best: F,
}
impl<F, E> NelderMead<F, E>
where
    F: Field,
{
    pub fn new<Func: Function<F, E> + 'static>(
        function: Func,
        x0: &[F],
        options: NelderMeadOptions<F>,
    ) -> Self {
        let n_simplex = x0.len() + 1;
        let simplex_size = options.simplex_size;
        Self {
            function: Box::new(function),
            options,
            simplex_x: Self::construct_simplex(x0, n_simplex, simplex_size),
            simplex_fx: vec![F::infinity(); n_simplex],
            centroid_x: vec![F::infinity(); x0.len()],
            centroid_fx: F::infinity(),
            x_best: vec![F::infinity(); x0.len()],
            fx_best: F::infinity(),
        }
    }
    fn construct_simplex(x0: &[F], n_simplex: usize, simplex_size: F) -> Vec<Vec<F>> {
        (0..n_simplex)
            .map(|i| {
                if i == 0 {
                    x0.to_vec()
                } else {
                    let mut xi = x0.to_vec();
                    xi[i - 1] += simplex_size;
                    xi
                }
            })
            .collect()
    }
    fn evaluate_simplex(&mut self) -> Result<(), E> {
        let simplex_fx_res = self
            .simplex_x
            .iter()
            .map(|x| self.function.evaluate(x))
            .collect::<Result<Vec<F>, E>>()?;
        self.simplex_fx = simplex_fx_res;
        Ok(())
    }
    fn order_simplex(&mut self) {
        let mut indices: Vec<usize> = (0..self.simplex_fx.len()).collect();
        indices.sort_by(|&i, &j| self.simplex_fx[i].partial_cmp(&self.simplex_fx[j]).unwrap());
        let (sorted_simplex_x, sorted_simplex_fx): (Vec<Vec<F>>, Vec<F>) = indices
            .iter()
            .map(|&i| (self.simplex_x[i].clone(), self.simplex_fx[i]))
            .unzip();
        self.simplex_x = sorted_simplex_x;
        self.simplex_fx = sorted_simplex_fx;
    }
    fn calculate_centroid(&mut self) -> Result<(), E> {
        assert!(!self.simplex_x.is_empty(), "Simplex is empty!");
        let dim = self.simplex_x[0].len();
        let n_points = F::from(self.simplex_x.len() - 1).unwrap();
        self.centroid_x = (0..dim)
            .map(|i| {
                self.simplex_x
                    .iter()
                    .take(self.simplex_x.len() - 1)
                    .map(|point| point[i])
                    .sum::<F>()
                    / n_points
            })
            .collect();
        self.centroid_fx = self.function.evaluate(&self.centroid_x)?;
        Ok(())
    }
    fn reflect(&self) -> Vec<F> {
        self.centroid_x
            .iter()
            .zip(self.simplex_x.last().unwrap())
            .map(|(&c, &x)| c + self.options.reflection_coeff * (c - x))
            .collect()
    }
    fn expand(&self, x_r: &[F]) -> Vec<F> {
        self.centroid_x
            .iter()
            .zip(x_r)
            .map(|(&c, &x)| c + self.options.expansion_coeff * (x - c))
            .collect()
    }
    fn contract_outside(&self, x_r: &[F]) -> Vec<F> {
        self.centroid_x
            .iter()
            .zip(x_r)
            .map(|(&c, &x)| c + self.options.outside_contraction_coeff * (x - c))
            .collect()
    }
    fn contract_inside(&self) -> Vec<F> {
        let x_pivot = self.simplex_x.last().unwrap();
        self.centroid_x
            .iter()
            .zip(x_pivot)
            .map(|(&c, &x)| c + self.options.inside_contraction_coeff * (x - c))
            .collect()
    }
    fn shrink(&mut self) {
        let simplex_x_best = self.simplex_x[0].clone();
        self.simplex_x.iter_mut().skip(1).for_each(|xi| {
            *xi = xi
                .iter()
                .zip(&simplex_x_best)
                .map(|(&x, &x_best)| x_best + self.options.shrink_coeff * (x - x_best))
                .collect();
        })
    }
    fn replace_worst(&mut self, x: &[F]) -> Result<(), E> {
        let i_last = self.simplex_x.len() - 1;
        self.simplex_x[i_last] = x.to_vec();
        self.simplex_fx[i_last] = self.function.evaluate(x)?;
        Ok(())
    }
}
pub enum NelderMeadMessage {
    Continue,
    Terminate,
}
impl<F, E> Minimizer<F, NelderMeadMessage, E> for NelderMead<F, E>
where
    F: Field,
{
    fn step(&mut self) -> Result<NelderMeadMessage, E> {
        let fx_best = *self.simplex_fx.first().unwrap();
        let fx_second_worst = *self.simplex_fx.iter().nth_back(1).unwrap();
        let fx_worst = *self.simplex_fx.last().unwrap();
        let x_r = self.reflect();
        let fx_r = self.function.evaluate(&x_r)?;
        if fx_r < fx_best {
            let x_e = self.expand(&x_r);
            let fx_e = self.function.evaluate(&x_e)?;
            self.replace_worst(if fx_e < fx_r { &x_e } else { &x_r })?;
            return Ok(NelderMeadMessage::Continue);
        } else if fx_r < fx_second_worst {
            self.replace_worst(&x_r)?;
            return Ok(NelderMeadMessage::Continue);
        } else if fx_r < fx_worst {
            let x_c = self.contract_outside(&x_r);
            let fx_c = self.function.evaluate(&x_c)?;
            if fx_c < fx_r {
                self.replace_worst(&x_c)?;
                return Ok(NelderMeadMessage::Continue);
            }
        } else {
            let x_c = self.contract_inside();
            let fx_c = self.function.evaluate(&x_c)?;
            if fx_c < fx_worst {
                self.replace_worst(&x_c)?;
                return Ok(NelderMeadMessage::Continue);
            }
        }
        self.shrink();
        Ok(NelderMeadMessage::Continue)
    }

    fn terminate(&self) -> bool {
        ComplexField::sqrt(
            self.simplex_fx
                .iter()
                .map(|&fx| ComplexField::powi(fx - self.centroid_fx, 2))
                .sum::<F>()
                / F::from(self.simplex_fx.len()).unwrap(),
        ) <= self.options.min_simplex_standard_deviation
    }

    fn minimize<Func: Fn(NelderMeadMessage)>(&mut self, callback: Func) -> Result<(), E> {
        self.evaluate_simplex()?;
        for i in 0..self.options.max_iters {
            self.order_simplex();
            self.x_best = self.simplex_x[0].clone();
            self.fx_best = self.simplex_fx[0];
            self.calculate_centroid()?;
            if self.terminate() {
                dbg!("Converged!");
                dbg!(i);
                return Ok(());
            }
            let m = self.step()?;
            callback(m);
        }
        Ok(())
    }

    fn best(&self) -> (Vec<F>, F) {
        (self.x_best.clone(), self.fx_best)
    }
}
