use nalgebra::ComplexField;
use typed_builder::TypedBuilder;

use crate::core::{Field, Function, Minimizer};

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

impl<F: Field> NelderMeadOptions<F> {
    /// A set of adaptive hyperparameters according to Gao and Han[^1]. This will produce a
    /// [`NelderMeadOptionsBuilder`] with most parameters set to their adaptive versions, leaving
    /// the [`simplex_size`](`NelderMeadOptions::simplex_size`),
    /// [`min_simplex_standard_deviation`](`NelderMeadOptions::min_simplex_standard_deviation`),
    /// and [`max_iters`](`NelderMeadOptions::max_iters`) fields free for the user to set or leave
    /// as defaults. This method, dubbed ANMS for Adaptive Nelder-Mead Simplex, is identical to the
    /// Standard Nelder-Mead Simplex (SNMS) when the input dimension is equal to 2. The authors of
    /// the paper show that this method can significantly reduce the number of function evaluations
    /// for dimensions greater than 10, although it has mixed results for the Moré-Garbow-Hilstrom
    /// test functions with dimensions between 2 and 6 and can sometimes lead to significantly more
    /// function evaluations. For dimensions greater than 6 for the subset of those test functions
    /// which support higher dimensions, it generally required less evaluations (with several
    /// exceptions, see **Table 4** in the paper for more details).
    ///
    /// [^1]: Gao, F., Han, L. Implementing the Nelder-Mead simplex algorithm with adaptive parameters. *Comput Optim Appl* **51**, 259–277 (2012). <https://doi.org/10.1007/s10589-010-9329-3>
    #[allow(clippy::type_complexity)]
    pub fn adaptive(
        dimension: usize,
    ) -> NelderMeadOptionsBuilder<F, ((), (F,), (F,), (F,), (F,), (F,), (), ())> {
        Self::builder()
            .reflection_coeff(F::convert(1.0))
            .expansion_coeff(F::convert(1.0 + 2.0 / (dimension as f64)))
            .outside_contraction_coeff(F::convert(0.75 - 1.0 / (2.0 * (dimension as f64))))
            .inside_contraction_coeff(F::convert(0.75 - 1.0 / (2.0 * (dimension as f64))))
            .shrink_coeff(F::convert(1.0 - 1.0 / (dimension as f64)))
    }
}

/// The Nelder-Mead method
///
/// The Nelder-Mead method uses a simplex of $`n+1`$ points where $`n`$ is the dimension of the
/// input vector. The algorithm is as follows:
///
/// 0. Pick a starting simplex. The current implementation just takes one simplex point to be the
///    starting point and the others to be steps of [`NelderMeadOptions::simplex_size`] in each
///    coordinate direction.
/// 1. Compute $`f(\vec{x}_i)`$ for each point in the simplex.
/// 2. Calculate the centroid of all but the worst point $`\vec{x}^\dagger`$ in the simplex,
///    $`\vec{x}_o`$.
/// 3. Check for convergence: Terminate if $`\sum_{i=1}^{n+1} (f(\vec{x}_i) - f(\vec{x}_o))^2
///    / (n + 1) < \varepsilon`$ where $`\varepsilon`$ can be set through
///    [`NelderMeadOptions::min_simplex_standard_deviation`].
/// 4. **Reflection**: Compute $`\vec{x}_r = \vec{x}_o + \alpha (\vec{x}_o - \vec{x}^\dagger)`$.
///    If $`f(\vec{x}_r)`$ is better than the second worst point $`\vec{x}^\ddagger`$ and not
///    better than the best point $`\vec{x}^*`$, then replace $`\vec{x}^\dagger`$ with
///    $`\vec{x}_r`$ and go to **Step 1**. Else, go to **Step 6**.
/// 5. **Expansion**: If $`\vec{x}_r`$ is the best point in the simplex, compute the $`\vec{x}_e =
///    \vec{x}_o + \gamma (\vec{x}_r - \vec{x}_o)`$, replace $`\vec{x}^\dagger`$ with whichever is
///    better, $`\vec{x}_r`$ or $`\vec{x}_e`$, and go to **Step 1**.
/// 6. **Contraction**: Here, $`\vec{x}_r`$ is either the worst or second worst point. If it's
///     second-worst, go to **Step 7**. If it's the worst, go to **Step 8**.
/// 7. Compute the "outside" contracted point $`\vec{x}_c + \rho_o (\vec{x}_r - \vec{x}_o)`$.
///    If $`f(\vec{x}_c) < f(\vec{x}_r)`$ (if the contraction improved the point),
///    replace $`\vec{x}^\dagger`$ with $`\vec{x}_c`$ and go to **Step 1**. Else, go to **Step 9**.
/// 8. Compute the "inside" contracted point $`\vec{x}_c - \rho_i (\vec{x}_r - \vec{x}_o)`$ (note
///    the minus sign). If $`f(\vec{x}_c) < f(\vec{x}^\dagger)`$ (if the contraction improved the
///    worst point), replace $`\vec{x}^\dagger`$ with $`\vec{x}_c`$ and go to **Step 1**. Else,
///    go to **Step 9**.
/// 9. **Shrink**: Replace all the points except the best, $`\vec{x}^*`$, with $`\vec{x}_i =
///    \vec{x}^* + \sigma (\vec{x}_i - \vec{x}^*)`$ and go to **Step 1**.
///
/// See [`NelderMeadOptions`] to set the values of $`\alpha`$, $`\gamma`$, $`\rho_i`$, $`\rho_o`$,
/// and $`\sigma`$.
///
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
    n_simplex: usize,
}
impl<F, E> NelderMead<F, E>
where
    F: Field,
{
    /// Create a new Nelder-Mead optimizer from a struct which implements [`Function`], an initial
    /// starting point `x0`, and some options.
    pub fn new<Func: Function<F, E> + 'static>(
        function: Func,
        x0: &[F],
        options: Option<NelderMeadOptions<F>>,
    ) -> Self {
        let n_simplex = x0.len() + 1;
        let options = options.unwrap_or_else(|| NelderMeadOptions::builder().build());
        let simplex_size = options.simplex_size;
        Self {
            function: Box::new(function),
            options,
            simplex_x: Self::construct_simplex(x0, n_simplex, simplex_size),
            simplex_fx: vec![F::nan(); n_simplex],
            centroid_x: vec![F::nan(); x0.len()],
            centroid_fx: F::nan(),
            x_best: vec![F::nan(); x0.len()],
            fx_best: F::nan(),
            n_simplex,
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
        indices.sort_by(|&i, &j| {
            self.simplex_fx[i]
                .partial_cmp(&self.simplex_fx[j])
                .unwrap_or(std::cmp::Ordering::Equal) // this happens for NaNs
        });
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
        let n_points = F::convert_usize(self.simplex_x.len() - 1);
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
            .zip(self.simplex_x[self.n_simplex - 1].iter())
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
    fn contract_inside(&self, x_r: &[F]) -> Vec<F> {
        self.centroid_x
            .iter()
            .zip(x_r)
            .map(|(&c, &x)| c - self.options.inside_contraction_coeff * (x - c))
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
/// A message passed into the [`NelderMead::minimize`] callback.
pub struct NelderMeadMessage<F> {
    /// The current step number.
    pub step: usize,
    /// The current position of the simplex centroid.
    pub x_c: Vec<F>,
    /// The current value of the function evaluated at the simplex centroid.
    pub fx_c: F,
    /// The current best position of the minimizer.
    pub x: Vec<F>,
    /// The current best value of the minimizer function.
    pub fx: F,
}

impl<F, E> Minimizer<F, NelderMeadMessage<F>, E> for NelderMead<F, E>
where
    F: Field,
{
    fn step(&mut self, i: usize) -> Result<NelderMeadMessage<F>, E> {
        let fx_best = self.simplex_fx[0];
        let fx_second_worst = self.simplex_fx[self.n_simplex - 2];
        let fx_worst = self.simplex_fx[self.n_simplex - 1];
        let x_r = self.reflect();
        let fx_r = self.function.evaluate(&x_r)?;
        if fx_r < fx_best {
            let x_e = self.expand(&x_r);
            let fx_e = self.function.evaluate(&x_e)?;
            self.replace_worst(if fx_e < fx_r { &x_e } else { &x_r })?;
            return Ok(NelderMeadMessage {
                step: i,
                x_c: self.centroid_x.clone(),
                fx_c: self.centroid_fx,
                x: x_r.clone(),
                fx: fx_r,
            });
        } else if fx_r < fx_second_worst {
            self.replace_worst(&x_r)?;
            return Ok(NelderMeadMessage {
                step: i,
                x_c: self.centroid_x.clone(),
                fx_c: self.centroid_fx,
                x: self.simplex_x[0].clone(),
                fx: fx_best,
            });
        } else if fx_r < fx_worst {
            let x_c = self.contract_outside(&x_r);
            let fx_c = self.function.evaluate(&x_c)?;
            if fx_c < fx_r {
                self.replace_worst(&x_c)?;
                return Ok(NelderMeadMessage {
                    step: i,
                    x_c: self.centroid_x.clone(),
                    fx_c: self.centroid_fx,
                    x: self.simplex_x[0].clone(),
                    fx: fx_best,
                });
            }
        } else {
            let x_c = self.contract_inside(&x_r);
            let fx_c = self.function.evaluate(&x_c)?;
            if fx_c < fx_worst {
                self.replace_worst(&x_c)?;
                return Ok(NelderMeadMessage {
                    step: i,
                    x_c: self.centroid_x.clone(),
                    fx_c: self.centroid_fx,
                    x: self.simplex_x[0].clone(),
                    fx: fx_best,
                });
            }
        }
        self.shrink();
        Ok(NelderMeadMessage {
            step: i,
            x_c: self.centroid_x.clone(),
            fx_c: self.centroid_fx,
            x: self.simplex_x[0].clone(),
            fx: fx_best,
        })
    }

    fn terminate(&self) -> bool {
        ComplexField::sqrt(
            self.simplex_fx
                .iter()
                .map(|&fx| ComplexField::powi(fx - self.centroid_fx, 2))
                .sum::<F>()
                / F::convert_usize(self.simplex_fx.len()),
        ) <= self.options.min_simplex_standard_deviation
    }

    fn minimize<Func: Fn(&NelderMeadMessage<F>)>(
        &mut self,
        callback: Func,
    ) -> Result<NelderMeadMessage<F>, E> {
        self.evaluate_simplex()?;
        self.order_simplex();
        let mut m = NelderMeadMessage {
            step: 0,
            x_c: self.centroid_x.clone(),
            fx_c: self.centroid_fx,
            x: self.x_best.clone(),
            fx: self.fx_best,
        };
        for i in 0..self.options.max_iters {
            self.order_simplex();
            self.x_best = self.simplex_x[0].clone();
            self.fx_best = self.simplex_fx[0];
            self.calculate_centroid()?;
            if self.terminate() {
                return Ok(m);
            }
            m = self.step(i)?;
            callback(&m);
        }
        Ok(m)
    }

    fn best(&self) -> (Vec<F>, F) {
        (self.x_best.clone(), self.fx_best)
    }
}
