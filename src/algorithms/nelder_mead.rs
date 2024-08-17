use nalgebra::DVector;
use typed_builder::TypedBuilder;

use crate::core::{convert, Field, Function, Minimizer};

/// Used to set options for the [`NelderMead`] optimizer.
///
/// See also: [`NelderMeadOptions::builder()`]
#[derive(TypedBuilder)]
pub struct NelderMeadOptions<F>
where
    F: Field,
{
    // TODO: validate coeffs, alpha > 0, gamma > 1, 0 < rho <= 0.5, sigma (0 < sigma < 1?)
    /// The step size from the starting point to each other point in the simplex (default = 1.0)
    #[builder(default = F::one())]
    pub simplex_size: F,
    /// The coefficient $`\alpha > 0`$ to use in the reflection step (default = 1.0)
    #[builder(default = F::one())]
    pub reflection_coeff: F,
    /// The coefficient $`\gamma > 1`$ to use in the expansion step (default = 2.0)
    #[builder(default = convert!(2.0, F))]
    pub expansion_coeff: F,
    /// The coefficient $`0 < \rho_o \leq 0.5`$ to use in the contraction step
    /// in the case where the reflected point is better than the worst point in the simplex (default = 0.5)
    #[builder(default = convert!(0.5, F))]
    pub outside_contraction_coeff: F,
    /// The coefficient $`0 < \rho_i \leq 0.5`$ to use in the contraction step
    /// in the case where the reflected point is worse than the worst point in the simplex (default = 0.5)
    #[builder(default = convert!(0.5, F))]
    pub inside_contraction_coeff: F,
    /// The coefficient $`\sigma > 0`$ to use in the shrink step (default = 0.5)
    #[builder(default = convert!(0.5, F))]
    pub shrink_coeff: F,
    /// If the standard deviation of the function at all points in the
    /// simplex falls below this value, the algorithm will terminate/converge (default=1e-8)
    #[builder(default = convert!(1e-8, F))]
    pub min_simplex_standard_deviation: F,
}

impl<F> NelderMeadOptions<F>
where
    F: Field,
{
    /// A set of adaptive hyperparameters according to Gao and Han[^1]. This will produce a
    /// [`NelderMeadOptionsBuilder`] with most parameters set to their adaptive versions, leaving
    /// the [`simplex_size`](`NelderMeadOptions::simplex_size`) and
    /// [`min_simplex_standard_deviation`](`NelderMeadOptions::min_simplex_standard_deviation`)
    /// fields free for the user to set or leave
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
    ) -> NelderMeadOptionsBuilder<F, ((), (F,), (F,), (F,), (F,), (F,), ())> {
        Self::builder()
            .reflection_coeff(F::one())
            .expansion_coeff(convert!(1.0 + 2.0 / (dimension as f32), F))
            .outside_contraction_coeff(convert!(0.75 - 1.0 / (2.0 * (dimension as f32)), F))
            .inside_contraction_coeff(convert!(0.75 - 1.0 / (2.0 * (dimension as f32)), F))
            .shrink_coeff(convert!(1.0 - 1.0 / (dimension as f32), F))
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
pub struct NelderMead<F, A, E>
where
    F: Field,
{
    function: Box<dyn Function<F, A, E>>,
    options: NelderMeadOptions<F>,
    simplex_x: Vec<DVector<F>>,
    simplex_fx: Vec<F>,
    centroid_x: DVector<F>,
    centroid_fx: F,
    x_best: DVector<F>,
    fx_best: F,
    sstd: F,
    n_simplex: usize,
    current_step: usize,
}
impl<F, A, E> NelderMead<F, A, E>
where
    F: Field + 'static,
{
    /// Create a new Nelder-Mead optimizer from a struct which implements [`Function`], an initial
    /// starting point `x0`, and some options.
    pub fn new<Func: Function<F, A, E> + 'static>(
        function: Func,
        x0: &[F],
        options: Option<NelderMeadOptions<F>>,
    ) -> Self {
        let x0 = DVector::from_row_slice(x0);
        let n_simplex = x0.len() + 1;
        let options = options.unwrap_or_else(|| NelderMeadOptions::builder().build());
        let simplex_size = options.simplex_size;
        Self {
            function: Box::new(function),
            options,
            simplex_x: Self::construct_simplex(&x0, n_simplex, simplex_size),
            simplex_fx: vec![F::nan(); n_simplex],
            centroid_x: DVector::from_element(x0.len(), F::nan()),
            centroid_fx: F::nan(),
            x_best: DVector::from_element(x0.len(), F::nan()),
            fx_best: F::infinity(),
            sstd: F::nan(),
            n_simplex,
            current_step: 0,
        }
    }
    fn construct_simplex(x0: &DVector<F>, n_simplex: usize, simplex_size: F) -> Vec<DVector<F>> {
        (0..n_simplex)
            .map(|i| {
                if i == 0 {
                    x0.clone_owned()
                } else {
                    let mut xi = x0.clone_owned();
                    xi[i - 1] += simplex_size;
                    xi
                }
            })
            .collect()
    }
    fn evaluate_simplex(&mut self, args: Option<&A>) -> Result<(), E> {
        self.simplex_fx = self
            .simplex_x
            .iter()
            .map(|x| self.function.evaluate(x, args))
            .collect::<Result<Vec<F>, E>>()?;
        Ok(())
    }
    fn order_simplex(&mut self) {
        let mut indices: Vec<usize> = (0..self.simplex_fx.len()).collect();
        indices.sort_by(|&i, &j| {
            self.simplex_fx[i]
                .partial_cmp(&self.simplex_fx[j])
                .unwrap_or(std::cmp::Ordering::Equal) // this happens for NaNs
        });
        let (sorted_simplex_x, sorted_simplex_fx): (Vec<DVector<F>>, Vec<F>) = indices
            .iter()
            .map(|&i| (self.simplex_x[i].clone_owned(), self.simplex_fx[i]))
            .unzip();
        self.simplex_x = sorted_simplex_x;
        self.simplex_fx = sorted_simplex_fx;
    }
    fn calculate_centroid(&mut self, args: Option<&A>) -> Result<(), E> {
        assert!(!self.simplex_x.is_empty(), "Simplex is empty!");
        let dim = self.simplex_x.len();
        self.centroid_x =
            self.simplex_x.iter().take(dim - 1).sum::<DVector<F>>() / convert!(dim as f32 - 1.0, F);
        self.centroid_fx = self.function.evaluate(&self.centroid_x, args)?;
        Ok(())
    }
    fn reflect(&self) -> DVector<F> {
        &self.centroid_x
            + (&self.centroid_x - &self.simplex_x[self.n_simplex - 1])
                * self.options.expansion_coeff
    }
    fn expand(&self, x_r: &DVector<F>) -> DVector<F> {
        &self.centroid_x + (x_r - &self.centroid_x) * self.options.expansion_coeff
    }
    fn contract_outside(&self, x_r: &DVector<F>) -> DVector<F> {
        &self.centroid_x + (x_r - &self.centroid_x) * self.options.outside_contraction_coeff
    }
    fn contract_inside(&self, x_r: &DVector<F>) -> DVector<F> {
        &self.centroid_x - (x_r - &self.centroid_x) * self.options.inside_contraction_coeff
    }
    fn shrink(&mut self) {
        let simplex_x_best = self.simplex_x[0].clone_owned();
        self.simplex_x.iter_mut().skip(1).for_each(|xi| {
            *xi = DVector::from_iterator(
                xi.len(),
                xi.iter()
                    .zip(&simplex_x_best)
                    .map(|(&x, &x_best)| x_best + self.options.shrink_coeff * (x - x_best)),
            );
        })
    }
    fn replace_worst(&mut self, x: &DVector<F>, args: Option<&A>) -> Result<(), E> {
        let i_last = self.simplex_x.len() - 1;
        self.simplex_x[i_last] = x.clone_owned();
        self.simplex_fx[i_last] = self.function.evaluate(x, args)?;
        Ok(())
    }
    fn calculate_standard_deviation(&mut self) {
        self.sstd = F::sqrt(
            self.simplex_fx
                .iter()
                .map(|&fx| F::powi(fx - self.centroid_fx, 2))
                .sum::<F>()
                / convert!(self.simplex_fx.len() as f32, F),
        );
    }
}

impl<F, A, E> Minimizer<F, A, E> for NelderMead<F, A, E>
where
    F: Field + 'static,
{
    fn step(&mut self, args: Option<&A>) -> Result<(), E> {
        self.current_step += 1;
        self.calculate_standard_deviation();
        let fx_best = self.simplex_fx[0];
        let fx_second_worst = self.simplex_fx[self.n_simplex - 2];
        let fx_worst = self.simplex_fx[self.n_simplex - 1];
        let x_r = self.reflect();
        let fx_r = self.function.evaluate(&x_r, args)?;
        if fx_r < fx_best {
            let x_e = self.expand(&x_r);
            let fx_e = self.function.evaluate(&x_e, args)?;
            self.replace_worst(if fx_e < fx_r { &x_e } else { &x_r }, args)?;
        } else if fx_r < fx_second_worst {
            self.replace_worst(&x_r, args)?;
        } else if fx_r < fx_worst {
            let x_c = self.contract_outside(&x_r);
            let fx_c = self.function.evaluate(&x_c, args)?;
            if fx_c < fx_r {
                self.replace_worst(&x_c, args)?;
            } else {
                self.shrink();
            }
        } else {
            let x_c = self.contract_inside(&x_r);
            let fx_c = self.function.evaluate(&x_c, args)?;
            if fx_c < fx_worst {
                self.replace_worst(&x_c, args)?;
            } else {
                self.shrink();
            }
        }
        self.order_simplex();
        self.calculate_centroid(args)?;
        Ok(())
    }

    fn check_for_termination(&self) -> bool {
        self.sstd <= self.options.min_simplex_standard_deviation
    }

    fn best(&self) -> (&DVector<F>, &F) {
        (&self.x_best, &self.fx_best)
    }

    fn update_best(&mut self) {
        self.x_best = self.simplex_x[0].clone();
        self.fx_best = self.simplex_fx[0];
    }

    fn initialize(&mut self, args: Option<&A>) -> Result<(), E> {
        self.evaluate_simplex(args)?;
        self.order_simplex();
        self.calculate_centroid(args)?;
        Ok(())
    }
}
