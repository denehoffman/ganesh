use crate::{
    algorithms::gradient_free::GradientFreeStatus,
    core::{Bounds, Callbacks, MinimizationSummary, Point},
    traits::{Algorithm, Boundable, Bounded, CostFunction, Terminator},
    DMatrix, DVector, Float,
};
use std::{fmt::Debug, ops::ControlFlow};

/// Gives a method for constructing a simplex.
#[derive(Debug, Clone)]
pub enum SimplexConstructionMethod {
    /// Creates a simplex by starting at the given `x0` and stepping a distance of `+simplex_size`
    /// in every orthogonal direction.
    Orthogonal {
        /// The distance from the starting point to each of the other points in the simplex.
        simplex_size: Float,
    },
    /// Creates a custom simplex from a list of points.
    Custom {
        /// The points to use in the simplex (ignores any given starting point).
        simplex: Vec<Vec<Float>>,
    },
}
impl Default for SimplexConstructionMethod {
    fn default() -> Self {
        Self::Orthogonal { simplex_size: 1.0 }
    }
}

impl SimplexConstructionMethod {
    fn generate<U, E>(
        &self,
        func: &dyn CostFunction<U, E, Input = DVector<Float>>,
        x0: &[Float],
        bounds: Option<&Bounds>,
        args: &U,
    ) -> Result<Simplex, E> {
        match self {
            Self::Orthogonal { simplex_size } => {
                let mut points = Vec::default();
                let mut point_0 =
                    Point::from(DVector::from_column_slice(x0).unconstrain_from(bounds));
                point_0.evaluate_bounded(func, bounds, args)?;
                points.push(point_0.clone());
                let dim = point_0.x.len();
                assert!(
                    dim >= 2,
                    "Nelder-Mead is only a suitable method for problems of dimension >= 2"
                );
                for i in 0..dim {
                    let mut point_i = point_0.clone();
                    point_i.x[i] += *simplex_size;
                    point_i.fx = Float::NAN;
                    point_i.evaluate_bounded(func, bounds, args)?;
                    points.push(point_i);
                }
                Ok(Simplex::new(&points))
            }
            Self::Custom { simplex } => {
                assert!(!simplex.is_empty());
                assert!(simplex.len() == simplex[0].len() + 1);
                assert!(simplex.len() > 2);
                Ok(Simplex::new(
                    &simplex
                        .iter()
                        .map(|x| {
                            let mut point_i =
                                Point::from(DVector::from_column_slice(x).unconstrain_from(bounds));
                            point_i.evaluate_bounded(func, bounds, args)?;
                            Ok(point_i)
                        })
                        .collect::<Result<Vec<Point<DVector<Float>>>, E>>()?,
                ))
            }
        }
    }
}

/// A [`Simplex`] represents a list of [`Point`]s. This particular implementation is intended to be
/// sorted.
#[derive(Default, Clone)]
pub struct Simplex {
    points: Vec<Point<DVector<Float>>>,
    dimension: usize,
    sorted: bool,
    total_centroid: DVector<Float>,
    volume: Float,
    initial_best: Point<DVector<Float>>,
    initial_worst: Point<DVector<Float>>,
    initial_volume: Float,
}
impl Debug for Simplex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.points)
    }
}
impl Simplex {
    fn new(points: &[Point<DVector<Float>>]) -> Self {
        let mut sorted_points = points.to_vec();
        sorted_points.sort_by(|a, b| a.total_cmp(b));
        let initial_best = sorted_points[0].clone();
        let initial_worst = sorted_points[sorted_points.len() - 1].clone();
        let n_params = points.len() - 1;
        let diffs: Vec<DVector<Float>> = sorted_points
            .iter()
            .skip(1)
            .map(|p| &p.x - &initial_best.x)
            .collect();
        let gram_mat = DMatrix::from_fn(n_params, n_params, |i, j| diffs[i].dot(&diffs[j]));
        // NOTE: volume calculation is off by a constant 1/n! which divides out on both sides
        // whenever we use this!
        let volume = Float::sqrt(gram_mat.determinant());
        let total_centroid =
            sorted_points.iter().map(|p| &p.x).sum::<DVector<Float>>() / points.len() as Float;
        Self {
            points: sorted_points,
            dimension: points.len(),
            sorted: false,
            total_centroid,
            volume,
            initial_best,
            initial_worst,
            initial_volume: volume,
        }
    }
    fn corrected_centroid(&self) -> DVector<Float> {
        let n = self.points.len();
        let total = &self.total_centroid * (n as Float);
        let sum = total - &self.points[n - 1].x;
        sum / ((n - 1) as Float)
    }
    fn best_position(&self, bounds: Option<&Bounds>) -> (DVector<Float>, Float) {
        let (y, fx) = self.best().clone().destructure();
        (y.constrain_to(bounds), fx)
    }
    fn best(&self) -> &Point<DVector<Float>> {
        &self.points[0]
    }
    fn worst(&self) -> &Point<DVector<Float>> {
        &self.points[self.points.len() - 1]
    }
    fn second_worst(&self) -> &Point<DVector<Float>> {
        &self.points[self.points.len() - 2]
    }
    fn insert_and_sort(&mut self, index: usize, element: Point<DVector<Float>>) {
        let removed = self.points.remove(self.points.len() - 1);
        let n = self.points.len() as Float + 1.0;
        self.total_centroid += (&element.x - &removed.x) / n;

        self.points.insert(index, element);
        self.sorted = false;
        self.sort();
    }
    fn insert_sorted(&mut self, index: usize, element: Point<DVector<Float>>) {
        let removed = self.points.remove(self.points.len() - 1);
        self.points.insert(index, element);
        self.sorted = true;

        let n = self.points.len() as Float;
        self.total_centroid += (&self.points[index].x - &removed.x) / n;
    }

    fn sort(&mut self) {
        if !self.sorted {
            self.sorted = true;
            self.points.sort_by(|a, b| a.total_cmp(b));
        }
    }
    fn compute_total_centroid(&mut self) {
        let n = self.points.len() as Float;
        self.total_centroid = self.points.iter().map(|p| &p.x).sum::<DVector<Float>>() / n;
    }
    fn scale_volume(&mut self, factor: Float) {
        self.volume *= factor;
    }
}

/// Selects the expansion method used in the Nelder-Mead algorithm. See Lagarias et al.[^1] for more details.
///
/// [^1]: [J. C. Lagarias, J. A. Reeds, M. H. Wright, and P. E. Wright, ‘Convergence Properties of the Nelder--Mead Simplex Method in Low Dimensions’, SIAM Journal on Optimization, vol. 9, no. 1, pp. 112–147, 1998.](https://doi.org/10.1137/S1052623496303470)
#[derive(Default, Debug, Clone)]
pub enum SimplexExpansionMethod {
    /// Greedy minimization will calculate both a reflected an expanded point in an expansion step
    /// but will return the one that gives the best minimum.
    #[default]
    GreedyMinimization,
    /// Greedy expansion will calculate both a reflected and expanded point in an expansion step
    /// but will return the expanded point always, even if the reflected point is a better minimum.
    GreedyExpansion,
}

/// Various termination methods based on the evaluation of the function at each point in the
/// simplex. See Singer et al.[^1] for more details.
///
/// [^1]: [S. Singer and S. Singer, ‘Efficient Implementation of the Nelder–Mead Search Algorithm’, Applied Numerical Analysis & Computational Mathematics, vol. 1, no. 2, pp. 524–534, 2004.](https://doi.org/10.1002/anac.200410015)
#[derive(Debug, Clone, Default)]
pub enum NelderMeadFTerminator {
    /// For the worst point $`x_h`$ and best point $`x_l`$, converge if the following is true:
    /// ```math
    /// 2 \frac{f(x_h) - f(x_l)}{|f(x_h)| + |f(x_l)|} <= \varepsilon
    /// ```
    Amoeba,
    /// For the worst point $`x_h`$ and best point $`x_l`$, converge if the following is true:
    /// ```math
    /// f(x_h) - f(x_l) <= \varepsilon
    /// ```
    Absolute,
    /// Converge if the standard deviation of the function evaluations of all points in the simplex
    /// is $`\sigma <= \varepsilon`$.
    #[default]
    StdDev,
}
impl<P, U, E> Terminator<NelderMead, P, GradientFreeStatus, U, E> for NelderMeadFTerminator
where
    P: CostFunction<U, E, Input = DVector<Float>>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut NelderMead,
        _problem: &P,
        status: &mut GradientFreeStatus,
        _args: &U,
    ) -> ControlFlow<()> {
        let simplex = &algorithm.config.simplex;
        match self {
            Self::Amoeba => {
                let fh = simplex.worst().fx_checked();
                let fl = simplex.best().fx_checked();
                if 2.0 * (fh - fl) / (Float::abs(fh) + Float::abs(fl)) <= algorithm.config.eps_f_rel
                {
                    status.set_converged();
                    status.with_message("term_f = AMOEBA");
                    return ControlFlow::Break(());
                }
            }
            Self::Absolute => {
                let fh = simplex.worst().fx_checked();
                let fl = simplex.best().fx_checked();
                if fh - fl <= algorithm.config.eps_f_abs {
                    status.set_converged();
                    status.with_message("term_f = ABSOLUTE");
                    return ControlFlow::Break(());
                }
            }
            Self::StdDev => {
                let dim = simplex.dimension as Float;
                let mean = simplex.points.iter().map(|point| point.fx).sum::<Float>() / dim;
                let std_dev = Float::sqrt(
                    simplex
                        .points
                        .iter()
                        .map(|point| Float::powi(point.fx - mean, 2))
                        .sum::<Float>()
                        / dim,
                );
                if std_dev <= algorithm.config.eps_f_abs {
                    status.set_converged();
                    status.with_message("term_f = STDDEV");
                    return ControlFlow::Break(());
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// Various termination methods based on the the position of points in the simplex.
/// See Singer et al.[^1] for more details.
///
/// [^1]: [S. Singer and S. Singer, ‘Efficient Implementation of the Nelder–Mead Search Algorithm’, Applied Numerical Analysis & Computational Mathematics, vol. 1, no. 2, pp. 524–534, 2004.](https://doi.org/10.1002/anac.200410015)
#[derive(Debug, Clone, Default)]
pub enum NelderMeadXTerminator {
    /// For the best point in the simplex $`x_l`$, converge if the following condition is met:
    /// ```math
    /// \max_{j\neq l} ||x_j - x_l||_{\inf} \leq \varepsilon
    /// ```
    Diameter,
    /// For the best point in the simplex $`x_l`$, converge if the following condition is met:
    /// ```math
    /// \frac{\max_{j\neq l} ||x_j - x_l||_1}{\max\left\{1, ||x_l||_1\right\}} \leq \varepsilon
    /// ```
    Higham,
    /// For the worst point $`x_h`$ and best point $`x_l`$, as well as the original values of those
    /// points at the beginning of the algorithm, denoted $`x_h^{(0)}`$ and $`x_l^{(0)}`$
    /// respectively, converge if the following condition is met:
    /// ```math
    /// ||x_h - x_l||_2 \leq \varepsilon ||x_h^{(0)} - x_l^{(0)}||_2
    /// ```
    Rowan,
    /// Given the volume of the simplex
    /// ```math
    /// V(S) \equiv \frac{1}{n!}\sqrt{\Gamma(x_1-x_0,...,x_n-x_0)}
    /// ```
    /// where $`\Gamma`$ is the Gram determinant, compute the linearized volume $`LV(S)\equiv
    /// \sqrt[n]{V(S)}`$ and converge if the following condition is met:
    /// ```math
    /// LV(S) \leq \varepsilon LV(S^{(0)})
    /// ```
    /// where $`S`$ is the current simplex and $`S^{(0)}`$ is the simplex at the beginning of the
    /// algorithm. Note that $`V(S)`$ is only calculated once in practice and updated according to
    /// each step type by a single multiplication, so this method is very efficient.
    #[default]
    Singer,
}

impl<P, U, E> Terminator<NelderMead, P, GradientFreeStatus, U, E> for NelderMeadXTerminator
where
    P: CostFunction<U, E, Input = DVector<Float>>,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut NelderMead,
        _problem: &P,
        status: &mut GradientFreeStatus,
        _args: &U,
    ) -> ControlFlow<()> {
        let simplex = &algorithm.config.simplex;
        match self {
            Self::Diameter => {
                let l = simplex.worst();
                let max_inf_norm = simplex
                    .points
                    .iter()
                    .rev()
                    .skip(1) // skip l itself
                    .map(|point| {
                        let diff = &point.x - &l.x;
                        let mut inf_norm = 0.0;
                        for i in 0..diff.len() {
                            if inf_norm < Float::abs(diff[i]) {
                                inf_norm = Float::abs(diff[i])
                            }
                        }
                        inf_norm
                    })
                    .max_by(|&a, &b| a.total_cmp(&b))
                    .unwrap_or(0.0);
                if max_inf_norm <= algorithm.config.eps_x_abs {
                    status.set_converged();
                    status.with_message("term_x = DIAMETER");
                    return ControlFlow::Break(());
                }
            }
            Self::Higham => {
                let l = simplex.worst();
                let l1_norm_l = l.x.lp_norm(1);
                let denom = Float::max(l1_norm_l, 1.0);
                let numer = simplex
                    .points
                    .iter()
                    .rev()
                    .skip(1)
                    .map(|point| {
                        let diff = &point.x - &l.x;
                        diff.lp_norm(1)
                    })
                    .max_by(|&a, &b| a.total_cmp(&b))
                    .unwrap_or(0.0);
                if numer / denom <= algorithm.config.eps_x_rel {
                    status.set_converged();
                    status.with_message("term_x = HIGHAM");
                    return ControlFlow::Break(());
                }
            }
            Self::Rowan => {
                let init_diff = (&simplex.initial_worst.x - &simplex.initial_best.x).lp_norm(2);
                let current_diff = (&simplex.worst().x - &simplex.best().x).lp_norm(2);
                if current_diff <= algorithm.config.eps_x_rel * init_diff {
                    status.set_converged();
                    status.with_message("term_x = ROWAN");
                    return ControlFlow::Break(());
                }
            }
            Self::Singer => {
                let dim = simplex.dimension as Float;
                let lv_init = Float::powf(simplex.initial_volume, 1.0 / dim);
                let lv_current = Float::powf(simplex.volume, 1.0 / dim);
                if lv_current <= algorithm.config.eps_x_rel * lv_init {
                    status.set_converged();
                    status.with_message("term_x = SINGER");
                    return ControlFlow::Break(());
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// The internal configuration struct for the [`NelderMead`] algorithm.
#[derive(Clone)]
pub struct NelderMeadConfig {
    x0: DVector<Float>,
    bounds: Option<Bounds>,
    alpha: Float,
    beta: Float,
    gamma: Float,
    delta: Float,
    simplex: Simplex,
    construction_method: SimplexConstructionMethod,
    expansion_method: SimplexExpansionMethod,
    eps_x_rel: Float,
    eps_x_abs: Float,
    eps_f_rel: Float,
    eps_f_abs: Float,
}
impl NelderMeadConfig {
    /// Set the starting position of the algorithm.
    pub fn with_x0<I: IntoIterator<Item = Float>>(mut self, x0: I) -> Self {
        let x0 = x0.into_iter().collect::<Vec<Float>>();
        self.x0 = DVector::from_column_slice(&x0);
        self
    }
    /// Set the relative x-convergence tolerance (default = `MACH_EPS^(1/4)`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\epsilon <= 0`$.
    pub fn with_eps_x_rel(mut self, value: Float) -> Self {
        assert!(value > 0.0);
        self.eps_x_rel = value;
        self
    }
    /// Set the absolute x-convergence tolerance (default = `MACH_EPS^(1/4)`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\epsilon <= 0`$.
    pub fn with_eps_x_abs(mut self, value: Float) -> Self {
        assert!(value > 0.0);
        self.eps_x_abs = value;
        self
    }
    /// Set the relative f-convergence tolerance (default = `MACH_EPS^(1/4)`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\epsilon <= 0`$.
    pub fn with_eps_f_rel(mut self, value: Float) -> Self {
        assert!(value > 0.0);
        self.eps_f_rel = value;
        self
    }
    /// Set the absolute f-convergence tolerance (default = `MACH_EPS^(1/4)`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\epsilon <= 0`$.
    pub fn with_eps_f_abs(mut self, value: Float) -> Self {
        assert!(value > 0.0);
        self.eps_f_abs = value;
        self
    }
    /// Set the reflection coefficient $`\alpha`$ (default = `1`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\alpha <= 0`$.
    pub fn with_alpha(mut self, value: Float) -> Self {
        assert!(value > 0.0);
        self.alpha = value;
        self
    }
    /// Set the expansion coefficient $`\beta`$ (default = `2`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\beta <= 1`$ or $`\beta <= \alpha`$.
    pub fn with_beta(mut self, value: Float) -> Self {
        assert!(value > 1.0);
        assert!(value > self.alpha);
        self.beta = value;
        self
    }
    /// Set the contraction coefficient $`\gamma`$ (default = `0.5`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\gamma >= 1`$ or $`\gamma <= 0`$.
    pub fn with_gamma(mut self, value: Float) -> Self {
        assert!(value > 0.0);
        assert!(value < 1.0);
        self.gamma = value;
        self
    }
    /// Set the shrink coefficient $`\delta`$ (default = `0.5`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\delta >= 1`$ or $`\delta <= 0`$.
    pub fn with_delta(mut self, value: Float) -> Self {
        assert!(value > 0.0);
        assert!(value < 1.0);
        self.delta = value;
        self
    }
    /// A set of adaptive hyperparameters according to Gao and Han[^1]. This method, dubbed ANMS
    /// for Adaptive Nelder-Mead Simplex, is identical to the Standard Nelder-Mead Simplex (SNMS)
    /// when the input dimension is equal to 2. The authors of the paper show that this method can
    /// significantly reduce the number of function evaluations for dimensions greater than 10,
    /// although it has mixed results for the Moré-Garbow-Hilstrom test functions with dimensions
    /// between 2 and 6 and can sometimes lead to significantly more function evaluations. For
    /// dimensions greater than 6 for the subset of those test functions which support higher
    /// dimensions, it generally required less evaluations (with several exceptions, see
    /// **Table 4** in the paper for more details).
    ///
    /// [^1]: [Gao, F., Han, L. Implementing the Nelder-Mead simplex algorithm with adaptive parameters. *Comput Optim Appl* **51**, 259–277 (2012).](https://doi.org/10.1007/s10589-010-9329-3)
    pub fn with_adaptive(mut self, n: usize) -> Self {
        let n = n as Float;
        self.alpha = 1.0;
        self.beta = 1.0 + (2.0 / n);
        self.gamma = 0.75 - 1.0 / (2.0 * n);
        self.delta = 1.0 - 1.0 / n;
        self
    }
    /// Use the given [`SimplexConstructionMethod`] to compute the starting [`Simplex`].
    pub fn with_construction_method(mut self, method: SimplexConstructionMethod) -> Self {
        self.construction_method = method;
        self
    }
    /// Set the [`SimplexExpansionMethod`].
    pub const fn with_expansion_method(mut self, method: SimplexExpansionMethod) -> Self {
        self.expansion_method = method;
        self
    }
}
impl Bounded for NelderMeadConfig {
    fn get_bounds_mut(&mut self) -> &mut Option<Bounds> {
        &mut self.bounds
    }
}
impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            x0: DVector::zeros(0),
            bounds: None,
            alpha: 1.0,
            beta: 2.0,
            gamma: 0.5,
            delta: 0.5,
            simplex: Simplex::default(),
            construction_method: SimplexConstructionMethod::default(),
            expansion_method: SimplexExpansionMethod::default(),
            eps_x_rel: Float::EPSILON.powf(0.25),
            eps_x_abs: Float::EPSILON.powf(0.25),
            eps_f_rel: Float::EPSILON.powf(0.25),
            eps_f_abs: Float::EPSILON.powf(0.25),
        }
    }
}

/// The Nelder-Mead method
///
/// The Nelder-Mead method uses a simplex of $`n+1`$ points where $`n`$ is the dimension of the
/// input vector. The algorithm is as follows:
///
/// 0. Pick a starting simplex. The default implementation just takes one simplex point to be the
///    starting point and the others to be steps of equal size in each coordinate direction.
/// 1. Compute $`f(\vec{x}_i)`$ for each point in the simplex.
/// 2. Calculate the centroid of all but the worst point $`\vec{x}^\dagger`$ in the simplex,
///    $`\vec{x}_o`$.
/// 3. Check for convergence: (see [`NelderMeadFTerminator`] and [`NelderMeadXTerminator`]).
/// 4. **Reflection**: Compute $`\vec{x}_r = \vec{x}_o + \alpha (\vec{x}_o - \vec{x}^\dagger)`$.
///    If $`f(\vec{x}_r)`$ is better than the second worst point $`\vec{x}^\ddagger`$ and not
///    better than the best point $`\vec{x}^*`$, then replace $`\vec{x}^\dagger`$ with
///    $`\vec{x}_r`$ and go to **Step 1**. Else, go to **Step 6**.
/// 5. **Expansion**: If $`\vec{x}_r`$ is the best point in the simplex, compute the $`\vec{x}_e =
///    \vec{x}_o + \gamma (\vec{x}_r - \vec{x}_o)`$, replace $`\vec{x}^\dagger`$ with whichever is
///    better, $`\vec{x}_r`$ or $`\vec{x}_e`$, if greedy minimization is used, otherwise choose
///    $`\vec{x}_e`$ if greedy expansion is used and go to **Step 1**.
/// 6. **Contraction**: Here, $`\vec{x}_r`$ is either the worst or second worst point. If it's
///    second-worst, go to **Step 7**. If it's the worst, go to **Step 8**.
/// 7. Compute the "outside" contracted point $`\vec{x}_c + \rho_o (\vec{x}_r - \vec{x}_o)`$.
///    If $`f(\vec{x}_c) < f(\vec{x}_r)`$ (if the contraction improved the point),
///    replace $`\vec{x}^\dagger`$ with $`\vec{x}_c`$ and go to **Step 1**. Else, go to **Step 9**.
/// 8. Compute the "inside" contracted point $`\vec{x}_c - \rho_i (\vec{x}_r - \vec{x}_o)`$ (note
///    the minus sign). If $`f(\vec{x}_c) < f(\vec{x}^\dagger)`$ (if the contraction improved the
///    worst point), replace $`\vec{x}^\dagger`$ with $`\vec{x}_c`$ and go to **Step 1**. Else,
///    go to **Step 9**.
/// 9. **Shrink**: Replace all the points except the best, $`\vec{x}^*`$, with $`\vec{x}_i =
///    \vec{x}^* + \sigma (\vec{x}_i - \vec{x}^*)`$ and go to **Step 1**.
#[derive(Clone, Default)]
pub struct NelderMead {
    config: NelderMeadConfig,
}
impl<P, U, E> Algorithm<P, GradientFreeStatus, U, E> for NelderMead
where
    P: CostFunction<U, E, Input = DVector<Float>>,
{
    type Summary = MinimizationSummary;
    type Config = NelderMeadConfig;
    fn initialize(
        &mut self,
        config: Self::Config,
        problem: &mut P,
        status: &mut GradientFreeStatus,
        args: &U,
    ) -> Result<(), E> {
        self.config = config;
        self.config.simplex = self.config.construction_method.generate(
            problem,
            self.config.x0.as_slice(),
            self.config.bounds.as_ref(),
            args,
        )?;
        status.with_position(
            self.config
                .simplex
                .best_position(self.config.bounds.as_ref()),
        );
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &mut P,
        status: &mut GradientFreeStatus,
        args: &U,
    ) -> Result<(), E> {
        let bounds = self.config.bounds.as_ref();
        let h = self.config.simplex.worst();
        let s = self.config.simplex.second_worst();
        let l = self.config.simplex.best();
        let c = &self.config.simplex.corrected_centroid();
        let mut xr = Point::from(c + (c - &h.x).scale(self.config.alpha));
        xr.evaluate_bounded(problem, bounds, args)?;
        status.inc_n_f_evals();
        if l <= &xr && &xr < s {
            // Reflect if l <= x_r < s
            // In this general case, we just know that r is better than s, we just don't know where
            // it should go. We have to do a sort, but it should be quick since most of the simplex
            // is already sorted.
            self.config
                .simplex
                .insert_and_sort(self.config.simplex.dimension - 2, xr);
            status.with_position(self.config.simplex.best_position(bounds));
            status.with_message("REFLECT");
            self.config.simplex.scale_volume(self.config.alpha);
            return Ok(());
        } else if &xr < l {
            // Expand if x_r < l
            // This means that x_r is certainly the best point so far. We should either expand and
            // accept the expanded point x_e regardless (greedy expansion), or we should do one
            // final comparison between x_r and x_e and choose the smallest (greedy minimization).
            let mut xe = Point::from(c + (&xr.x - c).scale(self.config.beta));
            xe.evaluate_bounded(problem, bounds, args)?;
            status.inc_n_f_evals();
            self.config.simplex.insert_sorted(
                0,
                match self.config.expansion_method {
                    SimplexExpansionMethod::GreedyMinimization => {
                        if xe < xr {
                            xe
                        } else {
                            xr
                        }
                    }
                    SimplexExpansionMethod::GreedyExpansion => xe,
                },
            );
            status.with_position(self.config.simplex.best_position(bounds));
            status.with_message("EXPAND");
            self.config
                .simplex
                .scale_volume(self.config.alpha * self.config.beta);
            return Ok(());
        } else if s <= &xr {
            // Try to contract if s <= x_r
            // This means x_r would just be another worst, although possibly an improvement from the
            // previous worst. If it is better than worst (in between worst and second-worst), we
            // try contracting on the segment c-x_r. Otherwise, x_r would be worse than the worst,
            // but we'll try one more attempt at at least improving it to be better than worst by
            // contracting on the segment h-c.
            // If all else fails, and we don't find improvement, we shrink the whole simplex
            // towards l, the best point.
            //
            // Note that if we had an improving x_r, we still reject it unless contracting improves
            // it, in which we take the improved value. Shouldn't we also accept x_r in the case
            // where x_c is not better, but x_r is still better than h? There must be a reason we
            // don't do this in practice. For instance, if x_r was worse than worst, contracting to
            // make it event the slightest bit better than worst will be accepted.
            if &xr < h {
                // Try to contract outside if x_r < h
                let mut xc = Point::from(c + (&xr.x - c).scale(self.config.gamma));
                xc.evaluate_bounded(problem, bounds, args)?;
                status.inc_n_f_evals();
                if xc <= xr {
                    if &xc < s {
                        // If we are better than the second-worst, we need to sort everything, we
                        // could technically be anywhere, even in a new best.
                        self.config
                            .simplex
                            .insert_and_sort(self.config.simplex.dimension - 1, xc);
                        status.with_position(self.config.simplex.best_position(bounds));
                    } else {
                        // Otherwise, we don't even need to update the best position, this was just
                        // a new worst or equal to second worst.
                        self.config
                            .simplex
                            .insert_sorted(self.config.simplex.dimension - 1, xc);
                    }
                    status.with_message("CONTRACT OUT");
                    self.config
                        .simplex
                        .scale_volume(self.config.alpha * self.config.gamma);
                    return Ok(());
                }
                // TODO: else try accepting x_r here?
            } else {
                // Contract inside if h <= x_r
                let mut xc = Point::from(c + (&h.x - c).scale(self.config.gamma));
                xc.evaluate_bounded(problem, bounds, args)?;
                status.inc_n_f_evals();
                if &xc < h {
                    if &xc < s {
                        // If we are better than the second-worst, we need to sort everything, we
                        // could technically be anywhere, even in a new best.
                        self.config
                            .simplex
                            .insert_and_sort(self.config.simplex.dimension - 1, xc);
                        status.with_position(self.config.simplex.best_position(bounds));
                    } else {
                        // Otherwise, we don't even need to update the best position, this was just
                        // a new worst or equal to second worst.
                        self.config
                            .simplex
                            .insert_sorted(self.config.simplex.dimension - 1, xc);
                    }
                    status.with_message("CONTRACT IN");
                    self.config.simplex.scale_volume(self.config.gamma);
                    return Ok(());
                }
            }
        }
        // If no point is accepted, shrink
        let l_clone = l.clone();
        for p in self.config.simplex.points.iter_mut().skip(1) {
            *p = Point::from(&l_clone.x + (&p.x - &l_clone.x).scale(self.config.delta));
            p.evaluate_bounded(problem, bounds, args)?;
            status.inc_n_f_evals();
        }
        // We must do a fresh sort here, since we don't know the ordering of the shrunken simplex,
        // things might have moved around a lot!
        self.config.simplex.sorted = false;
        self.config.simplex.sort();
        // We also need to recalculate the centroid and figure out if there's a new best position:
        self.config.simplex.compute_total_centroid();
        status.with_position(self.config.simplex.best_position(bounds));
        status.with_message("SHRINK");
        self.config.simplex.scale_volume(Float::powi(
            self.config.delta,
            self.config.simplex.dimension as i32,
        ));
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _func: &P,
        status: &GradientFreeStatus,
        _args: &U,
    ) -> Result<MinimizationSummary, E> {
        Ok(MinimizationSummary {
            x0: self.config.x0.clone(),
            x: status.x.clone(),
            fx: status.fx,
            bounds: self.config.bounds.clone(),
            converged: status.converged,
            cost_evals: status.n_f_evals,
            gradient_evals: 0,
            message: status.message.clone(),
            parameter_names: None,
            std: status
                .err
                .clone()
                .unwrap_or_else(|| DVector::from_element(status.x.len(), 0.0)),
            covariance: status
                .cov
                .clone()
                .unwrap_or_else(|| DMatrix::identity(status.x.len(), status.x.len())),
        })
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientFreeStatus, U, E>
    where
        Self: Sized,
    {
        Callbacks::empty()
            .with_terminator(NelderMeadFTerminator::default())
            .with_terminator(NelderMeadXTerminator::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::MaxSteps, test_functions::Rosenbrock};
    use approx::assert_relative_eq;
    use std::convert::Infallible;

    #[test]
    fn test_nelder_mead() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };
        let starting_values = vec![
            [-2.0, 2.0],
            [2.0, 2.0],
            [2.0, -2.0],
            [-2.0, -2.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ];
        for starting_value in starting_values {
            let result = solver
                .process(
                    &mut problem,
                    &(),
                    NelderMeadConfig::default().with_x0(starting_value),
                    NelderMead::default_callbacks().with_terminator(MaxSteps(1_000_000)),
                )
                .unwrap();
            assert!(result.converged);
            assert_relative_eq!(result.fx, 0.0, epsilon = Float::EPSILON.powf(0.2));
        }
    }

    #[test]
    fn test_bounded_nelder_mead() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };
        let starting_values = vec![
            [-2.0, 2.0],
            [2.0, 2.0],
            [2.0, -2.0],
            [-2.0, -2.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ];
        for starting_value in starting_values {
            let result = solver
                .process(
                    &mut problem,
                    &(),
                    NelderMeadConfig::default()
                        .with_x0(starting_value)
                        .with_bounds([(-4.0, 4.0), (-4.0, 4.0)]),
                    NelderMead::default_callbacks().with_terminator(MaxSteps(1_000_000)),
                )
                .unwrap();
            assert!(result.converged);
            assert_relative_eq!(result.fx, 0.0, epsilon = Float::EPSILON.powf(0.2));
        }
    }

    #[test]
    fn test_adaptive_nelder_mead() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };
        let starting_values = vec![
            [-2.0, 2.0],
            [2.0, 2.0],
            [2.0, -2.0],
            [-2.0, -2.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ];
        for starting_value in starting_values {
            let result = solver
                .process(
                    &mut problem,
                    &(),
                    NelderMeadConfig::default()
                        .with_x0(starting_value)
                        .with_adaptive(2),
                    NelderMead::default_callbacks().with_terminator(MaxSteps(1_000_000)),
                )
                .unwrap();
            assert!(result.converged);
            assert_relative_eq!(result.fx, 0.0, epsilon = Float::EPSILON.powf(0.2));
        }
    }

    fn point(x: &[Float], fx: Float) -> Point<DVector<Float>> {
        Point {
            x: DVector::from_column_slice(x),
            fx,
        }
    }
    #[test]
    fn test_corrected_centroid() {
        let pts = vec![
            point(&[1.0, 2.0], 1.0),
            point(&[2.0, 3.0], 2.0),
            point(&[3.0, 4.0], 3.0),
        ];
        let simplex = Simplex::new(&pts);

        let expected = (&pts[0].x + &pts[1].x) / 2.0;

        let actual = simplex.corrected_centroid();
        assert_eq!(actual, expected);
    }
    #[test]
    fn test_insert_sorted() {
        let mut simplex = Simplex::new(&[
            point(&[0.0, 0.0], 0.0),
            point(&[1.0, 1.0], 1.0),
            point(&[2.0, 2.0], 2.0),
        ]);

        let original_total = simplex.total_centroid.clone();

        let new_point = point(&[3.0, 3.0], 1.5);
        simplex.insert_sorted(1, new_point.clone());

        let expected_total = &original_total + (&new_point.x - &point(&[2.0, 2.0], 2.0).x) / 3.0;

        assert_eq!(simplex.total_centroid.clone(), expected_total);
    }
    #[test]
    fn test_insert_and_sort() {
        let mut simplex = Simplex::new(&[
            point(&[5.0, 0.0], 5.0),
            point(&[1.0, 1.0], 1.0),
            point(&[2.0, 2.0], 2.0),
        ]);

        let original_total = simplex.total_centroid.clone();
        let new_point = point(&[0.5, 0.5], 0.2);

        simplex.insert_and_sort(0, new_point.clone());

        let expected_total = &original_total + (&new_point.x - &point(&[5.0, 0.0], 5.0).x) / 3.0;

        assert_eq!(simplex.best(), &new_point);
        assert_eq!(simplex.total_centroid.clone(), expected_total);
    }

    #[test]
    fn terminates_with_f_amoeba() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };

        let cfg = NelderMeadConfig::default()
            .with_x0([0.5, -0.5])
            .with_eps_f_rel(0.01); // helps ensure quick convergence

        let callbacks = Callbacks::empty().with_terminator(NelderMeadFTerminator::Amoeba);

        let result = solver.process(&mut problem, &(), cfg, callbacks).unwrap();
        assert!(result.converged);
        assert_eq!(result.message, "term_f = AMOEBA");
    }

    #[test]
    fn terminates_with_f_absolute() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };

        let cfg = NelderMeadConfig::default().with_x0([0.5, -0.5]);

        let callbacks = Callbacks::empty().with_terminator(NelderMeadFTerminator::Absolute);

        let result = solver.process(&mut problem, &(), cfg, callbacks).unwrap();
        assert!(result.converged);
        assert_eq!(result.message, "term_f = ABSOLUTE");
    }

    #[test]
    fn terminates_with_f_stddev() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };

        let cfg = NelderMeadConfig::default().with_x0([0.5, -0.5]);

        let callbacks = Callbacks::empty().with_terminator(NelderMeadFTerminator::StdDev);

        let result = solver.process(&mut problem, &(), cfg, callbacks).unwrap();
        assert!(result.converged);
        assert_eq!(result.message, "term_f = STDDEV");
    }

    #[test]
    fn terminates_with_x_diameter() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };

        let cfg = NelderMeadConfig::default().with_x0([0.5, -0.5]);

        let callbacks = Callbacks::empty().with_terminator(NelderMeadXTerminator::Diameter);

        let result = solver.process(&mut problem, &(), cfg, callbacks).unwrap();
        assert!(result.converged);
        assert_eq!(result.message, "term_x = DIAMETER");
    }

    #[test]
    fn terminates_with_x_higham() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };

        let cfg = NelderMeadConfig::default().with_x0([0.5, -0.5]);

        let callbacks = Callbacks::empty().with_terminator(NelderMeadXTerminator::Higham);

        let result = solver.process(&mut problem, &(), cfg, callbacks).unwrap();
        assert!(result.converged);
        assert_eq!(result.message, "term_x = HIGHAM");
    }

    #[test]
    fn terminates_with_x_rowan() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };

        let cfg = NelderMeadConfig::default().with_x0([0.5, -0.5]);

        let callbacks = Callbacks::empty().with_terminator(NelderMeadXTerminator::Rowan);

        let result = solver.process(&mut problem, &(), cfg, callbacks).unwrap();
        assert!(result.converged);
        assert_eq!(result.message, "term_x = ROWAN");
    }

    #[test]
    fn terminates_with_x_singer() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };

        let cfg = NelderMeadConfig::default().with_x0([0.5, -0.5]);

        let callbacks = Callbacks::empty().with_terminator(NelderMeadXTerminator::Singer);

        let result = solver.process(&mut problem, &(), cfg, callbacks).unwrap();
        assert!(result.converged);
        assert_eq!(result.message, "term_x = SINGER");
    }

    #[test]
    fn simplex_total_centroid_matches_mean() {
        let simplex = Simplex::new(&[
            point(&[0.0, 0.0], 3.0),
            point(&[2.0, 0.0], 2.0),
            point(&[0.0, 2.0], 1.0),
        ]);
        let expected =
            (&point(&[0.0, 0.0], 0.0).x + &point(&[2.0, 0.0], 0.0).x + &point(&[0.0, 2.0], 0.0).x)
                / 3.0;
        assert_relative_eq!(simplex.total_centroid, expected);
    }

    #[test]
    fn simplex_scale_volume_multiplies() {
        let mut simplex = Simplex::new(&[
            point(&[0.0, 0.0], 3.0),
            point(&[2.0, 0.0], 2.0),
            point(&[0.0, 2.0], 1.0),
        ]);
        let v0 = simplex.volume;
        simplex.scale_volume(2.5);
        assert_relative_eq!(simplex.volume, v0 * 2.5);
    }

    #[test]
    #[should_panic(
        expected = "Nelder-Mead is only a suitable method for problems of dimension >= 2"
    )]
    fn orthogonal_simplex_panics_in_1d() {
        let method = SimplexConstructionMethod::Orthogonal { simplex_size: 1.0 };
        let problem = Rosenbrock { n: 1 };
        // x0 has dimension 1 → should panic inside generate
        let _ = method
            .generate::<_, Infallible>(&problem, &[0.0], None, &())
            .unwrap();
    }

    #[test]
    fn custom_simplex_ignores_x0_and_sorts_by_fx() {
        let method = SimplexConstructionMethod::Custom {
            simplex: vec![vec![2.0, 2.0], vec![1.0, 1.0], vec![0.0, 0.0]],
        };
        let problem = Rosenbrock { n: 2 };
        let simplex = method
            .generate::<_, Infallible>(&problem, &[99.0, 99.0], None, &())
            .unwrap();

        // Global min at (1,1) for Rosenbrock
        assert_relative_eq!(simplex.best().x[0], 1.0);
        assert_relative_eq!(simplex.best().x[1], 1.0);
        assert!(simplex.best().fx <= simplex.second_worst().fx);
        assert!(simplex.second_worst().fx <= simplex.worst().fx);
    }

    #[test]
    fn adaptive_parameters_match_gao_han() {
        // For n=2, ANMS sets: alpha=1, beta=1+2/n=2, gamma=0.75-1/(2n)=0.5, delta=1-1/n=0.5
        let cfg = NelderMeadConfig::default().with_adaptive(2);
        assert_relative_eq!(cfg.alpha, 1.0);
        assert_relative_eq!(cfg.beta, 2.0);
        assert_relative_eq!(cfg.gamma, 0.5);
        assert_relative_eq!(cfg.delta, 0.5);
    }

    #[test]
    fn expansion_and_construction_method_switches_are_accepted() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };
        let result = solver
            .process(
                &mut problem,
                &(),
                NelderMeadConfig::default()
                    .with_expansion_method(SimplexExpansionMethod::GreedyExpansion)
                    .with_construction_method(SimplexConstructionMethod::Custom {
                        simplex: vec![vec![0.5, -0.5], vec![1.5, -0.5], vec![0.5, 0.5]],
                    }),
                NelderMead::default_callbacks(),
            )
            .unwrap();
        assert!(result.converged);
    }

    #[test]
    #[should_panic]
    fn with_alpha_panics_on_nonpositive() {
        let _ = NelderMeadConfig::default().with_alpha(0.0);
    }

    #[test]
    #[should_panic]
    fn with_beta_panics_when_not_gt_one() {
        let _ = NelderMeadConfig::default().with_beta(1.0);
    }

    #[test]
    #[should_panic]
    fn with_beta_panics_when_not_gt_alpha() {
        let _ = NelderMeadConfig::default().with_alpha(1.5).with_beta(1.4);
    }

    #[test]
    #[should_panic]
    fn with_gamma_panics_if_not_in_unit() {
        // gamma must be in (0,1)
        let _ = NelderMeadConfig::default().with_gamma(0.0);
    }

    #[test]
    #[should_panic]
    fn with_delta_panics_if_not_in_unit() {
        // delta must be in (0,1)
        let _ = NelderMeadConfig::default().with_delta(1.0);
    }

    #[test]
    #[should_panic]
    fn with_eps_x_rel_panics_on_nonpositive() {
        let _ = NelderMeadConfig::default().with_eps_x_rel(0.0);
    }

    #[test]
    #[should_panic]
    fn with_eps_x_abs_panics_on_nonpositive() {
        let _ = NelderMeadConfig::default().with_eps_x_abs(0.0);
    }

    #[test]
    #[should_panic]
    fn with_eps_f_rel_panics_on_nonpositive() {
        let _ = NelderMeadConfig::default().with_eps_f_rel(0.0);
    }

    #[test]
    #[should_panic]
    fn with_eps_f_abs_panics_on_nonpositive() {
        let _ = NelderMeadConfig::default().with_eps_f_abs(0.0);
    }

    #[test]
    fn check_bounds_and_num_gradient_evals() {
        let mut solver = NelderMead::default();
        let mut problem = Rosenbrock { n: 2 };
        let result = solver
            .process(
                &mut problem,
                &(),
                NelderMeadConfig::default()
                    .with_x0([-3.0, 3.0])
                    .with_bounds([(-4.0, 4.0), (-4.0, 4.0)]),
                NelderMead::default_callbacks().with_terminator(MaxSteps(200_000)),
            )
            .unwrap();
        assert!(result.converged);
        assert_eq!(result.gradient_evals, 0);
    }
}
