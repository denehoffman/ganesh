use std::fmt::Debug;

use nalgebra::{DMatrix, DVector};

use crate::{algorithms::Point, Algorithm, Bound, Float, Function, Status};

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
        func: &dyn Function<U, E>,
        x0: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
    ) -> Result<Simplex, E> {
        match self {
            Self::Orthogonal { simplex_size } => {
                let mut points = Vec::default();
                let mut point_0 = Point::from(Bound::to_unbounded(x0, bounds));
                point_0.evaluate_bounded(func, bounds, user_data)?;
                points.push(point_0.clone());
                let dim = point_0.len();
                assert!(
                    dim >= 2,
                    "Nelder-Mead is only a suitable method for problems of dimension >= 2"
                );
                for i in 0..dim {
                    let mut point_i = point_0.clone();
                    point_i.x[i] += *simplex_size;
                    point_i.evaluate_bounded(func, bounds, user_data)?;
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
                            let mut point_i = Point::from(Bound::to_unbounded(x, bounds));
                            point_i.evaluate_bounded(func, bounds, user_data)?;
                            Ok(point_i)
                        })
                        .collect::<Result<Vec<Point>, E>>()?,
                ))
            }
        }
    }
}

/// A [`Simplex`] represents a list of [`Point`]s. This particular implementation is intended to be
/// sorted.
#[derive(Default, Clone)]
pub struct Simplex {
    points: Vec<Point>,
    dimension: usize,
    sorted: bool,
    centroid: DVector<Float>,
    volume: Float,
    initial_best: Point,
    initial_worst: Point,
    initial_volume: Float,
}
impl Debug for Simplex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.points)
    }
}
impl Simplex {
    fn new(points: &[Point]) -> Self {
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
        let dim = n_params as Float;
        let centroid: DVector<Float> = sorted_points
            .iter()
            .rev()
            .skip(1)
            .map(|p| (&p.x / dim))
            .sum();
        Self {
            points: sorted_points,
            dimension: points.len(),
            sorted: false,
            centroid,
            volume,
            initial_best,
            initial_worst,
            initial_volume: volume,
        }
    }
    fn best_position(&self, bounds: Option<&Vec<Bound>>) -> (DVector<Float>, Float) {
        let (y, fx) = self.best().clone().into_vec_val();
        (Bound::to_bounded(&y, bounds), fx)
    }
    fn best(&self) -> &Point {
        &self.points[0]
    }
    fn worst(&self) -> &Point {
        &self.points[self.points.len() - 1]
    }
    fn second_worst(&self) -> &Point {
        &self.points[self.points.len() - 2]
    }
    fn insert_and_sort(&mut self, index: usize, element: Point) {
        self.points.insert(index, element);
        self.points.pop();
        self.sorted = false;
        self.sort();
        self.compute_centroid();
    }
    fn insert_sorted(&mut self, index: usize, element: Point) {
        self.points.insert(index, element);
        self.points.pop();
        self.sorted = true;
        self.compute_centroid();
    }
    fn sort(&mut self) {
        if !self.sorted {
            self.sorted = true;
            self.points.sort_by(|a, b| a.total_cmp(b));
        }
    }
    fn compute_centroid(&mut self) {
        let dim = (self.points.len() - 1) as Float;
        self.centroid = self.points.iter().rev().skip(1).map(|p| &p.x / dim).sum()
    }
    // TODO: track centroid updates
    #[allow(dead_code)]
    fn centroid_add(&mut self, a: &Point) {
        let dim = (self.points.len() - 1) as Float;
        self.centroid += &a.x / dim;
    }
    // TODO: track centroid updates
    #[allow(dead_code)]
    fn centroid_remove(&mut self, a: &Point) {
        let dim = (self.points.len() - 1) as Float;
        self.centroid -= &a.x / dim;
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
#[derive(Debug, Clone)]
pub enum NelderMeadFTerminator {
    /// For the worst point $`x_h`$ and best point $`x_l`$, converge if the following is true:
    /// ```math
    /// 2 \frac{f(x_h) - f(x_l)}{|f(x_h)| + |f(x_l)|} <= \varepsilon
    /// ```
    Amoeba {
        /// Relative tolerance $`\varepsilon`$.
        tol_f_rel: Float,
    },
    /// For the worst point $`x_h`$ and best point $`x_l`$, converge if the following is true:
    /// ```math
    /// f(x_h) - f(x_l) <= \varepsilon
    /// ```
    Absolute {
        /// Absolute tolerance $`\varepsilon`$.
        tol_f_abs: Float,
    },
    /// Converge if the standard deviation of the function evaluations of all points in the simplex
    /// is $`\sigma <= \varepsilon`$.
    StdDev {
        /// Absolute tolerance $`\varepsilon`$.
        tol_f_abs: Float,
    },
    /// No termination condition.
    None,
}
impl NelderMeadFTerminator {
    fn update_convergence(&self, simplex: &Simplex, status: &mut Status) {
        match self {
            Self::Amoeba { tol_f_rel } => {
                let fh = simplex.worst().fx;
                let fl = simplex.best().fx;
                if 2.0 * (fh - fl) / (Float::abs(fh) + Float::abs(fl)) <= *tol_f_rel {
                    status.set_converged();
                    status.update_message("term_f = AMOEBA");
                }
            }
            Self::Absolute { tol_f_abs } => {
                let fh = simplex.worst().fx;
                let fl = simplex.best().fx;
                if fh - fl <= *tol_f_abs {
                    status.set_converged();
                    status.update_message("term_f = ABSOLUTE");
                }
            }
            Self::StdDev { tol_f_abs } => {
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
                if std_dev <= *tol_f_abs {
                    status.set_converged();
                    status.update_message("term_f = STDDEV")
                }
            }
            Self::None => {}
        }
    }
}

/// Various termination methods based on the the position of points in the simplex.
/// See Singer et al.[^1] for more details.
///
/// [^1]: [S. Singer and S. Singer, ‘Efficient Implementation of the Nelder–Mead Search Algorithm’, Applied Numerical Analysis & Computational Mathematics, vol. 1, no. 2, pp. 524–534, 2004.](https://doi.org/10.1002/anac.200410015)
#[derive(Debug, Clone)]
pub enum NelderMeadXTerminator {
    /// For the best point in the simplex $`x_l`$, converge if the following condition is met:
    /// ```math
    /// \max_{j\neq l} ||x_j - x_l||_{\inf} \leq \varepsilon
    /// ```
    Diameter {
        /// Absolute tolerance $`\varepsilon`$.
        tol_x_abs: Float,
    },
    /// For the best point in the simplex $`x_l`$, converge if the following condition is met:
    /// ```math
    /// \frac{\max_{j\neq l} ||x_j - x_l||_1}{\max\left\{1, ||x_l||_1\right\}} \leq \varepsilon
    /// ```
    Higham {
        /// Relative tolerance $`\varepsilon`$.
        tol_x_rel: Float,
    },
    /// For the worst point $`x_h`$ and best point $`x_l`$, as well as the original values of those
    /// points at the beginning of the algorithm, denoted $`x_h^{(0)}`$ and $`x_l^{(0)}`$
    /// respectively, converge if the following condition is met:
    /// ```math
    /// ||x_h - x_l||_2 \leq \varepsilon ||x_h^{(0)} - x_l^{(0)}||_2
    /// ```
    Rowan {
        /// Relative tolerance $`\varepsilon`$.
        tol_x_rel: Float,
    },
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
    Singer {
        /// Relative tolerance $`\varepsilon`$.
        tol_x_rel: Float,
    },
    /// No termination condition.
    None,
}

impl NelderMeadXTerminator {
    fn update_convergence(&self, simplex: &Simplex, status: &mut Status) {
        match self {
            Self::Diameter { tol_x_abs } => {
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
                if max_inf_norm <= *tol_x_abs {
                    status.set_converged();
                    status.update_message("term_x = DIAMETER");
                }
            }
            Self::Higham { tol_x_rel } => {
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
                if numer / denom <= *tol_x_rel {
                    status.set_converged();
                    status.update_message("term_x = HIGHAM");
                }
            }
            Self::Rowan { tol_x_rel } => {
                let init_diff = (&simplex.initial_worst.x - &simplex.initial_best.x).lp_norm(2);
                let current_diff = (&simplex.worst().x - &simplex.best().x).lp_norm(2);
                if current_diff <= *tol_x_rel * init_diff {
                    status.set_converged();
                    status.update_message("term_x = ROWAN");
                }
            }
            Self::Singer { tol_x_rel } => {
                let dim = simplex.dimension as Float;
                let lv_init = Float::powf(simplex.initial_volume, 1.0 / dim);
                let lv_current = Float::powf(simplex.volume, 1.0 / dim);
                if lv_current <= *tol_x_rel * lv_init {
                    status.set_converged();
                    status.update_message("term_x = SINGER");
                }
            }
            Self::None => {}
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
#[derive(Debug, Clone)]
pub struct NelderMead {
    alpha: Float,
    beta: Float,
    gamma: Float,
    delta: Float,
    simplex: Simplex,
    construction_method: SimplexConstructionMethod,
    expansion_method: SimplexExpansionMethod,
    terminator_f: NelderMeadFTerminator,
    terminator_x: NelderMeadXTerminator,
    compute_parameter_errors: bool,
}
impl Default for NelderMead {
    fn default() -> Self {
        Self::new()
    }
}
impl NelderMead {
    /// Create a new Nelder-Mead algorithm with all default values. This is equivalent to
    /// [`NelderMead::default()`].
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 2.0,
            gamma: 0.5,
            delta: 0.5,
            simplex: Simplex::default(),
            construction_method: SimplexConstructionMethod::default(),
            expansion_method: SimplexExpansionMethod::default(),
            terminator_f: NelderMeadFTerminator::StdDev {
                tol_f_abs: Float::EPSILON.powf(0.25),
            },
            terminator_x: NelderMeadXTerminator::Singer {
                tol_x_rel: Float::EPSILON.powf(0.25),
            },
            compute_parameter_errors: true,
        }
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
    /// Set the termination condition concerning the function values.
    pub const fn with_terminator_f(mut self, term: NelderMeadFTerminator) -> Self {
        self.terminator_f = term;
        self
    }
    /// Set the termination condition concerning the simplex positions.
    pub const fn with_terminator_x(mut self, term: NelderMeadXTerminator) -> Self {
        self.terminator_x = term;
        self
    }
    /// Disable covariance calculation upon convergence (not recommended except for testing very large
    /// problems).
    pub const fn with_no_error_calculation(mut self) -> Self {
        self.compute_parameter_errors = false;
        self
    }
}
impl<U, E> Algorithm<U, E> for NelderMead {
    fn initialize(
        &mut self,
        func: &dyn Function<U, E>,
        x0: &[Float],
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        self.simplex = self
            .construction_method
            .generate(func, x0, bounds, user_data)?;
        status.update_position(self.simplex.best_position(bounds));
        Ok(())
    }

    fn step(
        &mut self,
        _i_step: usize,
        func: &dyn Function<U, E>,
        bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        let h = self.simplex.worst();
        let s = self.simplex.second_worst();
        let l = self.simplex.best();
        let c = &self.simplex.centroid;
        let mut xr = Point::from(c + (c - &h.x).scale(self.alpha));
        xr.evaluate_bounded(func, bounds, user_data)?;
        status.inc_n_f_evals();
        if l <= &xr && &xr < s {
            // Reflect if l <= x_r < s
            // In this general case, we just know that r is better than s, we just don't know where
            // it should go. We have to do a sort, but it should be quick since most of the simplex
            // is already sorted.
            self.simplex.insert_and_sort(self.simplex.dimension - 2, xr);
            status.update_position(self.simplex.best_position(bounds));
            status.update_message("REFLECT");
            self.simplex.scale_volume(self.alpha);
            return Ok(());
        } else if &xr < l {
            // Expand if x_r < l
            // This means that x_r is certainly the best point so far. We should either expand and
            // accept the expanded point x_e regardless (greedy expansion), or we should do one
            // final comparison between x_r and x_e and choose the smallest (greedy minimization).
            let mut xe = Point::from(c + (&xr.x - c).scale(self.beta));
            xe.evaluate_bounded(func, bounds, user_data)?;
            status.inc_n_f_evals();
            self.simplex.insert_sorted(
                0,
                match self.expansion_method {
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
            status.update_position(self.simplex.best_position(bounds));
            status.update_message("EXPAND");
            self.simplex.scale_volume(self.alpha * self.beta);
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
                let mut xc = Point::from(c + (&xr.x - c).scale(self.gamma));
                xc.evaluate_bounded(func, bounds, user_data)?;
                status.inc_n_f_evals();
                if xc <= xr {
                    if &xc < s {
                        // If we are better than the second-worst, we need to sort everything, we
                        // could technically be anywhere, even in a new best.
                        self.simplex.insert_and_sort(self.simplex.dimension - 1, xc);
                        status.update_position(self.simplex.best_position(bounds));
                    } else {
                        // Otherwise, we don't even need to update the best position, this was just
                        // a new worst or equal to second worst.
                        self.simplex.insert_sorted(self.simplex.dimension - 1, xc);
                    }
                    status.update_message("CONTRACT OUT");
                    self.simplex.scale_volume(self.alpha * self.gamma);
                    return Ok(());
                }
                // TODO: else try accepting x_r here?
            } else {
                // Contract inside if h <= x_r
                let mut xc = Point::from(c + (&h.x - c).scale(self.gamma));
                xc.evaluate_bounded(func, bounds, user_data)?;
                status.inc_n_f_evals();
                if &xc < h {
                    if &xc < s {
                        // If we are better than the second-worst, we need to sort everything, we
                        // could technically be anywhere, even in a new best.
                        self.simplex.insert_and_sort(self.simplex.dimension - 1, xc);
                        status.update_position(self.simplex.best_position(bounds));
                    } else {
                        // Otherwise, we don't even need to update the best position, this was just
                        // a new worst or equal to second worst.
                        self.simplex.insert_sorted(self.simplex.dimension - 1, xc);
                    }
                    status.update_message("CONTRACT IN");
                    self.simplex.scale_volume(self.gamma);
                    return Ok(());
                }
            }
        }
        // If no point is accepted, shrink
        let l_clone = l.clone();
        for p in self.simplex.points.iter_mut().skip(1) {
            *p = Point::from(&l_clone.x + (&p.x - &l_clone.x).scale(self.delta));
            p.evaluate_bounded(func, bounds, user_data)?;
            status.inc_n_f_evals();
        }
        // We must do a fresh sort here, since we don't know the ordering of the shrunken simplex,
        // things might have moved around a lot!
        self.simplex.sorted = false;
        self.simplex.sort();
        // We also need to recalculate the centroid and figure out if there's a new best position:
        self.simplex.compute_centroid();
        status.update_position(self.simplex.best_position(bounds));
        status.update_message("SHRINK");
        self.simplex
            .scale_volume(Float::powi(self.delta, self.simplex.dimension as i32));
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        _func: &dyn Function<U, E>,
        _bounds: Option<&Vec<Bound>>,
        _user_data: &mut U,
        status: &mut Status,
    ) -> Result<bool, E> {
        self.terminator_x.update_convergence(&self.simplex, status);
        if status.converged {
            return Ok(true);
        }
        self.terminator_f.update_convergence(&self.simplex, status);
        if status.converged {
            return Ok(true);
        }
        Ok(false)
    }

    fn postprocessing(
        &mut self,
        func: &dyn Function<U, E>,
        _bounds: Option<&Vec<Bound>>,
        user_data: &mut U,
        status: &mut Status,
    ) -> Result<(), E> {
        if self.compute_parameter_errors {
            let hessian = func.hessian(status.x.as_slice(), user_data)?;
            status.set_hess(&hessian);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use approx::assert_relative_eq;

    use crate::{test_functions::Rosenbrock, Float, Minimizer};

    use super::NelderMead;

    #[test]
    fn test_nelder_mead() -> Result<(), Infallible> {
        let algo = NelderMead::default();
        let mut m = Minimizer::new(&algo, 2);
        let problem = Rosenbrock { n: 2 };
        m.minimize(&problem, &[-2.0, 2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(0.25));
        m.minimize(&problem, &[2.0, 2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(1.0 / 5.0));
        m.minimize(&problem, &[2.0, -2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(0.25));
        m.minimize(&problem, &[-2.0, -2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(0.25));
        m.minimize(&problem, &[0.0, 0.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(0.25));
        m.minimize(&problem, &[1.0, 1.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }

    #[test]
    fn test_bounded_nelder_mead() -> Result<(), Infallible> {
        let algo = NelderMead::default();
        let mut m = Minimizer::new(&algo, 2).with_bounds(Some(vec![(-4.0, 4.0), (-4.0, 4.0)]));
        let problem = Rosenbrock { n: 2 };
        m.minimize(&problem, &[-2.0, 2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(0.25));
        m.minimize(&problem, &[2.0, 2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(1.0 / 5.0));
        m.minimize(&problem, &[2.0, -2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(0.25));
        m.minimize(&problem, &[-2.0, -2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(1.0 / 5.0));
        m.minimize(&problem, &[0.0, 0.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(1.0 / 5.0));
        m.minimize(&problem, &[1.0, 1.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }

    #[test]
    fn test_adaptive_nelder_mead() -> Result<(), Infallible> {
        let algo = NelderMead::default().with_adaptive(2);
        let mut m = Minimizer::new(&algo, 2);
        let problem = Rosenbrock { n: 2 };
        m.minimize(&problem, &[-2.0, 2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(0.25));
        m.minimize(&problem, &[2.0, 2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(1.0 / 5.0));
        m.minimize(&problem, &[2.0, -2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(0.25));
        m.minimize(&problem, &[-2.0, -2.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(0.25));
        m.minimize(&problem, &[0.0, 0.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.powf(0.25));
        m.minimize(&problem, &[1.0, 1.0], &mut ())?;
        assert!(m.status.converged);
        assert_relative_eq!(m.status.fx, 0.0, epsilon = Float::EPSILON.sqrt());
        Ok(())
    }
}
