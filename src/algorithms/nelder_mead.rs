use std::{cmp::Ordering, fmt::Debug, iter::Sum};

use nalgebra::{DMatrix, DVector};
use num::{
    traits::{float::TotalOrder, NumAssign},
    Float, FromPrimitive, NumCast,
};

use crate::{convert, Algorithm, Bound, Function, Status};

/// Describes a point in a [`Simplex`].
#[derive(Eq, PartialEq, Clone, Default, Debug)]
pub struct Point<T>
where
    T: Clone + Debug + Float + 'static,
{
    x: DVector<T>,
    fx: T,
}
impl<T> Point<T>
where
    T: Clone + Debug + Float,
{
    fn len(&self) -> usize {
        self.x.len()
    }
    fn into_vec_val(self) -> (Vec<T>, T) {
        (self.x.data.into(), self.fx)
    }
}
impl<T> Point<T>
where
    T: Float + FromPrimitive + Debug + NumAssign + TotalOrder,
{
    fn evaluate<U, E>(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.fx = func.evaluate_bounded(self.x.as_slice(), bounds, user_data)?;
        Ok(())
    }
    fn total_cmp(&self, other: &Self) -> Ordering {
        self.fx.total_cmp(&other.fx)
    }
}
impl<T> From<DVector<T>> for Point<T>
where
    T: Float + Debug,
{
    fn from(value: DVector<T>) -> Self {
        Self {
            x: value,
            fx: T::nan(),
        }
    }
}
impl<T> From<Vec<T>> for Point<T>
where
    T: Float + Debug + 'static,
{
    fn from(value: Vec<T>) -> Self {
        Self {
            x: DVector::from_vec(value),
            fx: T::nan(),
        }
    }
}
impl<'a, T> From<&'a Point<T>> for &'a Vec<T>
where
    T: Debug + Float,
{
    fn from(value: &'a Point<T>) -> Self {
        value.x.data.as_vec()
    }
}
impl<T> From<&[T]> for Point<T>
where
    T: Float + Debug + 'static,
{
    fn from(value: &[T]) -> Self {
        Self {
            x: DVector::from_column_slice(value),
            fx: T::nan(),
        }
    }
}
impl<'a, T> From<&'a Point<T>> for &'a [T]
where
    T: Debug + Float,
{
    fn from(value: &'a Point<T>) -> Self {
        value.x.data.as_slice()
    }
}
impl<T> PartialOrd for Point<T>
where
    T: PartialOrd + Debug + Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.fx.partial_cmp(&other.fx)
    }
}

/// Gives a method for constructing a simplex.
#[derive(Debug)]
pub enum SimplexConstructionMethod<T> {
    /// Creates a simplex by starting at the given `x0` and stepping a distance of `+simplex_size`
    /// in every orthogonal direction.
    Orthogonal {
        /// The distance from the starting point to each of the other points in the simplex.
        simplex_size: T,
    },
    /// Creates a custom simplex from a list of points.
    Custom {
        /// The points to use in the simplex (ignores any given starting point).
        simplex: Vec<Vec<T>>,
    },
}
impl<T> Default for SimplexConstructionMethod<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::Orthogonal {
            simplex_size: T::one(),
        }
    }
}

impl<T> SimplexConstructionMethod<T>
where
    T: Float
        + Debug
        + NumAssign
        + Sum
        + Default
        + FromPrimitive
        + nalgebra::RealField
        + 'static
        + TotalOrder,
{
    fn generate<U, E>(
        &self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<Simplex<T>, E> {
        match self {
            Self::Orthogonal { simplex_size } => {
                let mut points = Vec::default();
                let mut point_0 = Point::from(Bound::to_unbounded(x0, bounds));
                point_0.evaluate(func, bounds, user_data)?;
                points.push(point_0.clone());
                let dim = point_0.len();
                assert!(
                    dim >= 2,
                    "Nelder-Mead is only a suitable method for problems of dimension >= 2"
                );
                for i in 0..dim {
                    let mut point_i = point_0.clone();
                    point_i.x[i] += *simplex_size;
                    point_i.evaluate(func, bounds, user_data)?;
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
                            point_i.evaluate(func, bounds, user_data)?;
                            Ok(point_i)
                        })
                        .collect::<Result<Vec<Point<T>>, E>>()?,
                ))
            }
        }
    }
}

/// A [`Simplex`] represents a list of [`Point`]s. This particular implementation is intended to be
/// sorted.
#[derive(Default)]
pub struct Simplex<T>
where
    T: Debug + Float + 'static,
{
    points: Vec<Point<T>>,
    dimension: usize,
    sorted: bool,
    centroid: DVector<T>,
    volume: T,
    initial_best: Point<T>,
    initial_worst: Point<T>,
    initial_volume: T,
}
impl<T> Debug for Simplex<T>
where
    T: Debug + Float,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.points)
    }
}
impl<T> Simplex<T>
where
    T: NumCast
        + Float
        + Sum
        + Debug
        + NumAssign
        + Default
        + nalgebra::RealField
        + 'static
        + TotalOrder,
{
    fn new(points: &[Point<T>]) -> Self {
        let mut sorted_points = points.to_vec();
        sorted_points.sort_by(|a, b| a.total_cmp(b));
        let initial_best = sorted_points[0].clone();
        let initial_worst = sorted_points[sorted_points.len() - 1].clone();
        let n_params = points.len() - 1;
        let diffs: Vec<DVector<T>> = sorted_points
            .iter()
            .skip(1)
            .map(|p| &p.x - &initial_best.x)
            .collect();
        let gram_mat = DMatrix::from_fn(n_params, n_params, |i, j| diffs[i].dot(&diffs[j]));
        // NOTE: volume calculation is off by a constant 1/n! which divides out on both sides
        // whenever we use this!
        let volume = Float::sqrt(gram_mat.determinant());
        let dim = convert!(n_params, T);
        let centroid: DVector<T> = sorted_points
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
    fn best_position(&self, bounds: Option<&Vec<Bound<T>>>) -> (DVector<T>, T) {
        let (y, fx) = self.best().clone().into_vec_val();
        (Bound::to_bounded(&y, bounds), fx)
    }
    fn best(&self) -> &Point<T> {
        &self.points[0]
    }
    fn worst(&self) -> &Point<T> {
        &self.points[self.points.len() - 1]
    }
    fn second_worst(&self) -> &Point<T> {
        &self.points[self.points.len() - 2]
    }
    fn insert_and_sort(&mut self, index: usize, element: Point<T>) {
        self.points.insert(index, element);
        self.points.pop();
        self.sorted = false;
        self.sort();
        self.compute_centroid();
    }
    fn insert_sorted(&mut self, index: usize, element: Point<T>) {
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
        let dim = convert!(self.points.len() - 1, T);
        self.centroid = self.points.iter().rev().skip(1).map(|p| &p.x / dim).sum()
    }
    // TODO: track centroid updates
    #[allow(dead_code)]
    fn centroid_add(&mut self, a: &Point<T>) {
        let dim = convert!(self.points.len() - 1, T);
        self.centroid += &a.x / dim;
    }
    // TODO: track centroid updates
    #[allow(dead_code)]
    fn centroid_remove(&mut self, a: &Point<T>) {
        let dim = convert!(self.points.len() - 1, T);
        self.centroid -= &a.x / dim;
    }
    fn scale_volume(&mut self, factor: T) {
        self.volume *= factor;
    }
}

/// Selects the expansion method used in the Nelder-Mead algorithm. See Lagarias et al.[^1] for more details.
///
/// [^1]: [J. C. Lagarias, J. A. Reeds, M. H. Wright, and P. E. Wright, ‘Convergence Properties of the Nelder--Mead Simplex Method in Low Dimensions’, SIAM Journal on Optimization, vol. 9, no. 1, pp. 112–147, 1998.](https://doi.org/10.1137/S1052623496303470)
#[derive(Default, Debug)]
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
#[derive(Debug)]
pub enum NelderMeadFTerminator<T> {
    /// For the worst point $`x_h`$ and best point $`x_l`$, converge if the following is true:
    /// ```math
    /// 2 \frac{f(x_h) - f(x_l)}{|f(x_h)| + |f(x_l)|} <= \varepsilon
    /// ```
    Amoeba {
        /// Relative tolerance $`\varepsilon`$.
        tol_f_rel: T,
    },
    /// For the worst point $`x_h`$ and best point $`x_l`$, converge if the following is true:
    /// ```math
    /// f(x_h) - f(x_l) <= \varepsilon
    /// ```
    Absolute {
        /// Absolute tolerance $`\varepsilon`$.
        tol_f_abs: T,
    },
    /// Converge if the standard deviation of the function evaluations of all points in the simplex
    /// is $`\sigma <= \varepsilon`$.
    StdDev {
        /// Absolute tolerance $`\varepsilon`$.
        tol_f_abs: T,
    },
    /// No termination condition.
    None,
}
impl<T> NelderMeadFTerminator<T>
where
    T: Float + Debug + NumAssign + Sum + Default + nalgebra::RealField + TotalOrder,
{
    fn update_convergence(&self, simplex: &Simplex<T>, status: &mut Status<T>) {
        match self {
            Self::Amoeba { tol_f_rel } => {
                let fh = simplex.worst().fx;
                let fl = simplex.best().fx;
                let two = T::one() + T::one();
                if two * (fh - fl) / (Float::abs(fh) + Float::abs(fl)) <= *tol_f_rel {
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
                let dim = convert!(simplex.dimension, T);
                let mean = simplex.points.iter().map(|point| point.fx).sum::<T>() / dim;
                let std_dev = Float::sqrt(
                    simplex
                        .points
                        .iter()
                        .map(|point| Float::powi(point.fx - mean, 2))
                        .sum::<T>()
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
#[derive(Debug)]
pub enum NelderMeadXTerminator<T> {
    /// For the best point in the simplex $`x_l`$, converge if the following condition is met:
    /// ```math
    /// \max_{j\neq l} ||x_j - x_l||_{\inf} \leq \varepsilon
    /// ```
    Diameter {
        /// Absolute tolerance $`\varepsilon`$.
        tol_x_abs: T,
    },
    /// For the best point in the simplex $`x_l`$, converge if the following condition is met:
    /// ```math
    /// \frac{\max_{j\neq l} ||x_j - x_l||_1}{\max\left\{1, ||x_l||_1\right\}} \leq \varepsilon
    /// ```
    Higham {
        /// Relative tolerance $`\varepsilon`$.
        tol_x_rel: T,
    },
    /// For the worst point $`x_h`$ and best point $`x_l`$, as well as the original values of those
    /// points at the beginning of the algorithm, denoted $`x_h^{(0)}`$ and $`x_l^{(0)}`$
    /// respectively, converge if the following condition is met:
    /// ```math
    /// ||x_h - x_l||_2 \leq \varepsilon ||x_h^{(0)} - x_l^{(0)}||_2
    /// ```
    Rowan {
        /// Relative tolerance $`\varepsilon`$.
        tol_x_rel: T,
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
        tol_x_rel: T,
    },
    /// No termination condition.
    None,
}

impl<T> NelderMeadXTerminator<T>
where
    T: Float + Debug + NumAssign + Sum + nalgebra::RealField + Default + TotalOrder,
{
    fn update_convergence(&self, simplex: &Simplex<T>, status: &mut Status<T>) {
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
                        let mut inf_norm = T::zero();
                        for i in 0..diff.len() {
                            if inf_norm < Float::abs(diff[i]) {
                                inf_norm = Float::abs(diff[i])
                            }
                        }
                        inf_norm
                    })
                    .max_by(|&a, &b| a.total_cmp(&b))
                    .unwrap_or_else(T::zero);
                if max_inf_norm <= *tol_x_abs {
                    status.set_converged();
                    status.update_message("term_x = DIAMETER");
                }
            }
            Self::Higham { tol_x_rel } => {
                let l = simplex.worst();
                let l1_norm_l = l.x.lp_norm(1);
                let denom = Float::max(l1_norm_l, T::one());
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
                    .unwrap_or_else(T::zero);
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
                let lv_init = Float::powf(
                    simplex.initial_volume,
                    T::one() / convert!(simplex.dimension, T),
                );
                let lv_current =
                    Float::powf(simplex.volume, T::one() / convert!(simplex.dimension, T));
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
#[derive(Debug)]
pub struct NelderMead<T>
where
    T: Float + Debug + 'static,
{
    status: Status<T>,
    alpha: T,
    beta: T,
    gamma: T,
    delta: T,
    simplex: Simplex<T>,
    construction_method: SimplexConstructionMethod<T>,
    expansion_method: SimplexExpansionMethod,
    terminator_f: NelderMeadFTerminator<T>,
    terminator_x: NelderMeadXTerminator<T>,
    compute_parameter_errors: bool,
}
impl<T> Default for NelderMead<T>
where
    T: Float + NumCast + Debug + Default + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}
impl<T> NelderMead<T>
where
    T: Float + NumCast + Debug + Default + 'static,
{
    /// Create a new Nelder-Mead algorithm with all default values. This is equivalent to
    /// [`NelderMead::default()`].
    pub fn new() -> Self {
        Self {
            status: Status::default(),
            alpha: convert!(1, T),
            beta: convert!(2, T),
            gamma: convert!(0.5, T),
            delta: convert!(0.5, T),
            simplex: Simplex::default(),
            construction_method: SimplexConstructionMethod::default(),
            expansion_method: SimplexExpansionMethod::default(),
            terminator_f: NelderMeadFTerminator::StdDev {
                tol_f_abs: T::epsilon(),
            },
            terminator_x: NelderMeadXTerminator::Singer {
                tol_x_rel: T::epsilon(),
            },
            compute_parameter_errors: true,
        }
    }
    /// Set the reflection coefficient $`\alpha`$ (default = `1`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\alpha <= 0`$.
    pub fn with_alpha(mut self, value: T) -> Self {
        assert!(value > T::zero());
        self.alpha = value;
        self
    }
    /// Set the expansion coefficient $`\beta`$ (default = `2`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\beta <= 1`$ or $`\beta <= \alpha`$.
    pub fn with_beta(mut self, value: T) -> Self {
        assert!(value > T::one());
        assert!(value > self.alpha);
        self.beta = value;
        self
    }
    /// Set the contraction coefficient $`\gamma`$ (default = `0.5`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\gamma >= 1`$ or $`\gamma <= 0`$.
    pub fn with_gamma(mut self, value: T) -> Self {
        assert!(value > T::zero());
        assert!(value < T::one());
        self.gamma = value;
        self
    }
    /// Set the shrink coefficient $`\delta`$ (default = `0.5`).
    ///
    /// # Panics
    ///
    /// This method will panic if $`\delta >= 1`$ or $`\delta <= 0`$.
    pub fn with_delta(mut self, value: T) -> Self {
        assert!(value > T::zero());
        assert!(value < T::one());
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
        let n = convert!(n, T);
        self.alpha = T::one();
        self.beta = T::one() + (convert!(2, T) / n);
        self.gamma = convert!(0.75, T) - T::one() / (convert!(2, T) * n);
        self.delta = T::one() - T::one() / n;
        self
    }
    /// Use the given [`SimplexConstructionMethod`] to compute the starting [`Simplex`].
    pub fn with_construction_method(mut self, method: SimplexConstructionMethod<T>) -> Self {
        self.construction_method = method;
        self
    }
    /// Set the [`SimplexExpansionMethod`].
    pub const fn with_expansion_method(mut self, method: SimplexExpansionMethod) -> Self {
        self.expansion_method = method;
        self
    }
    /// Set the termination condition concerning the function values.
    pub const fn with_terminator_f(mut self, term: NelderMeadFTerminator<T>) -> Self {
        self.terminator_f = term;
        self
    }
    /// Set the termination condition concerning the simplex positions.
    pub const fn with_terminator_x(mut self, term: NelderMeadXTerminator<T>) -> Self {
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
impl<T, U, E> Algorithm<T, U, E> for NelderMead<T>
where
    T: Float
        + NumAssign
        + Debug
        + FromPrimitive
        + Sum
        + nalgebra::RealField
        + Default
        + 'static
        + TotalOrder,
{
    fn initialize(
        &mut self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.status = Status::default();
        self.simplex = self
            .construction_method
            .generate(func, x0, bounds, user_data)?;
        self.status
            .update_position(self.simplex.best_position(bounds));
        Ok(())
    }

    fn step(
        &mut self,
        _i_step: usize,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        let h = self.simplex.worst();
        let s = self.simplex.second_worst();
        let l = self.simplex.best();
        let c = &self.simplex.centroid;
        let mut xr = Point::from(c + (c - &h.x).scale(self.alpha));
        xr.evaluate(func, bounds, user_data)?;
        self.status.inc_n_f_evals();
        if l <= &xr && &xr < s {
            // Reflect if l <= x_r < s
            // In this general case, we just know that r is better than s, we just don't know where
            // it should go. We have to do a sort, but it should be quick since most of the simplex
            // is already sorted.
            self.simplex.insert_and_sort(self.simplex.dimension - 2, xr);
            self.status
                .update_position(self.simplex.best_position(bounds));
            self.status.update_message("REFLECT");
            self.simplex.scale_volume(self.alpha);
            return Ok(());
        } else if &xr < l {
            // Expand if x_r < l
            // This means that x_r is certainly the best point so far. We should either expand and
            // accept the expanded point x_e regardless (greedy expansion), or we should do one
            // final comparison between x_r and x_e and choose the smallest (greedy minimization).
            let mut xe = Point::from(c + (&xr.x - c).scale(self.beta));
            xe.evaluate(func, bounds, user_data)?;
            self.status.inc_n_f_evals();
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
            self.status
                .update_position(self.simplex.best_position(bounds));
            self.status.update_message("EXPAND");
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
                xc.evaluate(func, bounds, user_data)?;
                self.status.inc_n_f_evals();
                if xc <= xr {
                    if &xc < s {
                        // If we are better than the second-worst, we need to sort everything, we
                        // could technically be anywhere, even in a new best.
                        self.simplex.insert_and_sort(self.simplex.dimension - 1, xc);
                        self.status
                            .update_position(self.simplex.best_position(bounds));
                    } else {
                        // Otherwise, we don't even need to update the best position, this was just
                        // a new worst or equal to second worst.
                        self.simplex.insert_sorted(self.simplex.dimension - 1, xc);
                    }
                    self.status.update_message("CONTRACT OUT");
                    self.simplex.scale_volume(self.alpha * self.gamma);
                    return Ok(());
                }
                // TODO: else try accepting x_r here?
            } else {
                // Contract inside if h <= x_r
                let mut xc = Point::from(c + (&h.x - c).scale(self.gamma));
                xc.evaluate(func, bounds, user_data)?;
                self.status.inc_n_f_evals();
                if &xc < h {
                    if &xc < s {
                        // If we are better than the second-worst, we need to sort everything, we
                        // could technically be anywhere, even in a new best.
                        self.simplex.insert_and_sort(self.simplex.dimension - 1, xc);
                        self.status
                            .update_position(self.simplex.best_position(bounds));
                    } else {
                        // Otherwise, we don't even need to update the best position, this was just
                        // a new worst or equal to second worst.
                        self.simplex.insert_sorted(self.simplex.dimension - 1, xc);
                    }
                    self.status.update_message("CONTRACT IN");
                    self.simplex.scale_volume(self.gamma);
                    return Ok(());
                }
            }
        }
        // If no point is accepted, shrink
        let l_clone = l.clone();
        for p in self.simplex.points.iter_mut().skip(1) {
            *p = Point::from(&l_clone.x + (&p.x - &l_clone.x).scale(self.delta));
            p.evaluate(func, bounds, user_data)?;
            self.status.inc_n_f_evals();
        }
        // We must do a fresh sort here, since we don't know the ordering of the shrunken simplex,
        // things might have moved around a lot!
        self.simplex.sorted = false;
        self.simplex.sort();
        // We also need to recalculate the centroid and figure out if there's a new best position:
        self.simplex.compute_centroid();
        self.status
            .update_position(self.simplex.best_position(bounds));
        self.status.update_message("SHRINK");
        self.simplex
            .scale_volume(Float::powi(self.delta, self.simplex.dimension as i32));
        Ok(())
    }

    fn check_for_termination(
        &mut self,
        _func: &dyn Function<T, U, E>,
        _bounds: Option<&Vec<Bound<T>>>,
        _user_data: &mut U,
    ) -> Result<bool, E> {
        self.terminator_x
            .update_convergence(&self.simplex, &mut self.status);
        if self.status.converged {
            return Ok(true);
        }
        self.terminator_f
            .update_convergence(&self.simplex, &mut self.status);
        if self.status.converged {
            return Ok(true);
        }
        Ok(false)
    }

    fn get_status(&self) -> &Status<T> {
        &self.status
    }

    fn postprocessing(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        if self.compute_parameter_errors {
            let hessian = func.hessian_bounded(self.status.x.as_slice(), bounds, user_data)?;
            let mut covariance = hessian.clone().try_inverse();
            if covariance.is_none() {
                covariance = hessian.pseudo_inverse(Float::cbrt(T::epsilon())).ok();
            }
            self.status.set_cov(covariance);
        }
        Ok(())
    }
}
