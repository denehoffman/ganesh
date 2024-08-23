use std::{cmp::Ordering, fmt::Debug, iter::Sum};

use nalgebra::{DMatrix, DVector};
use num::{traits::NumAssign, Float, FromPrimitive, NumCast};

use crate::{convert, Algorithm, Bound, Function, Status};

#[derive(Eq, PartialEq, Clone, Default, Debug)]
pub struct Point<T>
where
    T: Clone + Debug + Float + 'static,
{
    pub x: DVector<T>,
    pub fx: T,
}
impl<T> Point<T>
where
    T: Clone + Debug + Float,
{
    pub fn len(&self) -> usize {
        self.x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
    pub fn into_vec_val(self) -> (Vec<T>, T) {
        (self.x.data.into(), self.fx)
    }
}
impl<T> Point<T>
where
    T: Float + FromPrimitive + Debug,
{
    pub fn evaluate<U, E>(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        if let Some(bounds) = bounds {
            let x = Bound::to_bounded(self.x.data.as_slice(), bounds);
            self.fx = func.evaluate(&x, user_data)?;
        } else {
            self.fx = func.evaluate(self.x.data.as_slice(), user_data)?;
        }
        Ok(())
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

pub enum SimplexConstructionMethod<T> {
    Orthogonal { simplex_size: T },
    Custom { simplex: Vec<Vec<T>> },
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
    T: Float + Debug + NumAssign + Sum + Default + FromPrimitive + nalgebra::RealField + 'static,
{
    pub fn generate<U, E>(
        &self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<Simplex<T>, E> {
        match self {
            SimplexConstructionMethod::Orthogonal { simplex_size } => {
                let mut points = Vec::default();
                let mut point_0 = if let Some(bounds) = bounds {
                    Point::from(Bound::to_unbounded(x0, bounds))
                } else {
                    Point::from(x0)
                };
                point_0.evaluate(func, bounds, user_data)?;
                points.push(point_0.clone());
                let dim = point_0.len();
                assert!(dim >= 2);
                for i in 0..dim {
                    let mut point_i = point_0.clone();
                    point_i.x[i] += *simplex_size;
                    point_i.evaluate(func, bounds, user_data)?;
                    points.push(point_i);
                }
                Ok(Simplex::new(&points))
            }
            SimplexConstructionMethod::Custom { simplex } => {
                assert!(!simplex.is_empty());
                assert!(simplex.len() == simplex[0].len() + 1);
                assert!(simplex.len() > 2);
                Ok(Simplex::new(
                    &simplex
                        .iter()
                        .map(|x| {
                            let mut point_i = if let Some(bounds) = bounds {
                                Point::from(Bound::to_unbounded(x, bounds))
                            } else {
                                Point::from(x.clone())
                            };
                            point_i.evaluate(func, bounds, user_data)?;
                            Ok(point_i)
                        })
                        .collect::<Result<Vec<Point<T>>, E>>()?,
                ))
            }
        }
    }
}

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
    T: NumCast + Float + Sum + Debug + NumAssign + Default + nalgebra::RealField + 'static,
{
    pub fn new(points: &[Point<T>]) -> Self {
        let mut sorted_points = points.to_vec();
        sorted_points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let initial_best = sorted_points.first().unwrap().clone();
        let initial_worst = sorted_points.last().unwrap().clone();
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
    pub fn best_position(&self, bounds: Option<&Vec<Bound<T>>>) -> (Vec<T>, T) {
        let (y, fx) = self.best().clone().into_vec_val();
        if let Some(bounds) = bounds {
            (Bound::to_bounded(&y, bounds), fx)
        } else {
            (y, fx)
        }
    }
    pub fn best(&self) -> &Point<T> {
        self.points.first().unwrap()
    }
    pub fn worst(&self) -> &Point<T> {
        self.points.last().unwrap()
    }
    pub fn second_worst(&self) -> &Point<T> {
        self.points.iter().nth_back(1).unwrap()
    }
    pub fn insert_and_sort(&mut self, index: usize, element: Point<T>) {
        self.points.insert(index, element);
        self.points.pop();
        self.sorted = false;
        self.sort();
        self.compute_centroid();
    }
    pub fn insert_sorted(&mut self, index: usize, element: Point<T>) {
        self.points.insert(index, element);
        self.points.pop();
        self.sorted = true;
        self.compute_centroid();
    }
    pub fn sort(&mut self) {
        if !self.sorted {
            self.sorted = true;
            self.points
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        }
    }
    pub fn compute_centroid(&mut self) {
        let dim = convert!(self.points.len() - 1, T);
        self.centroid = self.points.iter().rev().skip(1).map(|p| &p.x / dim).sum()
    }
    pub fn centroid_add(&mut self, a: &Point<T>) {
        let dim = convert!(self.points.len() - 1, T);
        self.centroid += &a.x / dim;
    }
    pub fn centroid_remove(&mut self, a: &Point<T>) {
        let dim = convert!(self.points.len() - 1, T);
        self.centroid -= &a.x / dim;
    }
    pub fn scale_volume(&mut self, factor: T) {
        self.volume *= factor;
    }
}

#[derive(Default)]
pub enum SimplexExpansionMethod {
    #[default]
    GreedyMinimization,
    GreedyExpansion,
}

pub enum NelderMeadFTerminator<T> {
    Amoeba { tol_f_rel: T },
    Absolute { tol_f_abs: T },
    StdDev { tol_f_abs: T },
    None,
}
impl<T> NelderMeadFTerminator<T>
where
    T: Float + Debug + NumAssign + Sum + Default + nalgebra::RealField,
{
    pub fn update_convergence(&self, simplex: &Simplex<T>, status: &mut Status<T>) {
        match self {
            NelderMeadFTerminator::Amoeba { tol_f_rel } => {
                let fh = simplex.worst().fx;
                let fl = simplex.best().fx;
                let two = T::one() + T::one();
                if two * (fh - fl) / (Float::abs(fh) + Float::abs(fl)) <= *tol_f_rel {
                    status.set_converged();
                    status.update_message("term_f = AMOEBA");
                }
            }
            NelderMeadFTerminator::Absolute { tol_f_abs } => {
                let fh = simplex.worst().fx;
                let fl = simplex.best().fx;
                if fh - fl <= *tol_f_abs {
                    status.set_converged();
                    status.update_message("term_f = ABSOLUTE");
                }
            }
            NelderMeadFTerminator::StdDev { tol_f_abs } => {
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
            NelderMeadFTerminator::None => {}
        }
    }
}

pub enum NelderMeadXTerminator<T> {
    Diameter { tol_x_abs: T },
    Higham { tol_x_rel: T },
    Rowan { tol_x_rel: T },
    Singer { tol_x_rel: T },
    None,
}

impl<T> NelderMeadXTerminator<T>
where
    T: Float + Debug + NumAssign + Sum + nalgebra::RealField + Default,
{
    pub fn update_convergence(&self, simplex: &Simplex<T>, status: &mut Status<T>) {
        match self {
            NelderMeadXTerminator::Diameter { tol_x_abs } => {
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
                    .max_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
                    .unwrap_or(T::zero());
                if max_inf_norm <= *tol_x_abs {
                    status.set_converged();
                    status.update_message("term_x = DIAMETER");
                }
            }
            NelderMeadXTerminator::Higham { tol_x_rel } => {
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
                    .max_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal))
                    .unwrap_or(T::zero());
                if numer / denom <= *tol_x_rel {
                    status.set_converged();
                    status.update_message("term_x = HIGHAM");
                }
            }
            NelderMeadXTerminator::Rowan { tol_x_rel } => {
                let init_diff = (&simplex.initial_worst.x - &simplex.initial_best.x).lp_norm(2);
                let current_diff = (&simplex.worst().x - &simplex.best().x).lp_norm(2);
                if current_diff <= *tol_x_rel * init_diff {
                    status.set_converged();
                    status.update_message("term_x = ROWAN");
                }
            }
            NelderMeadXTerminator::Singer { tol_x_rel } => {
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
            NelderMeadXTerminator::None => {}
        }
    }
}

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
        }
    }
    pub fn with_alpha(mut self, value: T) -> Self {
        assert!(value > T::zero());
        self.alpha = value;
        self
    }
    pub fn with_beta(mut self, value: T) -> Self {
        assert!(value > T::one());
        assert!(value > self.alpha);
        self.beta = value;
        self
    }
    pub fn with_gamma(mut self, value: T) -> Self {
        assert!(value > T::zero());
        assert!(value < T::one());
        self.gamma = value;
        self
    }
    pub fn with_delta(mut self, value: T) -> Self {
        assert!(value > T::zero());
        assert!(value < T::one());
        self.delta = value;
        self
    }
    pub fn with_adaptive(mut self, n: usize) -> Self {
        let n = convert!(n, T);
        self.alpha = T::one();
        self.beta = T::one() + (convert!(2, T) / n);
        self.gamma = convert!(0.75, T) - T::one() / (convert!(2, T) * n);
        self.delta = T::one() - T::one() / n;
        self
    }
    pub fn with_construction_method(mut self, method: SimplexConstructionMethod<T>) -> Self {
        self.construction_method = method;
        self
    }
    pub fn with_expansion_method(mut self, method: SimplexExpansionMethod) -> Self {
        self.expansion_method = method;
        self
    }
    pub fn with_terminator_f(mut self, term: NelderMeadFTerminator<T>) -> Self {
        self.terminator_f = term;
        self
    }
    pub fn with_terminator_x(mut self, term: NelderMeadXTerminator<T>) -> Self {
        self.terminator_x = term;
        self
    }
}
impl<T, U, E> Algorithm<T, U, E> for NelderMead<T>
where
    T: Float + NumAssign + Debug + FromPrimitive + Sum + nalgebra::RealField + Default + 'static,
{
    fn initialize(
        &mut self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E> {
        self.simplex = self
            .construction_method
            .generate(func, x0, bounds, user_data)?;
        self.status
            .update_position(self.simplex.best_position(bounds));
        Ok(())
    }

    fn step(
        &mut self,
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
        self.status.increment_n_evals();
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
            self.status.increment_n_evals();
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
                self.status.increment_n_evals();
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
                self.status.increment_n_evals();
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
            self.status.increment_n_evals();
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
    ) -> bool {
        self.terminator_x
            .update_convergence(&self.simplex, &mut self.status);
        if self.status.converged {
            return true;
        }
        self.terminator_f
            .update_convergence(&self.simplex, &mut self.status);
        if self.status.converged {
            return true;
        }
        false
    }

    fn get_status(&self) -> &Status<T> {
        &self.status
    }
}
