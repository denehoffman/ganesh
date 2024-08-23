// TODO: Someday
// impl<T, F, U, E> Function<F, U, E> for T
// where
//     T: Fn(&[F], &mut U) -> Result<F, E>,
// {
//     fn evaluate(&self, x: &[F], user_data: &mut U) -> Result<F, E> {
//         self.call((x, user_data))
//     }
// }

use std::{fmt::Debug, marker::PhantomData};

use num::{Float, FromPrimitive};

pub mod algorithms;
pub mod observers;
pub mod test_functions;

pub mod prelude {
    pub use crate::{Algorithm, Bound, Function, Minimizer, Observer, Status};
}

#[macro_export]
/// Convenience macro for converting raw numeric values to a generic.
macro_rules! convert {
    ($value:expr, $type:ty) => {{
        #[allow(clippy::unwrap_used)]
        <$type as num::NumCast>::from($value).unwrap()
    }};
}
#[derive(Default, Copy, Clone)]
pub enum Bound<T> {
    #[default]
    NoBound,
    LowerBound(T),
    UpperBound(T),
    LowerAndUpperBound(T, T),
}
impl<T> From<(T, T)> for Bound<T>
where
    T: Float,
{
    fn from(value: (T, T)) -> Self {
        match (value.0.is_finite(), value.1.is_finite()) {
            (true, true) => Self::LowerAndUpperBound(value.0, value.1),
            (true, false) => Self::LowerBound(value.0),
            (false, true) => Self::UpperBound(value.0),
            (false, false) => Self::NoBound,
        }
    }
}
impl<T> Bound<T>
where
    T: Float,
{
    fn to_bounded(values: &[T], bounds: &[Bound<T>]) -> Vec<T> {
        values
            .iter()
            .zip(bounds)
            .map(|(val, bound)| bound._to_bounded(*val))
            .collect()
    }
    fn _to_bounded(&self, val: T) -> T {
        match *self {
            Bound::LowerBound(lb) => lb - T::one() + T::sqrt(T::powi(val, 2) + T::one()),
            Bound::UpperBound(ub) => ub + T::one() - T::sqrt(T::powi(val, 2) + T::one()),
            Bound::LowerAndUpperBound(lb, ub) => {
                lb + (T::sin(val) + T::one()) * (ub - lb) / (T::one() + T::one())
            }
            Bound::NoBound => val,
        }
    }
    fn to_unbounded(values: &[T], bounds: &[Bound<T>]) -> Vec<T> {
        values
            .iter()
            .zip(bounds)
            .map(|(val, bound)| bound._to_unbounded(*val))
            .collect()
    }
    fn _to_unbounded(&self, val: T) -> T {
        match *self {
            Bound::LowerBound(lb) => T::sqrt(T::powi(val - lb + T::one(), 2) - T::one()),
            Bound::UpperBound(ub) => T::sqrt(T::powi(ub - val + T::one(), 2) - T::one()),
            Bound::LowerAndUpperBound(lb, ub) => {
                T::asin((T::one() + T::one()) * (val - lb) / (ub - lb) - T::one())
            }
            Bound::NoBound => val,
        }
    }
}

pub trait Function<T, U, E>
where
    T: Float + FromPrimitive,
{
    fn evaluate(&self, x: &[T], user_data: &mut U) -> Result<T, E>;
    fn gradient(&self, x: &[T], grad: &mut [T], user_data: &mut U) -> Result<(), E> {
        let n = x.len();
        let mut x = x.to_vec();
        let eps = T::cbrt(T::epsilon());
        let two_eps = eps * (T::one() + T::one());
        for i in 0..n {
            let xi = x[i];
            x[i] = xi + eps;
            let fm = self.evaluate(&x, user_data)?;
            x[i] = xi - eps;
            let fp = self.evaluate(&x, user_data)?;
            grad[i] = (fp - fm) / (two_eps);
            x[i] = xi;
        }
        Ok(())
    }
}

#[derive(Debug, Default, Clone)]
pub struct Status<T> {
    pub message: String,
    pub x: Vec<T>,
    pub fx: T,
    pub n_evals: usize,
    pub converged: bool,
}

impl<T> Status<T> {
    pub fn update_message(&mut self, message: &str) {
        self.message = message.to_string();
    }
    pub fn update_position(&mut self, pos: (Vec<T>, T)) {
        self.x = pos.0;
        self.fx = pos.1;
    }
    pub fn set_converged(&mut self) {
        self.converged = true;
    }
    pub fn increment_n_evals(&mut self) {
        self.n_evals += 1;
    }
}

pub trait Algorithm<T, U, E> {
    fn initialize(
        &mut self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E>;
    fn step(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> Result<(), E>;
    fn check_for_termination(
        &mut self,
        func: &dyn Function<T, U, E>,
        bounds: Option<&Vec<Bound<T>>>,
        user_data: &mut U,
    ) -> bool;
    fn get_status(&self) -> &Status<T>;
}

pub trait Observer<T, U> {
    fn callback(&mut self, step: usize, status: &Status<T>, user_data: &mut U);
}

pub struct Minimizer<T, U, E, A>
where
    A: Algorithm<T, U, E>,
{
    algorithm: A,
    bounds: Option<Vec<Bound<T>>>,
    max_steps: usize,
    observers: Vec<Box<dyn Observer<T, U>>>,
    dimension: usize,
    _phantom: PhantomData<E>,
}

impl<T, U, E, A: Algorithm<T, U, E>> Minimizer<T, U, E, A>
where
    T: Float + FromPrimitive,
{
    const DEFAULT_MAX_STEPS: usize = 4000;
    pub fn new(algorithm: A, dimension: usize) -> Self {
        Self {
            algorithm,
            bounds: None,
            max_steps: Self::DEFAULT_MAX_STEPS,
            observers: Vec::default(),
            dimension,
            _phantom: PhantomData,
        }
    }
    pub fn with_algorithm(mut self, algorithm: A) -> Self {
        self.algorithm = algorithm;
        self
    }
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
    pub fn with_observers(mut self, observers: Vec<Box<dyn Observer<T, U>>>) -> Self {
        self.observers = observers;
        self
    }
    pub fn with_observer<O>(mut self, observer: O) -> Self
    where
        O: Observer<T, U> + 'static,
    {
        self.observers.push(Box::new(observer));
        self
    }
    pub fn with_bounds(mut self, bounds: Option<Vec<(T, T)>>) -> Self {
        if let Some(bounds) = bounds {
            assert!(bounds.len() == self.dimension);
            self.bounds = Some(bounds.into_iter().map(Bound::from).collect());
        } else {
            self.bounds = None
        }
        self
    }
    pub fn with_bound(mut self, index: usize, bound: (T, T)) -> Self {
        assert!(index < self.dimension);
        if let Some(bounds) = &mut self.bounds {
            bounds[index] = Bound::from(bound);
        } else {
            let mut bounds = vec![Bound::default(); self.dimension];
            bounds[index] = Bound::from(bound);
            self.bounds = Some(bounds);
        }
        self
    }
    pub fn minimize(
        &mut self,
        func: &dyn Function<T, U, E>,
        x0: &[T],
        user_data: &mut U,
    ) -> Result<Status<T>, E> {
        assert!(x0.len() == self.dimension);
        self.algorithm
            .initialize(func, x0, self.bounds.as_ref(), user_data)?;
        let mut current_step = 0;
        while current_step <= self.max_steps
            && !self
                .algorithm
                .check_for_termination(func, self.bounds.as_ref(), user_data)
        {
            self.algorithm.step(func, self.bounds.as_ref(), user_data)?;
            current_step += 1;
            if !self.observers.is_empty() {
                let status = self.algorithm.get_status();
                for observer in self.observers.iter_mut() {
                    observer.callback(current_step, status, user_data);
                }
            }
        }
        let mut status = self.algorithm.get_status().clone();
        if current_step == self.max_steps && !status.converged {
            status.update_message("MAX EVALS");
        }
        Ok(status)
    }
}
