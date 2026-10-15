//! Scalar- and linear-algebra-generic limited-memory BFGS with native box projection.

use crate::algorithms::gradient::GradientStatus;
use crate::algorithms::line_search::StrongWolfeLineSearch;
use crate::core::{
    Callbacks, LinearAlgebra, LinearSolve, LuSolver, Matrix, MinimizationSummary, NalgebraProvider,
    PseudoInverse, RealScalar, Vector,
};
use crate::error::{GaneshError, GaneshResult};
use crate::traits::{
    Algorithm, CheckpointableAlgorithm, Gradient, LineSearch, LineSearchOutput, LineSearchStart,
    ScalarBound, Status, SupportsParameterNames, Terminator, Transform, TransformedProblem,
};
use std::{marker::PhantomData, ops::ControlFlow, sync::Arc};

macro_rules! lbfgsb_terminator {
    ($name:ident, $default:expr, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone)]
        pub struct $name<T: RealScalar = f64> {
            eps_abs: T,
        }

        impl<T: RealScalar> Default for $name<T> {
            fn default() -> Self {
                Self { eps_abs: $default }
            }
        }

        impl<T: RealScalar> $name<T> {
            /// Construct the terminator with a validated absolute tolerance.
            pub fn new(eps_abs: T) -> GaneshResult<Self> {
                if eps_abs <= T::zero() {
                    return Err(GaneshError::ConfigError(
                        "eps_abs must be greater than 0".to_string(),
                    ));
                }
                Ok(Self { eps_abs })
            }
        }
    };
}

impl<T: RealScalar, L, B: LinearAlgebra<T>> SupportsParameterNames for LBFGSBConfig<T, L, B> {
    fn get_parameter_names_mut(&mut self) -> &mut Option<Vec<String>> {
        &mut self.parameter_names
    }
}

lbfgsb_terminator!(
    LBFGSBFTerminator,
    T::epsilon().sqrt(),
    "Terminates [`LBFGSB`] when the objective change is sufficiently small."
);
lbfgsb_terminator!(
    LBFGSBGTerminator,
    T::epsilon().cbrt(),
    "Terminates [`LBFGSB`] when the gradient norm is sufficiently small."
);
lbfgsb_terminator!(
    LBFGSBInfNormGTerminator,
    T::epsilon().cbrt(),
    "Terminates [`LBFGSB`] when the projected-gradient infinity norm is sufficiently small."
);

/// Controls final Hessian/error estimation for linear-algebra-generic L-BFGS-B.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LBFGSBErrorMode {
    /// Evaluate the problem Hessian after optimization.
    #[default]
    ExactHessian,
    /// Skip final Hessian and covariance estimation.
    Skip,
}

/// Configuration for linear-algebra-generic L-BFGS-B.
pub struct LBFGSBConfig<
    T: RealScalar = f64,
    L = StrongWolfeLineSearch<T, NalgebraProvider>,
    B: LinearAlgebra<T> = NalgebraProvider,
> {
    /// Number of correction pairs retained by the inverse-Hessian approximation.
    history_size: usize,
    /// Maximum line-search step before bounds are considered.
    max_step: T,
    /// Final Hessian/error-estimation policy.
    error_mode: LBFGSBErrorMode,
    /// Optional native parameter bounds.
    bounds: Option<Vec<ScalarBound<T>>>,
    /// Optional user-facing parameter names copied into summaries.
    parameter_names: Option<Vec<String>>,
    /// Optional coordinate transform.
    transform: Option<Box<dyn Transform<T, B>>>,
    /// Strong-Wolfe line search.
    line_search: L,
}

impl<T, L, B> Default for LBFGSBConfig<T, L, B>
where
    T: RealScalar,
    L: Default,
    B: LinearAlgebra<T>,
{
    fn default() -> Self {
        Self {
            history_size: 10,
            max_step: T::literal(1e8),
            error_mode: LBFGSBErrorMode::default(),
            bounds: None,
            parameter_names: None,
            transform: None,
            line_search: L::default(),
        }
    }
}

impl<T, L, B> LBFGSBConfig<T, L, B>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
{
    /// Create a default configuration.
    pub fn new() -> Self
    where
        L: Default,
    {
        Self::default()
    }

    /// Set the number of retained L-BFGS correction pairs.
    pub fn with_memory_limit(mut self, limit: usize) -> GaneshResult<Self> {
        if limit < 1 {
            return Err(GaneshError::ConfigError(
                "Memory limit must be at least 1".to_string(),
            ));
        }
        self.history_size = limit;
        Ok(self)
    }

    /// Set the maximum line-search step before bounds are considered.
    ///
    /// # Errors
    /// Returns a configuration error unless the step is finite and positive.
    pub fn with_max_step(mut self, max_step: T) -> GaneshResult<Self> {
        if !max_step.is_finite() || max_step <= T::zero() {
            return Err(GaneshError::ConfigError(
                "Maximum step must be finite and greater than 0".to_string(),
            ));
        }
        self.max_step = max_step;
        Ok(self)
    }

    /// Replace the line-search implementation, changing the configuration's line-search type.
    pub fn with_line_search<N>(self, line_search: N) -> LBFGSBConfig<T, N, B> {
        LBFGSBConfig {
            history_size: self.history_size,
            max_step: self.max_step,
            error_mode: self.error_mode,
            bounds: self.bounds,
            parameter_names: self.parameter_names,
            transform: self.transform,
            line_search,
        }
    }

    /// Configure native lower/upper bounds.
    ///
    /// # Errors
    /// Returns a configuration error for invalid endpoints.
    pub fn with_bounds<I>(mut self, bounds: I) -> GaneshResult<Self>
    where
        I: IntoIterator<Item = (T, T)>,
    {
        let bounds = bounds
            .into_iter()
            .map(|(lower, upper)| ScalarBound::new(lower, upper))
            .collect::<GaneshResult<Vec<_>>>()?;
        self.bounds = Some(bounds);
        Ok(self)
    }

    /// Apply a differentiable coordinate transform around the problem.
    ///
    /// Native bounds are constraints in the optimizer's coordinate space and are not rewritten by
    /// this transform. Pass [`crate::traits::Bounds`] explicitly here when bounds should instead be
    /// implemented as a parameter transform.
    pub fn with_transform<X>(mut self, transform: X) -> GaneshResult<Self>
    where
        X: Transform<T, B> + 'static,
    {
        self.transform = Some(Box::new(transform));
        Ok(self)
    }

    /// Select final Hessian/error estimation.
    pub const fn with_error_mode(mut self, error_mode: LBFGSBErrorMode) -> Self {
        self.error_mode = error_mode;
        self
    }
}

/// Step-boundary checkpoint for linear-algebra-generic L-BFGS-B.
#[derive(Clone, Debug)]
pub struct LBFGSBCheckpoint<T: RealScalar, B: LinearAlgebra<T>> {
    /// Saved parameter vector.
    pub x: Vector<T, B>,
    /// Saved objective value.
    pub fx: T,
    /// Previous objective value used by the objective-change terminator.
    pub f_previous: T,
    /// Saved gradient.
    pub gradient: Vector<T, B>,
    /// Saved displacement history.
    pub s_history: Vec<Vector<T, B>>,
    /// Saved gradient-difference history.
    pub y_history: Vec<Vector<T, B>>,
    /// Saved compact-Hessian scaling.
    pub theta: T,
    /// Saved status and evaluation counts.
    pub status: GradientStatus<T, B>,
    /// Step index at which processing resumes.
    pub next_step: usize,
}

/// Scalar- and linear-algebra-generic limited-memory BFGS optimizer with native box projection.
#[derive(Clone, Debug)]
pub struct LBFGSB<
    T: RealScalar = f64,
    B: LinearAlgebra<T> = NalgebraProvider,
    L = StrongWolfeLineSearch<T, B>,
> {
    x: Vector<T, B>,
    fx: T,
    f_previous: T,
    gradient: Vector<T, B>,
    s_history: Vec<Vector<T, B>>,
    y_history: Vec<Vector<T, B>>,
    theta: T,
    w_matrix: Matrix<T, B>,
    m_system: Option<Arc<dyn LuSolver<T, B>>>,
    line_search: L,
    _provider: PhantomData<B>,
}

impl<T, B, L> Default for LBFGSB<T, B, L>
where
    T: RealScalar,
    B: LinearAlgebra<T>,
    L: Default,
{
    fn default() -> Self {
        Self {
            x: Vector::zeros(0),
            fx: T::infinity(),
            f_previous: T::infinity(),
            gradient: Vector::zeros(0),
            s_history: Vec::new(),
            y_history: Vec::new(),
            theta: T::one(),
            w_matrix: Matrix::zeros(0, 0),
            m_system: None,
            line_search: L::default(),
            _provider: PhantomData,
        }
    }
}

impl<T, B, L> LBFGSB<T, B, L>
where
    T: RealScalar,
    B: LinearAlgebra<T> + LinearSolve<T>,
{
    fn bound_limits(bound: ScalarBound<T>) -> (T, T) {
        match bound {
            ScalarBound::Unbounded => (-T::infinity(), T::infinity()),
            ScalarBound::Lower(lower) => (lower, T::infinity()),
            ScalarBound::Upper(upper) => (-T::infinity(), upper),
            ScalarBound::Both(lower, upper) => (lower, upper),
        }
    }

    fn m_dot_vec(&self, value: &Vector<T, B>) -> Vector<T, B> {
        self.m_system
            .as_ref()
            .and_then(|system| system.solve(value.as_storage()))
            .map(Vector::from_storage)
            .unwrap_or_else(|| Vector::zeros(value.len()))
    }

    fn is_unconstrained(bounds: Option<&[ScalarBound<T>]>) -> bool {
        bounds.is_none_or(|bounds| {
            bounds
                .iter()
                .all(|bound| matches!(bound, ScalarBound::Unbounded))
        })
    }

    fn unconstrained_direction(&self) -> Vector<T, B> {
        if self.s_history.is_empty() {
            return self.gradient.scale(-T::one() / self.theta);
        }

        // Apply the inverse Hessian to -g using the L-BFGS two-loop recursion.
        let mut direction = self.gradient.neg();
        let mut coefficients = Vec::with_capacity(self.s_history.len());
        for (s, y) in self.s_history.iter().zip(&self.y_history).rev() {
            let curvature = s.dot(y);
            let alpha = s.dot(&direction) / curvature;
            direction = direction.add_scaled(y, -alpha);
            coefficients.push((curvature, alpha));
        }
        direction = direction.scale(T::one() / self.theta);
        for ((s, y), (curvature, alpha)) in self
            .s_history
            .iter()
            .zip(&self.y_history)
            .zip(coefficients.into_iter().rev())
        {
            let beta = y.dot(&direction) / curvature;
            direction = direction.add_scaled(s, alpha - beta);
        }
        direction
    }

    fn m_dot_mat(&self, value: &Matrix<T, B>) -> Matrix<T, B> {
        self.m_system
            .as_ref()
            .and_then(|system| system.solve_matrix(value.as_storage()))
            .map(Matrix::from_storage)
            .unwrap_or_else(|| Matrix::zeros(value.rows(), value.cols()))
    }

    fn update_compact_matrices(&mut self) {
        let m = self.s_history.len();
        let n = self.x.len();
        self.w_matrix = Matrix::zeros(n, 2 * m);
        let mut s: Matrix<T, B> = Matrix::zeros(n, m);
        let mut y: Matrix<T, B> = Matrix::zeros(n, m);
        for (column, (s_column, y_column)) in self.s_history.iter().zip(&self.y_history).enumerate()
        {
            for row in 0..n {
                let s_value = s_column.get(row);
                let y_value = y_column.get(row);
                s.set(row, column, s_value);
                y.set(row, column, y_value);
                self.w_matrix.set(row, column, y_value);
                self.w_matrix.set(row, m + column, self.theta * s_value);
            }
        }
        let sty = s.transpose().mul_mat(&y);
        let sts = s.transpose().mul_mat(&s).scale(self.theta);
        let mut system: Matrix<T, B> = Matrix::zeros(2 * m, 2 * m);
        for row in 0..m {
            system.set(row, row, -sty.get(row, row));
            for column in 0..m {
                if row > column {
                    system.set(m + row, column, sty.get(row, column));
                    system.set(column, m + row, sty.get(row, column));
                }
                system.set(m + row, m + column, sts.get(row, column));
            }
        }
        self.m_system = Some(B::lu_factorize(system.as_storage()));
    }

    fn generalized_cauchy_point(
        &self,
        bounds: Option<&[ScalarBound<T>]>,
    ) -> (Vector<T, B>, Vector<T, B>, Vec<usize>) {
        let n = self.x.len();
        let mut breakpoints = vec![T::infinity(); n];
        let mut direction = self.gradient.neg();
        for index in 0..n {
            let (lower, upper) = bounds
                .map(|values| Self::bound_limits(values[index]))
                .unwrap_or_else(|| (-T::infinity(), T::infinity()));
            let gradient = self.gradient.get(index);
            breakpoints[index] = if gradient < T::zero() {
                (self.x.get(index) - upper) / gradient
            } else if gradient > T::zero() {
                (self.x.get(index) - lower) / gradient
            } else {
                T::infinity()
            };
            if breakpoints[index] < T::epsilon() {
                direction.set(index, T::zero());
            }
        }
        let mut xcp = self.x.clone();
        let mut order = (0..n)
            .filter(|&index| breakpoints[index] > T::zero())
            .collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            breakpoints[a]
                .partial_cmp(&breakpoints[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut p = self.w_matrix.transpose().mul_vec(&direction);
        let mut c = Vector::zeros(p.len());
        if order.is_empty() {
            return (xcp, c, order);
        }
        let mut t_old = T::zero();
        let mut cursor = 0;
        let mut breakpoint = order[cursor];
        let mut t_break = breakpoints[breakpoint];
        let mut delta_break = t_break;
        let mut df = -direction.dot(&direction);
        let mut ddf = -self.theta * df - p.dot(&self.m_dot_vec(&p));
        let mut delta_min = -df / ddf;
        while delta_min >= delta_break && cursor < order.len() {
            let (lower, upper) = bounds
                .map(|values| Self::bound_limits(values[breakpoint]))
                .unwrap_or_else(|| (-T::infinity(), T::infinity()));
            xcp.set(
                breakpoint,
                if direction.get(breakpoint) > T::zero() {
                    upper
                } else {
                    lower
                },
            );
            let z = xcp.get(breakpoint) - self.x.get(breakpoint);
            c = c.add_scaled(&p, delta_break);
            let g = self.gradient.get(breakpoint);
            let w_row = Vector::from_vec(
                (0..self.w_matrix.cols())
                    .map(|column| self.w_matrix.get(breakpoint, column))
                    .collect(),
            );
            df = df + delta_break * ddf + g * (self.theta * z + g - w_row.dot(&self.m_dot_vec(&c)));
            ddf = ddf
                + -g * (self.theta * g + T::literal(-2.0) * w_row.dot(&self.m_dot_vec(&p))
                    - g * w_row.dot(&self.m_dot_vec(&w_row)));
            p = p.add_scaled(&w_row, g);
            direction.set(breakpoint, T::zero());
            delta_min = -df / ddf;
            t_old = t_break;
            cursor += 1;
            if cursor < order.len() {
                breakpoint = order[cursor];
                t_break = breakpoints[breakpoint];
                delta_break = t_break - t_old;
            } else {
                t_break = T::infinity();
            }
        }
        if delta_min < T::zero() {
            delta_min = T::zero();
        }
        t_old = t_old + delta_min;
        for (index, &breakpoint_value) in breakpoints.iter().enumerate() {
            if breakpoint_value >= t_break {
                xcp.set(index, xcp.get(index) + t_old * direction.get(index));
            }
        }
        c = c.add_scaled(&p, delta_min);
        let free = (0..n)
            .filter(|&index| {
                let (lower, upper) = bounds
                    .map(|values| Self::bound_limits(values[index]))
                    .unwrap_or_else(|| (-T::infinity(), T::infinity()));
                xcp.get(index) > lower && xcp.get(index) < upper
            })
            .collect();
        (xcp, c, free)
    }

    fn subspace_minimum(
        &self,
        xcp: &Vector<T, B>,
        c: &Vector<T, B>,
        free: &[usize],
        bounds: Option<&[ScalarBound<T>]>,
    ) -> Vector<T, B> {
        if free.is_empty() {
            return xcp.clone();
        }
        let k = free.len();
        let two_m = self.w_matrix.cols();
        let r_full = self
            .gradient
            .add_scaled(&xcp.sub(&self.x), self.theta)
            .sub(&self.w_matrix.mul_vec(&self.m_dot_vec(c)));
        let r = Vector::from_vec(free.iter().map(|&index| r_full.get(index)).collect());
        let unconstrained = if two_m == 0 {
            r.scale(-T::one() / self.theta)
        } else {
            let mut wt_z = Matrix::zeros(two_m, k);
            for (column, &index) in free.iter().enumerate() {
                for row in 0..two_m {
                    wt_z.set(row, column, self.w_matrix.get(index, row));
                }
            }
            let correction = self
                .m_dot_mat(&wt_z.mul_mat(&wt_z.transpose()))
                .scale(T::one() / self.theta);
            let n_matrix = &Matrix::identity(two_m) - &correction;
            let rhs = self.m_dot_vec(&wt_z.mul_vec(&r));
            let v = n_matrix
                .lu_solve(&rhs)
                .unwrap_or_else(|| Vector::zeros(two_m));
            r.add(&wt_z.transpose().mul_vec(&v).scale(T::one() / self.theta))
                .scale(-T::one() / self.theta)
        };
        let mut alpha = T::one();
        for (position, &index) in free.iter().enumerate() {
            let (lower, upper) = bounds
                .map(|values| Self::bound_limits(values[index]))
                .unwrap_or_else(|| (-T::infinity(), T::infinity()));
            let d = unconstrained.get(position);
            let candidate = if d > T::zero() {
                (upper - xcp.get(index)) / d
            } else if d < T::zero() {
                (lower - xcp.get(index)) / d
            } else {
                T::one()
            };
            if candidate < alpha {
                alpha = candidate;
            }
        }
        let mut result = xcp.clone();
        for (position, &index) in free.iter().enumerate() {
            result.set(
                index,
                result.get(index) + alpha * unconstrained.get(position),
            );
        }
        result
    }

    fn bound_constrained_direction(&self, bounds: Option<&[ScalarBound<T>]>) -> Vector<T, B> {
        let (xcp, c, free) = self.generalized_cauchy_point(bounds);
        self.subspace_minimum(&xcp, &c, &free, bounds).sub(&self.x)
    }
    fn clip(mut x: Vector<T, B>, bounds: Option<&[ScalarBound<T>]>) -> Vector<T, B> {
        if let Some(bounds) = bounds {
            for (index, bound) in bounds.iter().copied().enumerate() {
                let value = match bound {
                    ScalarBound::Unbounded => x.get(index),
                    ScalarBound::Lower(lower) => {
                        if x.get(index) < lower {
                            lower
                        } else {
                            x.get(index)
                        }
                    }
                    ScalarBound::Upper(upper) => {
                        if x.get(index) > upper {
                            upper
                        } else {
                            x.get(index)
                        }
                    }
                    ScalarBound::Both(lower, upper) => {
                        if x.get(index) < lower {
                            lower
                        } else if x.get(index) > upper {
                            upper
                        } else {
                            x.get(index)
                        }
                    }
                };
                x.set(index, value);
            }
        }
        x
    }

    fn projected_gradient(&self, bounds: Option<&[ScalarBound<T>]>) -> Vector<T, B> {
        let mut projected = self.gradient.clone();
        if let Some(bounds) = bounds {
            for (index, bound) in bounds.iter().copied().enumerate() {
                let at_lower = match bound {
                    ScalarBound::Lower(lower) | ScalarBound::Both(lower, _) => {
                        self.x.get(index) <= lower + T::epsilon().sqrt()
                    }
                    _ => false,
                };
                let at_upper = match bound {
                    ScalarBound::Upper(upper) | ScalarBound::Both(_, upper) => {
                        self.x.get(index) >= upper - T::epsilon().sqrt()
                    }
                    _ => false,
                };
                if (at_lower && projected.get(index) > T::zero())
                    || (at_upper && projected.get(index) < T::zero())
                {
                    projected.set(index, T::zero());
                }
            }
        }
        projected
    }

    fn maximum_step(
        &self,
        direction: &Vector<T, B>,
        bounds: Option<&[ScalarBound<T>]>,
    ) -> Option<T> {
        let bounds = bounds?;
        let mut maximum = T::infinity();
        for (index, bound) in bounds.iter().copied().enumerate() {
            let direction_i = direction.get(index);
            let candidate = match bound {
                ScalarBound::Lower(lower) if direction_i < T::zero() => {
                    Some((lower - self.x.get(index)) / direction_i)
                }
                ScalarBound::Upper(upper) if direction_i > T::zero() => {
                    Some((upper - self.x.get(index)) / direction_i)
                }
                ScalarBound::Both(lower, _) if direction_i < T::zero() => {
                    Some((lower - self.x.get(index)) / direction_i)
                }
                ScalarBound::Both(_, upper) if direction_i > T::zero() => {
                    Some((upper - self.x.get(index)) / direction_i)
                }
                _ => None,
            };
            if let Some(candidate) = candidate {
                if candidate < maximum {
                    maximum = candidate;
                }
            }
        }
        if maximum.is_finite() {
            Some(maximum)
        } else {
            None
        }
    }
}

impl<T, B, L, P, U, E>
    Terminator<LBFGSB<T, B, L>, P, GradientStatus<T, B>, U, E, LBFGSBConfig<T, L, B>>
    for LBFGSBFTerminator<T>
where
    T: RealScalar,
    B: LinearAlgebra<T> + LinearSolve<T> + PseudoInverse<T>,
    P: Gradient<T, B, U, E>,
    L: for<'a> LineSearch<T, B, TransformedProblem<'a, P, T, B>, U, E>
        + Clone
        + Default
        + Send
        + Sync,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut LBFGSB<T, B, L>,
        _problem: &P,
        status: &mut GradientStatus<T, B>,
        _args: &U,
        _config: &LBFGSBConfig<T, L, B>,
    ) -> ControlFlow<()> {
        if (algorithm.f_previous - algorithm.fx).abs() < self.eps_abs {
            status
                .set_message()
                .succeed_with_message("F_EVAL CONVERGED");
            ControlFlow::Break(())
        } else {
            algorithm.f_previous = algorithm.fx;
            ControlFlow::Continue(())
        }
    }
}

impl<T, B, L, P, U, E>
    Terminator<LBFGSB<T, B, L>, P, GradientStatus<T, B>, U, E, LBFGSBConfig<T, L, B>>
    for LBFGSBGTerminator<T>
where
    T: RealScalar,
    B: LinearAlgebra<T> + LinearSolve<T> + PseudoInverse<T>,
    P: Gradient<T, B, U, E>,
    L: for<'a> LineSearch<T, B, TransformedProblem<'a, P, T, B>, U, E>
        + Clone
        + Default
        + Send
        + Sync,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut LBFGSB<T, B, L>,
        _problem: &P,
        status: &mut GradientStatus<T, B>,
        _args: &U,
        _config: &LBFGSBConfig<T, L, B>,
    ) -> ControlFlow<()> {
        if algorithm.gradient.norm() < self.eps_abs {
            status
                .set_message()
                .succeed_with_message("GRADIENT CONVERGED");
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

impl<T, B, L, P, U, E>
    Terminator<LBFGSB<T, B, L>, P, GradientStatus<T, B>, U, E, LBFGSBConfig<T, L, B>>
    for LBFGSBInfNormGTerminator<T>
where
    T: RealScalar,
    B: LinearAlgebra<T> + LinearSolve<T> + PseudoInverse<T>,
    P: Gradient<T, B, U, E>,
    L: for<'a> LineSearch<T, B, TransformedProblem<'a, P, T, B>, U, E>
        + Clone
        + Default
        + Send
        + Sync,
{
    fn check_for_termination(
        &mut self,
        _current_step: usize,
        algorithm: &mut LBFGSB<T, B, L>,
        _problem: &P,
        status: &mut GradientStatus<T, B>,
        _args: &U,
        config: &LBFGSBConfig<T, L, B>,
    ) -> ControlFlow<()> {
        let projected = algorithm.projected_gradient(config.bounds.as_deref());
        let mut norm = T::zero();
        for index in 0..projected.len() {
            let value = projected.get(index).abs();
            if value > norm {
                norm = value;
            }
        }
        if norm < self.eps_abs {
            status
                .set_message()
                .succeed_with_message("PROJECTED GRADIENT WITHIN TOLERANCE");
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

impl<T, B, L, P, U, E> Algorithm<P, GradientStatus<T, B>, U, E> for LBFGSB<T, B, L>
where
    T: RealScalar,
    B: LinearAlgebra<T> + LinearSolve<T> + PseudoInverse<T>,
    P: Gradient<T, B, U, E>,
    L: for<'a> LineSearch<T, B, TransformedProblem<'a, P, T, B>, U, E>
        + Clone
        + Default
        + Send
        + Sync,
{
    type Summary = MinimizationSummary<T, B>;
    type Config = LBFGSBConfig<T, L, B>;
    type Init = Vector<T, B>;

    fn initialize(
        &mut self,
        problem: &P,
        status: &mut GradientStatus<T, B>,
        args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<(), E> {
        if let Some(bounds) = &config.bounds {
            debug_assert_eq!(bounds.len(), init.len());
        }
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        self.x = Self::clip(transformed.to_internal(init), config.bounds.as_deref());
        (self.fx, self.gradient) = transformed.evaluate_with_gradient(&self.x, args)?;
        self.f_previous = T::infinity();
        self.s_history.clear();
        self.y_history.clear();
        self.theta = T::one();
        self.w_matrix = Matrix::zeros(self.x.len(), 0);
        self.m_system = None;
        self.line_search = config.line_search.clone();
        status.evals.record_fg();
        status.initialize(init.clone(), self.fx);
        Ok(())
    }

    fn step(
        &mut self,
        _current_step: usize,
        problem: &P,
        status: &mut GradientStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        let transformed = TransformedProblem::new(problem, config.transform.as_deref());
        let unconstrained = Self::is_unconstrained(config.bounds.as_deref());
        let direction = if unconstrained {
            self.unconstrained_direction()
        } else {
            // History can originate from an unconstrained checkpoint or a previous configuration.
            if self.m_system.is_none() && !self.s_history.is_empty() {
                self.update_compact_matrices();
            }
            self.bound_constrained_direction(config.bounds.as_deref())
        };
        let maximum_step = self
            .maximum_step(&direction, config.bounds.as_deref())
            .map_or(config.max_step, |bounded| {
                if bounded < config.max_step {
                    bounded
                } else {
                    config.max_step
                }
            });
        if let Ok(LineSearchOutput {
            alpha,
            fx,
            gradient,
        }) = self.line_search.search_from(
            LineSearchStart {
                x: &self.x,
                fx: self.fx,
                gradient: &self.gradient,
            },
            &direction,
            Some(maximum_step),
            &transformed,
            args,
            &mut status.evals,
        )? {
            let next_x = Self::clip(
                self.x.add_scaled(&direction, alpha),
                config.bounds.as_deref(),
            );
            let s = next_x.sub(&self.x);
            let y = gradient.sub(&self.gradient);
            let sy = s.dot(&y);
            let yy = y.dot(&y);
            if sy > T::epsilon() * yy {
                if self.s_history.len() == config.history_size.max(1) {
                    self.s_history.remove(0);
                    self.y_history.remove(0);
                }
                self.s_history.push(s);
                self.y_history.push(y);
                self.theta = yy / sy;
                if unconstrained {
                    self.w_matrix = Matrix::zeros(self.x.len(), 0);
                    self.m_system = None;
                } else {
                    self.update_compact_matrices();
                }
            }
            self.x = next_x;
            self.fx = fx;
            self.gradient = gradient;
            status.set_position(transformed.to_external(&self.x), self.fx);
        } else {
            self.s_history.clear();
            self.y_history.clear();
            self.theta = T::one();
            self.w_matrix = Matrix::zeros(self.x.len(), 0);
            self.m_system = None;
        }
        Ok(())
    }

    fn postprocessing(
        &mut self,
        problem: &P,
        status: &mut GradientStatus<T, B>,
        args: &U,
        config: &Self::Config,
    ) -> Result<(), E> {
        if config.error_mode == LBFGSBErrorMode::ExactHessian {
            let transformed = TransformedProblem::new(problem, config.transform.as_deref());
            let hessian = transformed.hessian(&self.x, args)?;
            status.evals.record_h();
            status.set_hess(hessian);
        }
        Ok(())
    }

    fn summarize(
        &self,
        _current_step: usize,
        _problem: &P,
        status: &GradientStatus<T, B>,
        _args: &U,
        init: &Self::Init,
        config: &Self::Config,
    ) -> Result<Self::Summary, E> {
        let dimension = status.x.len();
        Ok(MinimizationSummary {
            bounds: config.bounds.clone(),
            parameter_names: config.parameter_names.clone(),
            message: status.message.clone(),
            x0: init.clone(),
            x: status.x.clone(),
            std: status
                .err
                .clone()
                .unwrap_or_else(|| crate::core::summary::unknown_uncertainties(dimension)),
            fx: status.fx,
            evals: status.evals,
            covariance: status
                .cov
                .clone()
                .unwrap_or_else(|| Matrix::identity(dimension)),
        })
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn default_callbacks() -> Callbacks<Self, P, GradientStatus<T, B>, U, E, Self::Config> {
        Callbacks::empty()
            .with_terminator(LBFGSBFTerminator::default())
            .with_terminator(LBFGSBGTerminator::default())
            .with_terminator(LBFGSBInfNormGTerminator::default())
    }
}

impl<T, B, L, P, U, E> CheckpointableAlgorithm<P, GradientStatus<T, B>, U, E> for LBFGSB<T, B, L>
where
    T: RealScalar,
    B: LinearAlgebra<T> + LinearSolve<T> + PseudoInverse<T>,
    P: Gradient<T, B, U, E>,
    L: for<'a> LineSearch<T, B, TransformedProblem<'a, P, T, B>, U, E>
        + Clone
        + Default
        + Send
        + Sync,
{
    type Checkpoint = LBFGSBCheckpoint<T, B>;

    fn checkpoint(&self, status: &GradientStatus<T, B>, next_step: usize) -> Self::Checkpoint {
        LBFGSBCheckpoint {
            x: self.x.clone(),
            fx: self.fx,
            f_previous: self.f_previous,
            gradient: self.gradient.clone(),
            s_history: self.s_history.clone(),
            y_history: self.y_history.clone(),
            theta: self.theta,
            status: status.clone(),
            next_step,
        }
    }

    fn restore(
        &mut self,
        checkpoint: &Self::Checkpoint,
        config: &Self::Config,
    ) -> (GradientStatus<T, B>, usize) {
        self.x = checkpoint.x.clone();
        self.fx = checkpoint.fx;
        self.f_previous = checkpoint.f_previous;
        self.gradient = checkpoint.gradient.clone();
        self.s_history = checkpoint.s_history.clone();
        self.y_history = checkpoint.y_history.clone();
        self.theta = checkpoint.theta;
        if self.s_history.is_empty() || Self::is_unconstrained(config.bounds.as_deref()) {
            self.w_matrix = Matrix::zeros(self.x.len(), 0);
            self.m_system = None;
        } else {
            self.update_compact_matrices();
        }
        self.line_search = config.line_search.clone();
        (checkpoint.status.clone(), checkpoint.next_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CostFunction;
    use crate::{Bounds, ScaleTransform};
    use std::convert::Infallible;

    struct ShiftedQuadratic;

    impl<T, B> CostFunction<T, B> for ShiftedQuadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn evaluate(&self, x: &Vector<T, B>, _: &()) -> Result<T, Infallible> {
            Ok((x.get(0) - T::literal(3.0)).powi(2) + x.get(1).powi(2))
        }
    }

    impl<T, B> Gradient<T, B> for ShiftedQuadratic
    where
        T: RealScalar,
        B: LinearAlgebra<T>,
    {
        fn gradient(&self, x: &Vector<T, B>, _: &()) -> Result<Vector<T, B>, Infallible> {
            Ok(Vector::from_vec(vec![
                T::literal(2.0) * (x.get(0) - T::literal(3.0)),
                T::literal(2.0) * x.get(1),
            ]))
        }

        fn hessian(&self, _x: &Vector<T, B>, _: &()) -> Result<Matrix<T, B>, Infallible> {
            Ok(Matrix::identity(2).scale(T::literal(2.0)))
        }
    }

    #[test]
    fn lbfgsb_respects_native_f32_bounds() {
        let config = LBFGSBConfig::<f32>::default()
            .with_bounds([(0.0_f32, 2.0), (-1.0, 1.0)])
            .unwrap();
        let mut algorithm = LBFGSB::<f32>::default();
        let result = algorithm
            .process(
                &ShiftedQuadratic,
                &(),
                Vector::from_vec(vec![0.5, 0.5]),
                config,
                LBFGSB::<f32>::default_callbacks(),
            )
            .unwrap();
        assert!((result.x.get(0) - 2.0).abs() < 1e-4);
        assert_eq!(result.bounds.as_ref().map(Vec::len), Some(2));
        assert!(result.to_string().contains("Bounds"));
        assert!(
            result.x.get(1).abs() < 5e-4,
            "x={:?}, fx={}",
            result.x,
            result.fx
        );
    }

    #[test]
    fn lbfgsb_does_not_rewrite_native_bounds_for_transform() {
        let transform =
            ScaleTransform::<f64, NalgebraProvider>::from_parameter_scales([2.0, 4.0]).unwrap();
        let config = LBFGSBConfig::<f64>::default()
            .with_bounds([(0.0, 2.0), (-1.0, 1.0)])
            .unwrap()
            .with_transform(transform)
            .unwrap();
        let result = LBFGSB::<f64>::default()
            .process(
                &ShiftedQuadratic,
                &(),
                Vector::from_vec(vec![0.5, 0.5]),
                config,
                LBFGSB::<f64>::default_callbacks(),
            )
            .unwrap();

        assert!(
            (result.x.get(0) - 3.0).abs() < 1e-6,
            "x={:?}, fx={}",
            result.x,
            result.fx
        );
        assert!(result.x.get(1).abs() < 1e-6);
        assert_eq!(result.x0.to_vec(), vec![0.5, 0.5]);
        assert_eq!(result.bounds.as_ref().map(Vec::len), Some(2));
    }

    #[test]
    fn lbfgsb_accepts_explicit_bounds_transform() {
        let transform = Bounds::<f64, NalgebraProvider>::new([(0.0, 2.0), (-1.0, 1.0)]).unwrap();
        let config = LBFGSBConfig::<f64>::default()
            .with_transform(transform)
            .unwrap();

        assert!(config.bounds.is_none());
        assert_eq!(
            config
                .transform
                .as_deref()
                .and_then(Transform::parameter_bounds)
                .map(<[_]>::len),
            Some(2)
        );
    }

    #[test]
    fn lbfgsb_skip_error_mode_avoids_hessian_evaluation() {
        let config = LBFGSBConfig::<f64>::default().with_error_mode(LBFGSBErrorMode::Skip);
        let result = LBFGSB::<f64>::default()
            .process(
                &ShiftedQuadratic,
                &(),
                Vector::from_vec(vec![0.5, 0.5]),
                config,
                LBFGSB::<f64>::default_callbacks(),
            )
            .unwrap();
        assert_eq!(result.evals.h(), 0);
        assert!(result.std.to_vec().iter().all(|value| value.is_nan()));
    }

    #[test]
    fn lbfgsb_checkpoint_restores_generic_state() {
        let config = LBFGSBConfig::<f64>::default();
        let init = Vector::from_vec(vec![0.5, 0.5]);
        let mut algorithm = LBFGSB::<f64>::default();
        let mut status = GradientStatus::default();
        algorithm
            .initialize(&ShiftedQuadratic, &mut status, &(), &init, &config)
            .unwrap();
        algorithm
            .step(0, &ShiftedQuadratic, &mut status, &(), &config)
            .unwrap();
        type Checkpointable = LBFGSB<f64>;
        let checkpoint = <Checkpointable as CheckpointableAlgorithm<
            ShiftedQuadratic,
            GradientStatus<f64>,
            (),
            Infallible,
        >>::checkpoint(&algorithm, &status, 1);

        let mut restored = LBFGSB::<f64>::default();
        let (restored_status, next_step) = <Checkpointable as CheckpointableAlgorithm<
            ShiftedQuadratic,
            GradientStatus<f64>,
            (),
            Infallible,
        >>::restore(&mut restored, &checkpoint, &config);
        assert_eq!(next_step, 1);
        assert_eq!(restored.x.to_vec(), algorithm.x.to_vec());
        assert_eq!(restored.gradient.to_vec(), algorithm.gradient.to_vec());
        assert_eq!(restored_status.evals, status.evals);
    }
}

#[cfg(test)]
mod unconstrained_tests {
    use super::*;
    use crate::test_functions::Rosenbrock;
    use crate::traits::CostFunction;
    use std::convert::Infallible;

    fn history_fixture(count: usize) -> LBFGSB<f64> {
        let mut algorithm = LBFGSB::<f64> {
            x: Vector::zeros(4),
            gradient: Vector::from_vec(vec![0.7, -1.3, 0.2, 2.0]),
            theta: 2.5,
            ..LBFGSB::default()
        };
        for pair in 0..count {
            let s: Vector = Vector::from_vec(
                (0..4)
                    .map(|i| {
                        ((pair + 1) as f64 * (i + 2) as f64 * 0.43).sin() + 0.2 / (i + 1) as f64
                    })
                    .collect(),
            );
            let y = Vector::from_vec(
                (0..4)
                    .map(|i| s.get(i) * ((i + 1) as f64).powi(2))
                    .collect(),
            );
            algorithm.theta = y.dot(&y) / s.dot(&y);
            algorithm.s_history.push(s);
            algorithm.y_history.push(y);
        }
        algorithm
    }

    #[derive(Debug)]
    struct FreshLu(Matrix);

    impl LuSolver<f64, NalgebraProvider> for FreshLu {
        fn solve(&self, rhs: &nalgebra::DVector<f64>) -> Option<nalgebra::DVector<f64>> {
            self.0
                .lu_solve(&Vector::from_storage(rhs.clone()))
                .map(Vector::into_storage)
        }

        fn solve_matrix(&self, rhs: &nalgebra::DMatrix<f64>) -> Option<nalgebra::DMatrix<f64>> {
            self.0
                .lu_solve_matrix(&Matrix::from_storage(rhs.clone()))
                .map(Matrix::into_storage)
        }
    }

    fn fresh_compact_solver(algorithm: &LBFGSB) -> Arc<dyn LuSolver<f64, NalgebraProvider>> {
        // Construct the reference system directly from pairwise dots, independently
        // of the production matrix products. FreshLu factors it on every solve.
        let m = algorithm.s_history.len();
        let mut system = Matrix::zeros(2 * m, 2 * m);
        for i in 0..m {
            system.set(i, i, -algorithm.s_history[i].dot(&algorithm.y_history[i]));
            for j in 0..m {
                if i > j {
                    let sy = algorithm.s_history[i].dot(&algorithm.y_history[j]);
                    system.set(m + i, j, sy);
                    system.set(j, m + i, sy);
                }
                system.set(
                    m + i,
                    m + j,
                    algorithm.theta * algorithm.s_history[i].dot(&algorithm.s_history[j]),
                );
            }
        }
        Arc::new(FreshLu(system))
    }

    #[test]
    fn compact_solver_reuses_factors_for_vector_and_matrix_rhs() {
        for count in [1, 3, 7] {
            let mut algorithm = history_fixture(count);
            algorithm.update_compact_matrices();
            let cached = algorithm.m_system.clone().unwrap();
            let reference = fresh_compact_solver(&algorithm);
            for scale in [0.0, 1.0, -2.5] {
                let rhs =
                    Vector::from_vec((0..2 * count).map(|i| scale * (i as f64 + 0.3)).collect());
                let actual = algorithm.m_dot_vec(&rhs);
                let expected: Vector =
                    Vector::from_storage(reference.solve(rhs.as_storage()).unwrap());
                assert!(actual.sub(&expected).norm() < 1e-8 * (1.0 + expected.norm()));
                let rhs = Matrix::identity(2 * count).scale(scale);
                let actual = algorithm.m_dot_mat(&rhs);
                let expected: Matrix =
                    Matrix::from_storage(reference.solve_matrix(rhs.as_storage()).unwrap());
                for row in 0..2 * count {
                    for col in 0..2 * count {
                        assert!(
                            (actual.get(row, col) - expected.get(row, col)).abs()
                                < 1e-8 * (1.0 + expected.get(row, col).abs())
                        );
                    }
                }
                assert!(Arc::ptr_eq(algorithm.m_system.as_ref().unwrap(), &cached));
            }
            algorithm.theta *= 1.7;
            algorithm.update_compact_matrices();
            assert!(!Arc::ptr_eq(algorithm.m_system.as_ref().unwrap(), &cached));
            let rhs = Vector::from_vec(vec![1.0; 2 * count]);
            let expected: Vector = Vector::from_storage(
                fresh_compact_solver(&algorithm)
                    .solve(rhs.as_storage())
                    .unwrap(),
            );
            assert!(
                algorithm.m_dot_vec(&rhs).sub(&expected).norm() < 1e-8 * (1.0 + expected.norm())
            );
        }
    }

    #[test]
    fn singular_compact_solver_preserves_zero_fallback() {
        let mut algorithm = history_fixture(1);
        algorithm.y_history[0] = Vector::zeros(4);
        algorithm.update_compact_matrices();
        let cached = algorithm.m_system.clone().unwrap();
        for _ in 0..3 {
            assert_eq!(
                algorithm.m_dot_vec(&Vector::from_vec(vec![1.0; 2])),
                Vector::zeros(2)
            );
            assert_eq!(
                algorithm.m_dot_mat(&Matrix::identity(2)),
                Matrix::zeros(2, 2)
            );
            assert!(Arc::ptr_eq(algorithm.m_system.as_ref().unwrap(), &cached));
        }
    }

    #[test]
    fn cached_bounded_steps_match_fresh_factorizations_after_history_eviction() {
        let config = LBFGSBConfig::default()
            .with_memory_limit(3)
            .unwrap()
            .with_bounds([(-5.0, 5.0); 2])
            .unwrap();
        let (mut cached, mut status) = initialized_rosenbrock(&config);
        let mut fresh = cached.clone();
        let mut fresh_status = status.clone();
        let shared = cached.m_system.clone().unwrap();
        assert!(Arc::ptr_eq(fresh.m_system.as_ref().unwrap(), &shared));
        let mut saw_eviction = false;
        for step in 1..20 {
            if !fresh.s_history.is_empty() {
                fresh.m_system = Some(fresh_compact_solver(&fresh));
            }
            let old = cached.m_system.clone();
            cached
                .step(step, &Rosenbrock { n: 2 }, &mut status, &(), &config)
                .unwrap();
            fresh
                .step(step, &Rosenbrock { n: 2 }, &mut fresh_status, &(), &config)
                .unwrap();
            assert!(cached.x.sub(&fresh.x).norm() < 1e-10);
            assert!((cached.fx - fresh.fx).abs() < 1e-10);
            assert_eq!(status.evals, fresh_status.evals);
            if cached.s_history.len() == 3 {
                saw_eviction = true;
                assert!(!Arc::ptr_eq(
                    cached.m_system.as_ref().unwrap(),
                    old.as_ref().unwrap()
                ));
            }
        }
        assert!(saw_eviction);
    }

    #[test]
    fn rejected_search_and_reinitialization_discard_cached_factors() {
        #[derive(Clone, Default)]
        struct RejectSearch;

        impl<P: Gradient> LineSearch<f64, NalgebraProvider, P> for RejectSearch {
            fn search_from(
                &mut self,
                start: LineSearchStart<'_, f64, NalgebraProvider>,
                _: &Vector,
                _: Option<f64>,
                _: &P,
                _: &(),
                _: &mut crate::core::EvalCounts,
            ) -> Result<crate::traits::LineSearchResult<f64, NalgebraProvider>, Infallible>
            {
                Ok(Err(LineSearchOutput {
                    alpha: 0.0,
                    fx: start.fx,
                    gradient: start.gradient.clone(),
                }))
            }
        }

        let config = LBFGSBConfig::default()
            .with_bounds([(-5.0, 5.0); 2])
            .unwrap();
        let (fixture, mut status) = initialized_rosenbrock(&config);
        let mut algorithm = LBFGSB::<f64, NalgebraProvider, RejectSearch> {
            x: fixture.x,
            fx: fixture.fx,
            gradient: fixture.gradient,
            s_history: fixture.s_history,
            y_history: fixture.y_history,
            theta: fixture.theta,
            ..LBFGSB::default()
        };
        algorithm.update_compact_matrices();
        let rejected_config = LBFGSBConfig::<f64>::default()
            .with_bounds([(-5.0, 5.0); 2])
            .unwrap()
            .with_line_search(RejectSearch);
        algorithm
            .step(1, &Rosenbrock { n: 2 }, &mut status, &(), &rejected_config)
            .unwrap();
        assert!(algorithm.s_history.is_empty());
        assert!(algorithm.m_system.is_none());

        let (mut algorithm, mut status) = initialized_rosenbrock(&config);
        assert!(algorithm.m_system.is_some());
        algorithm
            .initialize(
                &Rosenbrock { n: 2 },
                &mut status,
                &(),
                &Vector::from_vec(vec![1.5, 2.0]),
                &config,
            )
            .unwrap();
        assert!(algorithm.s_history.is_empty());
        assert!(algorithm.m_system.is_none());
    }

    #[test]
    fn bounded_checkpoint_rebuilds_factors_and_resumes_identically() {
        let config = LBFGSBConfig::default()
            .with_bounds([(-5.0, 5.0); 2])
            .unwrap();
        let (mut original, mut status) = initialized_rosenbrock(&config);
        let checkpoint = <LBFGSB as CheckpointableAlgorithm<
            Rosenbrock,
            GradientStatus,
            (),
            Infallible,
        >>::checkpoint(&original, &status, 1);
        let mut restored = LBFGSB::default();
        let (mut restored_status, next_step) = <LBFGSB as CheckpointableAlgorithm<
            Rosenbrock,
            GradientStatus,
            (),
            Infallible,
        >>::restore(
            &mut restored, &checkpoint, &config
        );
        assert!(!Arc::ptr_eq(
            original.m_system.as_ref().unwrap(),
            restored.m_system.as_ref().unwrap()
        ));
        for step in next_step..next_step + 5 {
            original
                .step(step, &Rosenbrock { n: 2 }, &mut status, &(), &config)
                .unwrap();
            restored
                .step(
                    step,
                    &Rosenbrock { n: 2 },
                    &mut restored_status,
                    &(),
                    &config,
                )
                .unwrap();
            assert_eq!(original.x, restored.x);
            assert_eq!(original.gradient, restored.gradient);
            assert_eq!(original.fx, restored.fx);
            assert_eq!(status.evals, restored_status.evals);
        }
    }

    // Independent dense rank-two inverse-BFGS updates, rather than another two-loop recursion.
    fn dense_direction(algorithm: &LBFGSB<f64>) -> Vector {
        let n = algorithm.gradient.len();
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            h[i * n + i] = 1.0 / algorithm.theta;
        }
        for (s, y) in algorithm.s_history.iter().zip(&algorithm.y_history) {
            let sy = s.dot(y);
            let hy: Vec<f64> = (0..n)
                .map(|row| (0..n).map(|col| h[row * n + col] * y.get(col)).sum())
                .collect();
            let yhy: f64 = (0..n).map(|i| y.get(i) * hy[i]).sum();
            for row in 0..n {
                for col in 0..n {
                    h[row * n + col] += (1.0 + yhy / sy) * s.get(row) * s.get(col) / sy
                        - s.get(row).mul_add(hy[col], hy[row] * s.get(col)) / sy;
                }
            }
        }
        Vector::from_vec(
            (0..n)
                .map(|row| {
                    -(0..n)
                        .map(|col| h[row * n + col] * algorithm.gradient.get(col))
                        .sum::<f64>()
                })
                .collect(),
        )
    }

    #[test]
    fn two_loop_matches_dense_inverse_bfgs() {
        for count in [0, 1, 3, 7] {
            let algorithm = history_fixture(count);
            let expected = dense_direction(&algorithm);
            let actual = algorithm.unconstrained_direction();
            assert!(
                actual.sub(&expected).norm() < 1e-11 * (1.0 + expected.norm()),
                "history={count}"
            );
            assert!(actual.dot(&algorithm.gradient) < 0.0);
        }
    }

    #[test]
    fn two_loop_matches_the_compact_model() {
        for count in [1, 3] {
            let mut algorithm = history_fixture(count);
            algorithm.update_compact_matrices();
            let expected = algorithm.bound_constrained_direction(None);
            assert!(
                algorithm.unconstrained_direction().sub(&expected).norm()
                    < 1e-10 * (1.0 + expected.norm())
            );
        }
    }

    #[test]
    fn zero_gradient_produces_a_finite_zero_direction() {
        for count in [0, 3] {
            let mut algorithm = history_fixture(count);
            algorithm.gradient = Vector::zeros(4);
            assert_eq!(algorithm.unconstrained_direction(), Vector::zeros(4));
        }
    }

    #[test]
    fn two_loop_supports_native_f32() {
        let reference = history_fixture(3);
        let expected = dense_direction(&reference);
        let algorithm = LBFGSB::<f32> {
            x: Vector::zeros(4),
            gradient: Vector::from_vec(
                reference
                    .gradient
                    .to_vec()
                    .into_iter()
                    .map(|x| x as f32)
                    .collect(),
            ),
            theta: reference.theta as f32,
            s_history: reference
                .s_history
                .iter()
                .map(|s| Vector::from_vec(s.to_vec().into_iter().map(|x| x as f32).collect()))
                .collect(),
            y_history: reference
                .y_history
                .iter()
                .map(|y| Vector::from_vec(y.to_vec().into_iter().map(|x| x as f32).collect()))
                .collect(),
            ..LBFGSB::default()
        };
        let actual = algorithm.unconstrained_direction();
        for i in 0..4 {
            assert!((f64::from(actual.get(i)) - expected.get(i)).abs() < 2e-6);
        }
    }

    #[test]
    fn only_entirely_unbounded_native_limits_select_the_fast_path() {
        assert!(LBFGSB::<f64>::is_unconstrained(None));
        assert!(LBFGSB::<f64>::is_unconstrained(Some(
            &[ScalarBound::Unbounded; 3]
        )));
        for bound in [
            ScalarBound::Lower(0.0),
            ScalarBound::Upper(1.0),
            ScalarBound::Both(-1.0, 1.0),
        ] {
            assert!(!LBFGSB::<f64>::is_unconstrained(Some(&[
                ScalarBound::Unbounded,
                bound
            ])));
        }
    }

    fn initialized_rosenbrock(config: &LBFGSBConfig) -> (LBFGSB, GradientStatus) {
        let mut algorithm = LBFGSB::<f64>::default();
        let mut status = GradientStatus::default();
        algorithm
            .initialize(
                &Rosenbrock { n: 2 },
                &mut status,
                &(),
                &Vector::from_vec(vec![2.0, 2.0]),
                config,
            )
            .unwrap();
        algorithm
            .step(0, &Rosenbrock { n: 2 }, &mut status, &(), config)
            .unwrap();
        assert!(!algorithm.s_history.is_empty());
        (algorithm, status)
    }

    #[test]
    fn unconstrained_steps_do_not_build_compact_matrices() {
        for config in [
            LBFGSBConfig::default(),
            LBFGSBConfig::default()
                .with_bounds([(f64::NEG_INFINITY, f64::INFINITY); 2])
                .unwrap(),
        ] {
            let (algorithm, _) = initialized_rosenbrock(&config);
            assert!(algorithm.m_system.is_none());
            assert_eq!(algorithm.w_matrix.cols(), 0);
        }
    }

    #[test]
    fn adding_native_bounds_rebuilds_the_compact_model() {
        let (mut algorithm, mut status) = initialized_rosenbrock(&LBFGSBConfig::default());
        let mut reference = algorithm.clone();
        reference.update_compact_matrices();
        let mut reference_status = status.clone();
        let bounded = LBFGSBConfig::default()
            .with_bounds([(-5.0, 5.0); 2])
            .unwrap();
        algorithm
            .step(1, &Rosenbrock { n: 2 }, &mut status, &(), &bounded)
            .unwrap();
        reference
            .step(
                1,
                &Rosenbrock { n: 2 },
                &mut reference_status,
                &(),
                &bounded,
            )
            .unwrap();
        assert_eq!(algorithm.x, reference.x);
        assert_eq!(algorithm.gradient, reference.gradient);
        assert_eq!(algorithm.fx, reference.fx);
        assert!(algorithm.m_system.is_some());
    }

    #[test]
    fn removing_native_bounds_invalidates_old_compact_matrices() {
        let bounded = LBFGSBConfig::default()
            .with_bounds([(-5.0, 5.0); 2])
            .unwrap();
        let (mut algorithm, mut status) = initialized_rosenbrock(&bounded);
        assert!(algorithm.m_system.is_some());
        algorithm
            .step(
                1,
                &Rosenbrock { n: 2 },
                &mut status,
                &(),
                &LBFGSBConfig::default(),
            )
            .unwrap();
        assert!(algorithm.m_system.is_none());
        assert_eq!(algorithm.w_matrix.cols(), 0);
        assert!(!algorithm.s_history.is_empty());
    }

    #[test]
    fn checkpoints_restore_and_resume_both_paths() {
        let unbounded = LBFGSBConfig::default();
        let (mut algorithm, mut status) = initialized_rosenbrock(&unbounded);
        let checkpoint = <LBFGSB as CheckpointableAlgorithm<
            Rosenbrock,
            GradientStatus,
            (),
            Infallible,
        >>::checkpoint(&algorithm, &status, 1);
        let mut restored = LBFGSB::default();
        let (mut restored_status, next_step) = <LBFGSB as CheckpointableAlgorithm<
            Rosenbrock,
            GradientStatus,
            (),
            Infallible,
        >>::restore(
            &mut restored, &checkpoint, &unbounded
        );
        assert_eq!(next_step, 1);
        assert!(restored.m_system.is_none());
        assert_eq!(restored.w_matrix.cols(), 0);
        algorithm
            .step(1, &Rosenbrock { n: 2 }, &mut status, &(), &unbounded)
            .unwrap();
        restored
            .step(
                1,
                &Rosenbrock { n: 2 },
                &mut restored_status,
                &(),
                &unbounded,
            )
            .unwrap();
        assert_eq!(algorithm.x, restored.x);
        assert_eq!(algorithm.fx, restored.fx);
        assert_eq!(algorithm.gradient, restored.gradient);
        assert_eq!(status.evals, restored_status.evals);

        let bounded = LBFGSBConfig::default()
            .with_bounds([(-5.0, 5.0); 2])
            .unwrap();
        let _ = <LBFGSB as CheckpointableAlgorithm<Rosenbrock, GradientStatus, (), Infallible>>::restore(&mut restored, &checkpoint, &bounded);
        assert!(restored.m_system.is_some());
        assert_eq!(restored.w_matrix.cols(), 2 * checkpoint.s_history.len());
    }

    struct MixedBoundQuadratic;

    impl CostFunction for MixedBoundQuadratic {
        fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, Infallible> {
            Ok([
                0.5 * (x.get(0) - 0.5).powi(2),
                2.0 * (x.get(1) + 2.0).powi(2),
                4.5 * (x.get(2) - 3.0).powi(2),
            ]
            .into_iter()
            .sum())
        }
    }

    impl Gradient for MixedBoundQuadratic {
        fn gradient(&self, x: &Vector, _: &()) -> Result<Vector, Infallible> {
            Ok(Vector::from_vec(vec![
                x.get(0) - 0.5,
                4.0 * (x.get(1) + 2.0),
                9.0 * (x.get(2) - 3.0),
            ]))
        }
    }

    #[test]
    fn mixed_native_bounds_keep_the_constrained_solution() {
        let mut algorithm = LBFGSB::<f64>::default();
        let result = algorithm
            .process(
                &MixedBoundQuadratic,
                &(),
                Vector::from_vec(vec![1.0, 1.0, 0.0]),
                LBFGSBConfig::default()
                    .with_error_mode(LBFGSBErrorMode::Skip)
                    .with_bounds([
                        (f64::NEG_INFINITY, f64::INFINITY),
                        (0.0, f64::INFINITY),
                        (f64::NEG_INFINITY, 2.0),
                    ])
                    .unwrap(),
                LBFGSB::<f64>::default_callbacks(),
            )
            .unwrap();
        assert!((result.x.get(0) - 0.5).abs() < 1e-5);
        assert_eq!(result.x.get(1), 0.0);
        assert_eq!(result.x.get(2), 2.0);
        assert!((result.fx - 12.5).abs() < 1e-10);
        assert!(algorithm.m_system.is_some());
    }
}
