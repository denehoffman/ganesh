use nalgebra::DVector;
use typed_builder::TypedBuilder;

use crate::core::{convert, Field, Function, LineSearch};

/// A trivial line search algorithm which just steps at a fixed rate.
///
/// ```math
/// \vec{x}_{i+1} = \vec{x}_i + \alpha \vec{p}
/// ```
#[derive(TypedBuilder)]
pub struct FixedRateLineSearch<F>
where
    F: Field,
{
    #[builder(default = F::one())]
    base_learning_rate: F,
}

impl<F, A, E> LineSearch<F, A, E> for FixedRateLineSearch<F>
where
    F: Field + 'static,
{
    fn search(
        &self,
        func: &dyn Function<F, A, E>,
        args: Option<&A>,
        x: &DVector<F>,
        _x_prev: &Option<DVector<F>>,
        p: &DVector<F>,
        _p_prev: &Option<DVector<F>>,
        _alpha_prev: Option<F>,
    ) -> Result<(DVector<F>, F, F), E> {
        let res_x = x + p * self.base_learning_rate;
        let res_fx = func.evaluate(&res_x, args)?;
        Ok((res_x, res_fx, self.base_learning_rate))
    }

    fn get_base_learning_rate(&self) -> F {
        self.base_learning_rate
    }
}

/// A backtracking line search.
///
/// Given a maximum step size $`\alpha_0`$, a shrink parameter
/// $`\tau \in (0, 1)`$, and a control parameter $`c \in (0, 1)`$,
/// this method iteratively computes an optimal step size by the
/// following algorithm[^1]:
///
/// 1. Compute $`t = c\vec{g}\cdot\vec{p}`$ where $`\vec{g}`$ is the gradient at $`\vec{x}`$.
/// 2. While $`f(\vec{x}) - f(\vec{x} + \alpha_j \vec{p}) \geq \alpha_j t`$, let $`\alpha_{j+1} =
///    \tau\alpha_{j}`$.
/// 3. Return the final value of $`\alpha_{j}`$ which satisfies the condition.
///
/// Then, the step is given by:
/// ```math
/// \vec{x}_{i+1} = \vec{x}_i + \alpha \vec{p}
/// ```
///
/// [^1]: Armijo, Larry (1966). "Minimization of functions having Lipschitz continuous first partial derivatives". Pacific J. Math. **16** (1): 1–3. doi:10.2140/pjm.1966.16.1.
#[derive(TypedBuilder)]
pub struct BacktrackingLineSearch<F>
where
    F: Field + 'static,
{
    #[builder(default = convert!(0.5, F))]
    control: F,
    #[builder(default = convert!(0.5, F))]
    shrink: F,
    #[builder(default = F::one())]
    base_learning_rate: F,
}

impl<F, A, E> LineSearch<F, A, E> for BacktrackingLineSearch<F>
where
    F: Field,
{
    fn search(
        &self,
        func: &dyn Function<F, A, E>,
        args: Option<&A>,
        x: &DVector<F>,
        _x_prev: &Option<DVector<F>>,
        p: &DVector<F>,
        _p_prev: &Option<DVector<F>>,
        _alpha_prev: Option<F>,
    ) -> Result<(DVector<F>, F, F), E> {
        let g = func.gradient(x, args)?;
        let m = g.dot(p);
        let t = -self.control * m;
        let mut alpha_i = self.base_learning_rate;
        loop {
            let res_x = x + p * alpha_i;
            let res_fx = func.evaluate(&res_x, args)?;
            if (func.evaluate(x, args)? - res_fx) >= (alpha_i * t) {
                return Ok((res_x, res_fx, alpha_i));
            }
            alpha_i *= self.shrink;
        }
    }

    fn get_base_learning_rate(&self) -> F {
        self.base_learning_rate
    }
}
/// A two-way backtracking line search.
///
/// This is similar to the backtracking line search implemented in
/// [`BacktrackingLineSearch`](`BacktrackingLineSearch`)[^1]. However, with the assumption that the next value of the
/// learning rate $`\alpha`$ is similar to the learning rate from the previous step, this method
/// can be used to skip a lot of the steps involved in the convergence to the Armijo condition,
/// especially if the base learning rate is set too high. However, it is possible that the
/// optimal step is larger than the previous step (but no larger than the base step), so this
/// algorithm begins with a growth phase rather than shrinking the learning rate if it already
/// satisfies the Armijo condition. The growth rate is just the inverse of the shrink parameter
/// used in the typical backtracking line search.[^2] [^3]
///
/// [^1]: Armijo, Larry (1966). "Minimization of functions having Lipschitz continuous first partial derivatives". Pacific J. Math. **16** (1): 1–3. doi:10.2140/pjm.1966.16.1.
/// [^2]: Nocedal, Jorge; Wright, Stephen J. (2000), Numerical Optimization, Springer-Verlag, ISBN 0-387-98793-2
/// [^3]: Truong, T. T.; Nguyen, H.-T. (6 September 2020). "Backtracking Gradient Descent Method and Some Applications in Large Scale Optimisation. Part 2: Algorithms and Experiments". Applied Mathematics & Optimization. **84** (3): 2557–2586. doi:10.1007/s00245-020-09718-8. hdl:10852/79322.
#[derive(TypedBuilder)]
pub struct TwoWayBacktrackingLineSearch<F>
where
    F: Field,
{
    #[builder(default = convert!(0.5, F))]
    control: F,
    #[builder(default = convert!(0.5, F))]
    shrink: F,
    #[builder(default = F::one())]
    base_learning_rate: F,
}

impl<F, A, E> LineSearch<F, A, E> for TwoWayBacktrackingLineSearch<F>
where
    F: Field + 'static,
{
    fn search(
        &self,
        func: &dyn Function<F, A, E>,
        args: Option<&A>,
        x: &DVector<F>,
        _x_prev: &Option<DVector<F>>,
        p: &DVector<F>,
        _p_prev: &Option<DVector<F>>,
        alpha_prev: Option<F>,
    ) -> Result<(DVector<F>, F, F), E> {
        let g = func.gradient(x, args)?;
        let m = g.dot(p);
        let t = -self.control * m;
        let mut alpha = alpha_prev.unwrap_or(self.base_learning_rate);
        // In this case, alpha_0 := alpha_{n-1} and
        // base_learning_rate := alpha_0, so we might want to increase
        // the learning rate first:
        let res_x = x + p * alpha;
        let res_fx = func.evaluate(&res_x, args)?;
        if (func.evaluate(x, args)? - res_fx) >= (alpha * t) {
            // if Armijo's condition is already satisfied
            let mut res_x_old = res_x;
            let mut res_fx_old = res_fx;
            loop {
                alpha /= self.shrink; // increase learning rate
                let res_x = x + p * alpha;
                let res_fx = func.evaluate(&res_x, args)?;
                if (func.evaluate(x, args)? - res_fx) < (alpha * t)
                    || alpha > self.base_learning_rate
                {
                    // if we pass the best, return the results from the previous step
                    return Ok((res_x_old, res_fx_old, alpha * self.shrink));
                }
                res_x_old = res_x;
                res_fx_old = res_fx;
            }
        } else {
            // if Armijo's condition is not satisfied
            loop {
                alpha *= self.shrink; // decrease learning rate
                let res_x = x + p * alpha;
                let res_fx = func.evaluate(&res_x, args)?;
                if (func.evaluate(x, args)? - res_fx) >= (alpha * t) {
                    // once satisfied, return the values which satisfied the condition
                    return Ok((res_x, res_fx, alpha));
                }
            }
        }
    }

    fn get_base_learning_rate(&self) -> F {
        self.base_learning_rate
    }
}

/// Types of steps used by the Barzilai-Borwein algorithm implemented in [`BarzilaiBorwein`].
pub enum BarzilaiBorweinStep {
    /// A short step which uses $`\alpha =
    /// \frac{\Delta\vec{x}\cdot\Delta\vec{x}}{\Delta\vec{x}\cdot\Delta\vec{g}}`$.
    Short,
    /// A long step which uses $`\alpha =
    /// \frac{\Delta\vec{x}\cdot\Delta\vec{g}}{\Delta\vec{g}\cdot\Delta\vec{g}}`$.
    Long,
}

/// An implementation of the Barzilai-Borwein method, which uses the previous two positions
/// $`\vec{x}_i`$ and $`\vec{x}_{i-1}`$ along with the previous two gradient evaluations, $`\vec{g}_i`$ and $`\vec{g}_{i-1}`$ to determine either a long or short stepped learning rate:[^1]
///
/// ```math
/// \alpha_S = \frac{\Delta\vec{x}\cdot\Delta\vec{x}}{\Delta\vec{x}\cdot\Delta\vec{g}}
/// ```
/// or
/// ```math
/// \alpha_L = \frac{\Delta\vec{x}\cdot\Delta\vec{g}}{\Delta\vec{g}\cdot\Delta\vec{g}}
/// ```
///
/// ```math
/// \alpha_L = \frac{\Delta\vec{x}\cdot\Delta\vec{g}}{\Delta\vec{g}\cdot\Delta\vec{g}}
/// ```
///
/// This implementation uses a [`TwoWayBacktrackingLineSearch`] step if either the previous
/// position or gradient is unavailable (for instance, in the first step of an optimization).
///
/// Note that, because we define $`\vec{p}\equiv -\vec{g}`$, we must also have
/// $`\Delta\vec{g}\equiv -\Delta\vec{p}`$ in the implementation below.
///
/// [^1]: Barzilai, Jonathan; Borwein, Jonathan M. (1988). "Two-Point Step Size Gradient Methods". IMA Journal of Numerical Analysis. **8**: 141–148. doi:10.1093/imanum/8.1.141.
#[derive(TypedBuilder)]
pub struct BarzilaiBorwein<F>
where
    F: Field,
{
    #[builder(default = convert!(0.5, F))]
    control: F,
    #[builder(default = convert!(0.5, F))]
    shrink: F,
    #[builder(default = F::one())]
    base_learning_rate: F,
    #[builder(default = BarzilaiBorweinStep::Short)]
    step_type: BarzilaiBorweinStep,
}

impl<F, A, E> LineSearch<F, A, E> for BarzilaiBorwein<F>
where
    F: Field + 'static,
{
    fn search(
        &self,
        func: &dyn Function<F, A, E>,
        args: Option<&A>,
        x: &DVector<F>,
        x_prev: &Option<DVector<F>>,
        p: &DVector<F>,
        p_prev: &Option<DVector<F>>,
        alpha_prev: Option<F>,
    ) -> Result<(DVector<F>, F, F), E> {
        if let (Some(x_p), Some(p_p)) = (x_prev, p_prev) {
            let dx = x - x_p;
            let dp = p_p - p;
            let alpha = match self.step_type {
                BarzilaiBorweinStep::Short => dx.dot(&dx) / dx.dot(&dp),
                BarzilaiBorweinStep::Long => dx.dot(&dp) / dp.dot(&dp),
            };
            let res_x = x + p * alpha;
            let res_fx = func.evaluate(&res_x, args)?;
            Ok((res_x, res_fx, alpha))
        } else {
            let g = func.gradient(x, args)?;
            let m = g.dot(p);
            let t = -self.control * m;
            let mut alpha = alpha_prev.unwrap_or(self.base_learning_rate);
            // In this case, alpha_0 := alpha_{n-1} and
            // base_learning_rate := alpha_0, so we might want to increase
            // the learning rate first:
            let res_x = x + p * alpha;
            let res_fx = func.evaluate(&res_x, args)?;
            if (func.evaluate(x, args)? - res_fx) >= (alpha * t) {
                // if Armijo's condition is already satisfied
                let mut res_x_old = res_x;
                let mut res_fx_old = res_fx;
                loop {
                    alpha /= self.shrink; // increase learning rate
                    let res_x = x + p * alpha;
                    let res_fx = func.evaluate(&res_x, args)?;
                    if (func.evaluate(x, args)? - res_fx) < (alpha * t)
                        || alpha > self.base_learning_rate
                    {
                        // if we pass the best, return the results from the previous step
                        return Ok((res_x_old, res_fx_old, alpha * self.shrink));
                    }
                    res_x_old = res_x;
                    res_fx_old = res_fx;
                }
            } else {
                // if Armijo's condition is not satisfied
                loop {
                    alpha *= self.shrink; // decrease learning rate
                    let res_x = x + p * alpha;
                    let res_fx = func.evaluate(&res_x, args)?;
                    if (func.evaluate(x, args)? - res_fx) >= (alpha * t) {
                        // once satisfied, return the values which satisfied the condition
                        return Ok((res_x, res_fx, alpha));
                    }
                }
            }
        }
    }

    fn get_base_learning_rate(&self) -> F {
        self.base_learning_rate
    }
}
