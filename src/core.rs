use nalgebra::{DMatrix, DVector, RealField};

/// Represents a multivariate function that can be evaluated and differentiated.
///
/// This trait is generic over the field type `F`, an argument type `A`, and an error type `E`.
///
/// # Type Parameters
///
/// * `F`: A type that implements [`RealField`], [`Copy`], [`From<f32>`], and has a
///   `'static` lifetime.
/// * `A`: A type that can be used to pass arguments to the wrapped function.
/// * `E`: The error type returned by the function's methods.
pub trait Function<F, A, E>: Send + Sync
where
    F: RealField + Copy + From<f32> + 'static,
{
    /// Evaluates the function at the given point.
    ///
    /// # Arguments
    ///
    /// * `x`: A [`DVector`] of `F` representing the point at which to evaluate the function.
    /// * `args`: An optional argument struct used to pass static arguments to the internal
    ///   function.
    ///
    /// # Returns
    ///
    /// The function value at `x` of type `F`.
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if the evaluation fails.
    fn evaluate(&self, x: &DVector<F>, args: Option<&A>) -> Result<F, E>;

    /// Computes the gradient of the function at the given point using central finite
    /// differences.
    ///
    /// Overwrite this method if the true gradient function is known.
    ///
    /// # Arguments
    ///
    /// * `x`: A [`DVector`] of `F` representing the point at which to compute the gradient.
    /// * `args`: An optional argument struct used to pass static arguments to the internal
    ///   function.
    ///
    /// # Returns
    ///
    /// A [`DVector`] of `F` representing the gradient.
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if [`Function::evaluate`] fails.
    fn gradient(&self, x: &DVector<F>, args: Option<&A>) -> Result<DVector<F>, E> {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        // This is technically the best step size for the gradient, cbrt(eps) * x_i (or just
        // cbrt(eps) if x_i = 0)
        let h: Vec<F> = x
            .iter()
            .map(|&xi| {
                F::cbrt(F::default_epsilon()) * (if xi == F::zero() { F::one() } else { xi })
            })
            .collect();
        for i in 0..n {
            let mut x_plus = x.clone_owned();
            let mut x_minus = x.clone_owned();
            x_plus[i] += h[i];
            x_minus[i] -= h[i];
            let f_plus = self.evaluate(&x_plus, args)?;
            let f_minus = self.evaluate(&x_minus, args)?;
            grad[i] = (f_plus - f_minus) / (F::from(2.0) * h[i]);
        }

        Ok(grad)
    }

    /// Computes both the gradient and the Hessian matrix of the function at the given point.
    ///
    /// This method uses central finite differences to approximate both the gradient and the Hessian.
    ///
    /// # Arguments
    ///
    /// * `x`: A [`DVector`] of `F` representing the point at which to compute the gradient and Hessian.
    /// * `args`: An optional argument struct used to pass static arguments to the internal
    ///   function.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - The gradient at `x` as a [`DVector`] of `F`
    /// - The Hessian at `x` as a [`DMatrix`] of `F`
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if [`Function::evaluate`] fails.
    fn gradient_and_hessian(
        &self,
        x: &DVector<F>,
        args: Option<&A>,
    ) -> Result<(DVector<F>, DMatrix<F>), E> {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        let mut hess = DMatrix::zeros(n, n);
        // This is technically the best step size for the gradient, cbrt(eps) * x_i (or just
        // cbrt(eps) if x_i = 0)
        let h: Vec<F> = x
            .iter()
            .map(|&xi| {
                F::cbrt(F::default_epsilon()) * (if xi == F::zero() { F::one() } else { xi })
            })
            .collect();
        let two = F::from(2.0);
        let four = F::from(4.0);

        // Compute Hessian
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal element using central difference
                    let mut x_plus = x.clone_owned();
                    let mut x_minus = x.clone_owned();
                    x_plus[i] += h[i];
                    x_minus[i] -= h[i];

                    let f_plus = self.evaluate(&x_plus, args)?;
                    let f_minus = self.evaluate(&x_minus, args)?;
                    let f_center = self.evaluate(x, args)?;

                    grad[i] = (f_plus - f_minus) / (two * h[i]);
                    hess[(i, i)] = (f_plus - two * f_center + f_minus) / (h[i] * h[i]);
                } else {
                    // Off-diagonal element
                    let mut x_plus_plus = x.clone_owned();
                    let mut x_plus_minus = x.clone_owned();
                    let mut x_minus_plus = x.clone_owned();
                    let mut x_minus_minus = x.clone_owned();

                    x_plus_plus[i] += h[i];
                    x_plus_plus[j] += h[i];
                    x_plus_minus[i] += h[i];
                    x_plus_minus[j] -= h[i];
                    x_minus_plus[i] -= h[i];
                    x_minus_plus[j] += h[i];
                    x_minus_minus[i] -= h[i];
                    x_minus_minus[j] -= h[i];

                    let f_plus_plus = self.evaluate(&x_plus_plus, args)?;
                    let f_plus_minus = self.evaluate(&x_plus_minus, args)?;
                    let f_minus_plus = self.evaluate(&x_minus_plus, args)?;
                    let f_minus_minus = self.evaluate(&x_minus_minus, args)?;

                    hess[(i, j)] = (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus)
                        / (four * h[i] * h[i]);
                    hess[(j, i)] = hess[(i, j)];
                }
            }
        }

        Ok((grad, hess))
    }

    /// Computes both the gradient and the inverse of the Hessian matrix of the function at the given point.
    ///
    /// This method uses central finite differences to approximate both the gradient and the Hessian. It
    /// then returns the gradient along with the inverse (if Hessian is invertable) or the pseudoinverse
    /// of the Hessian.
    ///
    /// # Arguments
    ///
    /// * `x`: A [`DVector`] of `F` representing the point at which to compute the gradient and Hessian.
    /// * `args`: An optional argument struct used to pass static arguments to the internal
    ///   function.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - The gradient at `x` as a [`DVector`] of `F`
    /// - The inverse of the Hessian at `x` as a [`DMatrix`] of `F`
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if [`Function::evaluate`] fails.
    fn gradient_and_inverse_hessian(
        &self,
        x: &DVector<F>,
        args: Option<&A>,
    ) -> Result<(DVector<F>, DMatrix<F>), E> {
        let (gradient, hessian) = self.gradient_and_hessian(x, args)?;
        if hessian.is_invertible() {
            return Ok((gradient, hessian.try_inverse().expect("Hessian isn't square, something is horribly wrong. Please create an issue on the GitHub repository for `ganesh`!")));
        } else {
            return Ok((
                gradient,
                hessian
                    .pseudo_inverse(F::from(f32::EPSILON))
                    .expect("SVD pseudo inverse: the epsilon must be non-negative.\nThis is an `nalgebra` error! If this happens, please create an issue on the GitHub repository for `ganesh`!"),
            ));
        }
    }
}

/// Represents an optimization algorithm for minimizing a function.
///
/// This trait is generic over the field type `F`, a message type `M`, and an error type `E`.
///
/// # Type Parameters
///
/// * `F`: A generic field used to represent real numbers.
/// * `A`: A type that can be used to pass arguments to the wrapped function.
/// * `E`: The error type returned by the minimizer's methods.
pub trait Minimizer<F, A, E> {
    /// Start the algorithm with any initialization steps or evaluations.
    ///
    /// # Arguments
    ///
    /// * `args`: An optional argument struct used to pass static arguments to the internal
    ///   function.
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if the initialization fails.
    fn initialize(&mut self, _args: Option<&A>) -> Result<(), E> {
        Ok(())
    }

    /// Performs a single step of the minimization algorithm.
    ///
    /// # Arguments
    ///
    /// * `args`: An optional argument struct used to pass static arguments to the internal
    ///   function.
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if the step fails.
    fn step(&mut self, args: Option<&A>) -> Result<(), E>;

    /// Checks if the termination condition for the algorithm has been met.
    ///
    /// # Returns
    ///
    /// `true` if the algorithm should terminate, `false` otherwise.
    fn check_for_termination(&self) -> bool;

    /// Update the best point found in the algorithm.
    ///
    /// Run any methods to update the best-yet point and evaluation. This method should probably
    /// set some fields like `self.fx_best` and `self.x_best` which are then returned by
    /// [`Minimizer::best`].
    fn update_best(&mut self);

    /// Runs the minimization algorithm to completion, calling the provided callback after each step.
    ///
    /// # Type Parameters
    ///
    /// * `Callback`: A function type that takes a message of type `M` as an argument.
    ///
    /// # Arguments
    ///
    /// * `args`: An optional argument struct used to pass static arguments to the internal
    ///   function.
    /// * `callback`: A function that is called after each step with a message of type `M`. Use
    ///   `minimize(|_| {})` as a pass-through function if you don't need a callback.
    ///
    /// # Errors
    ///
    /// Returns an error of type `E` if the minimization process fails.
    fn minimize<Callback: Fn(&Self)>(
        &mut self,
        args: Option<&A>,
        steps: usize,
        callback: Callback,
    ) -> Result<(), E> {
        self.initialize(args)?;
        for _ in 0..steps {
            self.step(args)?;
            self.update_best();
            callback(self);
            if self.check_for_termination() {
                return Ok(());
            }
        }
        Ok(())
    }

    /// Returns the best solution found so far by the minimization algorithm.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A [`DVector`] of `F` representing the best point found.
    /// - The function value of type `F` at the best point.
    ///
    /// These are both returned as references with lifetimes tied to the minimizing struct.
    fn best(&self) -> (&DVector<F>, &F);
}

/// A macro to clean up minimization statements
///
/// Usage:
///
/// Run a maximum of 1000 steps of an algorithm:
/// ```ignore
/// minimize!(minimization_algorithm, 1000)?;
/// ```
///
/// Run a maximum of 1000 steps of an algorithm, printing the best result after each step:
/// ```ignore
/// minimize!(minimization_algorithm, 1000, |ma| println!("{:?}", ma.best))?;
/// ```
#[macro_export]
macro_rules! minimize {
    ($minimizer:expr, $nsteps:expr) => {
        $minimizer.minimize(None, $nsteps as usize, |_| {})
    };
    ($minimizer:expr, $nsteps:expr, $callback:expr) => {
        $minimizer.minimize(None, $nsteps as usize, $callback)
    };
}

/// A trait which describes line search algorithms.
///
/// These algorithms typically involve some step direction given by `p` from the current location
/// `x`. The implementation dictates how to use that information to get a "learning rate" `alpha`
/// which determines some step in the direction `p`:
///
/// ```math
/// \vec{x}_{i+1} = \vec{x}_{i} + \alpha \vec{p}
/// ```
///
/// These methods occasionally require information about the previous iteration, so `x_prev`,
/// `p_prev`, and `alpha_prev` are optionally given in the signature of the [`LineSearch::search`]
/// method to provide these values.
///
/// See also: [`GradientDescent`](`crate::algorithms::GradientDescent`)
pub trait LineSearch<F, A, E>: std::fmt::Debug
where
    F: RealField + Copy + From<f32>,
{
    /// A method which takes a function `func` and its arguments `args`, along with the current
    /// position of the optimizer `x` (and optionally the previous position, `x_prev`), a step
    /// direction vector `p` (and optionally the previous step direction vector, `p_prev`), and
    /// optionally the previous learning rate, `alpha_prev`.
    ///
    /// # Returns
    /// A tuple containing ($`\vec{x}_{i+1}`$, $`f(\vec{x}_{i+1})`$, $`\alpha`$), where $`\alpha`$
    /// was the learning rate used to achieve this step.
    ///
    /// # Errors
    /// This method returns an error of generic type `E` given by the [`Function`] being optimized
    /// if that function fails.
    fn search(
        &self,
        func: &dyn Function<F, A, E>,
        args: Option<&A>,
        x: &DVector<F>,
        x_prev: &Option<DVector<F>>,
        p: &DVector<F>,
        p_prev: &Option<DVector<F>>,
        alpha_prev: Option<F>,
    ) -> Result<(DVector<F>, F, F), E>;

    /// A method to get the base learning rate $`\alpha_0`$.
    fn get_base_learning_rate(&self) -> F;
}
