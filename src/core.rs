use nalgebra::{ComplexField, DMatrix, DVector};
use num_traits::Float;

/// A trait that extends [`Float`] and [`ComplexField`] with additional conversion methods.
///
/// This trait is implemented for types that satisfy the following bounds:
/// - `Float`: Provides basic floating-point operations.
/// - `ComplexField`: Allows for complex number operations.
/// - `std::iter::Sum`: Enables summing of collections of this type.
pub trait Field: Float + ComplexField + std::iter::Sum {
    /// Converts an f64 value to Self.
    ///
    /// # Arguments
    ///
    /// * `x` - The f64 value to convert.
    ///
    /// # Returns
    ///
    /// The converted value of type Self.
    fn convert(x: f64) -> Self;

    /// Converts a usize value to Self.
    ///
    /// # Arguments
    ///
    /// * `x` - The usize value to convert.
    ///
    /// # Returns
    ///
    /// The converted value of type Self.
    fn convert_usize(x: usize) -> Self;
}
impl Field for f32 {
    fn convert(x: f64) -> Self {
        x as Self
    }

    fn convert_usize(x: usize) -> Self {
        x as Self
    }
}
impl Field for f64 {
    fn convert(x: f64) -> Self {
        x as Self
    }

    fn convert_usize(x: usize) -> Self {
        x as Self
    }
}

/// Represents a multivariate function that can be evaluated and differentiated.
///
/// This trait is generic over the field type `F`, an argument type `A`, and an error type `E`.
///
/// # Type Parameters
///
/// * `F`: A type that implements [`Field`] and has a `'static` lifetime.
/// * `A`: A type that can be used to pass arguments to the wrapped function.
/// * `E`: The error type returned by the function's methods.
pub trait Function<F, A, E>: Send + Sync
where
    F: Field + 'static,
{
    /// Evaluates the function at the given point.
    ///
    /// # Arguments
    ///
    /// * `x`: A slice of `F` representing the point at which to evaluate the function.
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
    fn evaluate(&self, x: &[F], args: Option<&A>) -> Result<F, E>;

    /// Computes the gradient of the function at the given point using central finite
    /// differences.
    ///
    /// Overwrite this method if the true gradient function is known.
    ///
    /// # Arguments
    ///
    /// * `x`: A slice of `F` representing the point at which to compute the gradient.
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
    fn gradient(&self, x: &[F], args: Option<&A>) -> Result<DVector<F>, E> {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        let h: Vec<F> = x
            .iter()
            .map(|&xi| {
                ComplexField::cbrt(F::epsilon()) * (if xi == F::zero() { F::one() } else { xi })
            })
            .collect();

        for i in 0..n {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            x_plus[i] += h[i];
            x_minus[i] -= h[i];
            let f_plus = self.evaluate(&x_plus, args)?;
            let f_minus = self.evaluate(&x_minus, args)?;
            grad[i] = (f_plus - f_minus) / (F::convert(2.0) * h[i]);
        }

        Ok(grad)
    }

    /// Computes both the gradient and the Hessian matrix of the function at the given point.
    ///
    /// This method uses central finite differences to approximate both the gradient and the Hessian.
    ///
    /// # Arguments
    ///
    /// * `x`: A slice of `F` representing the point at which to compute the gradient and Hessian.
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
        x: &[F],
        args: Option<&A>,
    ) -> Result<(DVector<F>, DMatrix<F>), E> {
        let n = x.len();
        let mut grad = DVector::zeros(n);
        let mut hess = DMatrix::zeros(n, n);
        let h: Vec<F> = x
            .iter()
            .map(|&xi| {
                ComplexField::cbrt(F::epsilon()) * (if xi == F::zero() { F::one() } else { xi })
            })
            .collect();
        let two = F::convert(2.0);
        let four = two * two;

        // Compute Hessian
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal element using central difference
                    let mut x_plus = x.to_vec();
                    let mut x_minus = x.to_vec();
                    x_plus[i] += h[i];
                    x_minus[i] -= h[i];

                    let f_plus = self.evaluate(&x_plus, args)?;
                    let f_minus = self.evaluate(&x_minus, args)?;
                    let f_center = self.evaluate(x, args)?;

                    grad[i] = (f_plus - f_minus) / (two * h[i]);
                    hess[(i, i)] = (f_plus - two * f_center + f_minus) / (h[i] * h[i]);
                } else {
                    // Off-diagonal element
                    let mut x_plus_plus = x.to_vec();
                    let mut x_plus_minus = x.to_vec();
                    let mut x_minus_plus = x.to_vec();
                    let mut x_minus_minus = x.to_vec();

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
}

/// Represents an optimization algorithm for minimizing a function.
///
/// This trait is generic over the field type `F`, a message type `M`, and an error type `E`.
///
/// # Type Parameters
///
/// * `F`: A type that implements [`Field`].
/// * `A`: A type that can be used to pass arguments to the wrapped function.
/// * `E`: The error type returned by the minimizer's methods.
pub trait Minimizer<F, A, E>
where
    F: Field,
{
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
    /// - A vector of `F` representing the best point found.
    /// - The function value of type `F` at the best point.
    fn best(&self) -> (&Vec<F>, &F);
}

/// A macro to clean up minimization statements
#[macro_export]
macro_rules! minimize {
    ($minimizer:expr, $nsteps:expr) => {
        $minimizer.minimize(None, $nsteps as usize, |_| {})
    };
    ($minimizer:expr, $nsteps:expr, $callback:expr) => {
        $minimizer.minimize(None, $nsteps as usize, $callback)
    };
}
