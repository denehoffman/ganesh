#![allow(clippy::suboptimal_flops)]
use std::{convert::Infallible, f64::consts::PI};

use crate::Function;

/// The Rastrigin function, a non-convex function with multiple modes.
///
/// ```math
/// f(\vec{x}) = 10n + \sum_{i=1}^{n} [x_i^2 - 10\cos(2\pi x_i)]
/// ```
/// where $`x_i \in [-5.12, 5.12]`$. The global minimum is $`f(\vec{0}) = 0`$.
pub struct Rastrigin {
    /// Number of dimensions
    pub n: usize,
}
impl Function<f64, Infallible> for Rastrigin {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
        Ok(10.0 * (self.n as f64)
            + (0..self.n)
                .map(|i| x[i].powi(2) - 10.0 * f64::cos(2.0 * PI * x[i]))
                .sum::<f64>())
    }
}

/// A generalized spherical function with a single minimum.
///
/// ```math
/// f(\vec{x}) = \sum_{i=1}^{n} x_i^2
/// ```
/// The global minimum is at $`f(\vec{0}) = 0`$.
pub struct Sphere {
    /// Number of dimensions
    pub n: usize,
}
impl Function<f64, Infallible> for Sphere {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
        Ok((0..self.n).map(|i| x[i].powi(2)).sum())
    }
}

/// The Rosenbrock function, a non-convex function with a single minimum.
///
/// ```math
/// f(\vec{x}) = \sum_{i=1}^{n-1} \left[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right]
/// ```
/// where $`n \geq 2`$. This function has a minimum at $`f(\vec{1}) = 0`$.
pub struct Rosenbrock {
    /// Number of dimensions (must be at least 2)
    pub n: usize,
}
impl Function<f64, Infallible> for Rosenbrock {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
        Ok((0..(self.n - 1))
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum())
    }
}

/// The Goldstein-Price function, which has several local minima and a single global minimum.
///
/// ```math
/// f(x, y) = \left[1 + (x + y + 1)^2 (19 - 14x + 3x^2 - 14y + 6xy + 3y^2)\right]\left[30 +
/// (2x-3y)^2 (18 - 32x + 12x^2 + 48y - 36xy + 27y^2)\right]
/// ```
/// where $`-2 \leq x,y \leq 2`$. This function has a global minimum at $`f(0, -1) = 3`$.
pub struct GoldsteinPrice;
impl Function<f64, Infallible> for GoldsteinPrice {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
        Ok((1.0
            + (x[0] + x[1] + 1.0).powi(2)
                * (19.0 - 14.0 * x[0] + 3.0 * x[0].powi(2) - 14.0 * x[1]
                    + 6.0 * x[0] * x[1]
                    + 3.0 * x[1].powi(2)))
            * (30.0
                + (2.0 * x[0] - 3.0 * x[1]).powi(2)
                    * (18.0 - 32.0 * x[0] + 12.0 * x[0].powi(2) + 48.0 * x[1]
                        - 36.0 * x[0] * x[1]
                        + 27.0 * x[1].powi(2))))
    }
}

/// The sixth Bukin function, which has a single global minimum in a very long and narrow valley.
///
/// ```math
/// f(x,y) = 100 \sqrt{\left|y - 0.01x^2\right|} + 0.01\left| x + 10 \right|
/// ```
/// where $`-15 \leq x \leq -5`$ and $`-3 \leq y \leq 3`$. This function has a global minimum at
/// $`f(-10, 1) = 0`$.
pub struct Bukin6;
impl Function<f64, Infallible> for Bukin6 {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
        Ok(100.0 * f64::sqrt(f64::abs(x[1] - 0.01 * x[0].powi(2))) + 0.01 * f64::abs(x[0] + 10.0))
    }
}

/// Powell's quartic function.
///
/// ```math
/// f(x,y,z,w) = (x + 10y)^2 + 5(z - w)^2 + (y - 2z)^4 + 10(x - w)^4
/// ```
/// where $`-4 \leq x,y,z,w \leq 5`$. This function has a global minimum at $`f(0,0,0,0) = 0`$.
pub struct Powell;
impl Function<f64, Infallible> for Powell {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
        Ok((x[0] + 10.0 * x[1]).powi(2)
            + 5.0 * (x[2] - x[3]).powi(2)
            + (x[1] - 2.0 * x[2]).powi(4)
            + 10.0 * (x[0] - x[3]).powi(4))
    }
}

/// Powell and Fletcher's Helical Valley function.
///
/// ```math
/// f(x,y,z) = 100 \left[ (z - 10\theta(x, y))^2 + \left(\sqrt{x^2 + y^2} - 1\right)^2\right] + z^2
/// ```
/// where
/// $`2\pi\theta(x,y) = \arctan(y/x) + \pi\Theta(x)`$, $`\Theta(x)`$ is the Heaviside function, and
/// $`-100 \leq x,y,z \leq 100`$. This function has a global minimum at $`f(1,0,0) = 0`$.
pub struct PowellFletcher;
impl Function<f64, Infallible> for PowellFletcher {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
        let theta = if x[0] > 0.0 {
            f64::atan(x[1] / x[0]) / (2.0 * PI)
        } else {
            (PI + f64::atan(x[1] / x[0])) / (2.0 * PI)
        };
        Ok(100.0 * (x[2] - 10.0 * theta).powi(2)
            + (f64::sqrt(x[0] * x[0] + x[1] * x[1]) - 1.0).powi(2)
            + x[2].powi(2))
    }
}
