use std::{convert::Infallible, f64::consts::PI};

use crate::Function;

pub struct Rastrigin {
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
pub struct Sphere {
    pub n: usize,
}
impl Function<f64, Infallible> for Sphere {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
        Ok((0..self.n).map(|i| x[i].powi(2)).sum())
    }
}
pub struct Rosenbrock {
    pub n: usize,
}
impl Function<f64, Infallible> for Rosenbrock {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
        Ok((0..(self.n - 1))
            .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
            .sum())
    }
}

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

pub struct Bukin6;
impl Function<f64, Infallible> for Bukin6 {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
        Ok(100.0 * f64::sqrt(f64::abs(x[1] - 0.01 * x[0].powi(2))) + 0.01 * f64::abs(x[0] + 10.0))
    }
}

pub struct Powell;
impl Function<f64, Infallible> for Powell {
    fn evaluate(&self, x: &[f64]) -> Result<f64, Infallible> {
        Ok((x[0] + 10.0 * x[1]).powi(2)
            + 5.0 * (x[2] - x[3]).powi(2)
            + (x[1] - 2.0 * x[2]).powi(4)
            + 10.0 * (x[0] - x[3]).powi(4))
    }
}

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
