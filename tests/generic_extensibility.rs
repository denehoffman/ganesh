use ganesh::algorithms::gradient_free::NelderMead;
use ganesh::traits::{Algorithm, CostFunction};
use ganesh::{LinearAlgebra, RealScalar, Vector};
use std::convert::Infallible;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct Wrapped(f64);

impl fmt::Display for Wrapped {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

macro_rules! binary_op {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl $trait for Wrapped {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self { Self(self.0 $operator rhs.0) }
        }
    };
}

binary_op!(Add, add, +);
binary_op!(Sub, sub, -);
binary_op!(Mul, mul, *);
binary_op!(Div, div, /);

impl Neg for Wrapped {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl RealScalar for Wrapped {
    fn zero() -> Self {
        Self(0.0)
    }
    fn one() -> Self {
        Self(1.0)
    }
    fn infinity() -> Self {
        Self(f64::INFINITY)
    }
    fn epsilon() -> Self {
        Self(f64::EPSILON)
    }
    fn literal(value: f64) -> Self {
        Self(value)
    }
    fn to_f64(self) -> Option<f64> {
        Some(self.0)
    }
    fn abs(self) -> Self {
        Self(self.0.abs())
    }
    fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }
    fn cbrt(self) -> Self {
        Self(self.0.cbrt())
    }
    fn powi(self, exponent: i32) -> Self {
        Self(self.0.powi(exponent))
    }
    fn exp(self) -> Self {
        Self(self.0.exp())
    }
    fn ln(self) -> Self {
        Self(self.0.ln())
    }
    fn cos(self) -> Self {
        Self(self.0.cos())
    }
    fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        Self(self.0.mul_add(multiplier.0, addend.0))
    }
    fn is_finite(self) -> bool {
        self.0.is_finite()
    }
    fn is_nan(self) -> bool {
        self.0.is_nan()
    }
    fn total_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct DenseMatrix {
    rows: usize,
    cols: usize,
    values: Vec<Wrapped>,
}

#[derive(Clone, Copy, Debug)]
struct VecProvider;

impl LinearAlgebra<Wrapped> for VecProvider {
    type VectorStorage = Vec<Wrapped>;
    type MatrixStorage = DenseMatrix;

    fn vector_zeros(len: usize) -> Vec<Wrapped> {
        vec![Wrapped::zero(); len]
    }
    fn vector_from_vec(values: Vec<Wrapped>) -> Vec<Wrapped> {
        values
    }
    fn vector_len(value: &Vec<Wrapped>) -> usize {
        value.len()
    }
    fn vector_get(value: &Vec<Wrapped>, index: usize) -> Wrapped {
        value[index]
    }
    fn vector_set(value: &mut Vec<Wrapped>, index: usize, entry: Wrapped) {
        value[index] = entry;
    }
    fn vector_to_vec(value: &Vec<Wrapped>) -> Vec<Wrapped> {
        value.clone()
    }
    fn vec_add(lhs: &Vec<Wrapped>, rhs: &Vec<Wrapped>) -> Vec<Wrapped> {
        lhs.iter().zip(rhs).map(|(a, b)| *a + *b).collect()
    }
    fn vec_sub(lhs: &Vec<Wrapped>, rhs: &Vec<Wrapped>) -> Vec<Wrapped> {
        lhs.iter().zip(rhs).map(|(a, b)| *a - *b).collect()
    }
    fn vec_neg(value: &Vec<Wrapped>) -> Vec<Wrapped> {
        value.iter().map(|x| -*x).collect()
    }
    fn vec_scale(value: &Vec<Wrapped>, alpha: Wrapped) -> Vec<Wrapped> {
        value.iter().map(|x| *x * alpha).collect()
    }
    fn vec_dot(lhs: &Vec<Wrapped>, rhs: &Vec<Wrapped>) -> Wrapped {
        lhs.iter()
            .zip(rhs)
            .fold(Wrapped::zero(), |sum, (a, b)| sum + *a * *b)
    }
    fn vec_norm(value: &Vec<Wrapped>) -> Wrapped {
        Self::vec_dot(value, value).sqrt()
    }
    fn vec_all_finite(value: &Vec<Wrapped>) -> bool {
        value.iter().all(|x| x.is_finite())
    }

    fn matrix_zeros(rows: usize, cols: usize) -> DenseMatrix {
        DenseMatrix {
            rows,
            cols,
            values: vec![Wrapped::zero(); rows * cols],
        }
    }
    fn matrix_identity(n: usize) -> DenseMatrix {
        let mut matrix = Self::matrix_zeros(n, n);
        for i in 0..n {
            matrix.values[i * n + i] = Wrapped::one();
        }
        matrix
    }
    fn matrix_from_vec(rows: usize, cols: usize, values: Vec<Wrapped>) -> DenseMatrix {
        DenseMatrix { rows, cols, values }
    }
    fn matrix_rows(value: &DenseMatrix) -> usize {
        value.rows
    }
    fn matrix_cols(value: &DenseMatrix) -> usize {
        value.cols
    }
    fn matrix_get(value: &DenseMatrix, row: usize, col: usize) -> Wrapped {
        value.values[row * value.cols + col]
    }
    fn matrix_set(value: &mut DenseMatrix, row: usize, col: usize, entry: Wrapped) {
        value.values[row * value.cols + col] = entry;
    }
    fn mat_add(lhs: &DenseMatrix, rhs: &DenseMatrix) -> DenseMatrix {
        Self::matrix_from_vec(lhs.rows, lhs.cols, Self::vec_add(&lhs.values, &rhs.values))
    }
    fn mat_sub(lhs: &DenseMatrix, rhs: &DenseMatrix) -> DenseMatrix {
        Self::matrix_from_vec(lhs.rows, lhs.cols, Self::vec_sub(&lhs.values, &rhs.values))
    }
    fn mat_scale(value: &DenseMatrix, alpha: Wrapped) -> DenseMatrix {
        Self::matrix_from_vec(
            value.rows,
            value.cols,
            Self::vec_scale(&value.values, alpha),
        )
    }
    fn mat_transpose(value: &DenseMatrix) -> DenseMatrix {
        let mut result = Self::matrix_zeros(value.cols, value.rows);
        for row in 0..value.rows {
            for col in 0..value.cols {
                Self::matrix_set(&mut result, col, row, Self::matrix_get(value, row, col));
            }
        }
        result
    }
    fn mat_mul(lhs: &DenseMatrix, rhs: &DenseMatrix) -> DenseMatrix {
        let mut result = Self::matrix_zeros(lhs.rows, rhs.cols);
        for row in 0..lhs.rows {
            for col in 0..rhs.cols {
                let entry = (0..lhs.cols).fold(Wrapped::zero(), |sum, inner| {
                    sum + Self::matrix_get(lhs, row, inner) * Self::matrix_get(rhs, inner, col)
                });
                Self::matrix_set(&mut result, row, col, entry);
            }
        }
        result
    }
    fn mat_vec(matrix: &DenseMatrix, vector: &Vec<Wrapped>) -> Vec<Wrapped> {
        (0..matrix.rows)
            .map(|row| {
                (0..matrix.cols).fold(Wrapped::zero(), |sum, col| {
                    sum + Self::matrix_get(matrix, row, col) * vector[col]
                })
            })
            .collect()
    }
}

struct Quadratic;

impl CostFunction<Wrapped, VecProvider> for Quadratic {
    fn evaluate(&self, x: &Vector<Wrapped, VecProvider>, _: &()) -> Result<Wrapped, Infallible> {
        Ok(x.dot(x))
    }
}

#[test]
fn downstream_scalar_and_provider_run_public_algorithm() {
    let result = NelderMead::<Wrapped, VecProvider>::default()
        .process_default(
            &Quadratic,
            &(),
            Vector::from_vec(vec![Wrapped(2.0), Wrapped(-1.0)]),
        )
        .unwrap();

    assert!(result.fx.0 < 1.0e-3, "unexpected objective: {}", result.fx);
}
