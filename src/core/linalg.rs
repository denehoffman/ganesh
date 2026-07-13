//! Linear algebra backend wrappers and implementations.

use crate::core::RealScalar;
use nalgebra::{DMatrix, DVector, LU};
#[cfg(feature = "backend-ndarray")]
use ndarray::{Array1, Array2};
#[cfg(feature = "backend-ndarray")]
use ndarray_linalg::{Determinant as NdDeterminant, Eigh, Inverse, Lapack, Solve, UPLO};
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Backend marker and operation surface for dense vector and matrix storage.
pub trait LinearAlgebra<T: RealScalar>: Clone + Debug + Send + Sync + 'static {
    /// Owned vector storage used by this backend.
    type VectorStorage: Clone + Debug + PartialEq + Send + Sync + 'static;
    /// Owned matrix storage used by this backend.
    type MatrixStorage: Clone + Debug + PartialEq + Send + Sync + 'static;

    /// Create a zero vector.
    fn vector_zeros(len: usize) -> Self::VectorStorage;
    /// Create a vector from owned values.
    fn vector_from_vec(values: Vec<T>) -> Self::VectorStorage;
    /// Return vector length.
    fn vector_len(value: &Self::VectorStorage) -> usize;
    /// Get one vector element.
    fn vector_get(value: &Self::VectorStorage, index: usize) -> T;
    /// Set one vector element.
    fn vector_set(value: &mut Self::VectorStorage, index: usize, entry: T);
    /// Return vector entries as owned values.
    fn vector_to_vec(value: &Self::VectorStorage) -> Vec<T>;

    /// Vector addition.
    fn vec_add(lhs: &Self::VectorStorage, rhs: &Self::VectorStorage) -> Self::VectorStorage;
    /// Vector subtraction.
    fn vec_sub(lhs: &Self::VectorStorage, rhs: &Self::VectorStorage) -> Self::VectorStorage;
    /// Vector negation.
    fn vec_neg(value: &Self::VectorStorage) -> Self::VectorStorage;
    /// Vector scaling.
    fn vec_scale(value: &Self::VectorStorage, alpha: T) -> Self::VectorStorage;
    /// Dot product.
    fn vec_dot(lhs: &Self::VectorStorage, rhs: &Self::VectorStorage) -> T;
    /// Euclidean norm.
    fn vec_norm(value: &Self::VectorStorage) -> T;
    /// Return `true` when all vector entries are finite.
    fn vec_all_finite(value: &Self::VectorStorage) -> bool;

    /// Create a zero matrix.
    fn matrix_zeros(rows: usize, cols: usize) -> Self::MatrixStorage;
    /// Create an identity matrix.
    fn matrix_identity(n: usize) -> Self::MatrixStorage;
    /// Create a matrix from row-major values.
    fn matrix_from_vec(rows: usize, cols: usize, values: Vec<T>) -> Self::MatrixStorage;
    /// Return row count.
    fn matrix_rows(value: &Self::MatrixStorage) -> usize;
    /// Return column count.
    fn matrix_cols(value: &Self::MatrixStorage) -> usize;
    /// Get one matrix element.
    fn matrix_get(value: &Self::MatrixStorage, row: usize, col: usize) -> T;
    /// Set one matrix element.
    fn matrix_set(value: &mut Self::MatrixStorage, row: usize, col: usize, entry: T);

    /// Matrix addition.
    fn mat_add(lhs: &Self::MatrixStorage, rhs: &Self::MatrixStorage) -> Self::MatrixStorage;
    /// Matrix subtraction.
    fn mat_sub(lhs: &Self::MatrixStorage, rhs: &Self::MatrixStorage) -> Self::MatrixStorage;
    /// Matrix scaling.
    fn mat_scale(value: &Self::MatrixStorage, alpha: T) -> Self::MatrixStorage;
    /// Matrix transpose.
    fn mat_transpose(value: &Self::MatrixStorage) -> Self::MatrixStorage;
    /// Matrix-matrix product.
    fn mat_mul(lhs: &Self::MatrixStorage, rhs: &Self::MatrixStorage) -> Self::MatrixStorage;
    /// Matrix-vector product.
    fn mat_vec(matrix: &Self::MatrixStorage, vector: &Self::VectorStorage) -> Self::VectorStorage;
}

/// Backend capability for solving dense linear systems and computing inverses.
pub trait LinearSolve<T: RealScalar>: LinearAlgebra<T> {
    /// Solve `matrix * x = rhs` using an LU-style solve.
    fn lu_solve(
        matrix: &Self::MatrixStorage,
        rhs: &Self::VectorStorage,
    ) -> Option<Self::VectorStorage>;
    /// Solve `matrix * x = rhs` for matrix right-hand sides using an LU-style solve.
    fn lu_solve_matrix(
        matrix: &Self::MatrixStorage,
        rhs: &Self::MatrixStorage,
    ) -> Option<Self::MatrixStorage>;
    /// Compute the inverse using an LU-style operation.
    fn lu_inverse(matrix: &Self::MatrixStorage) -> Option<Self::MatrixStorage>;
}

/// Backend capability for computing a matrix pseudoinverse.
pub trait PseudoInverse<T: RealScalar>: LinearAlgebra<T> {
    /// Compute the pseudoinverse.
    fn pseudo_inverse(matrix: &Self::MatrixStorage, epsilon: T) -> Option<Self::MatrixStorage>;
}

/// Backend capability for computing a determinant.
pub trait Determinant<T: RealScalar>: LinearAlgebra<T> {
    /// Compute the determinant.
    fn determinant(matrix: &Self::MatrixStorage) -> Option<T>;
}

/// Backend capability for symmetric eigendecomposition.
pub trait SymmetricEigen<T: RealScalar>: LinearAlgebra<T> {
    /// Return eigenvalues and column-oriented eigenvectors of a symmetric matrix.
    fn symmetric_eigen(
        matrix: &Self::MatrixStorage,
    ) -> Option<(Self::VectorStorage, Self::MatrixStorage)>;
}

/// Scalar wrapper for future public APIs that want a crate-owned scalar type.
#[repr(transparent)]
pub struct Scalar<T: RealScalar>(pub T);

impl<T: RealScalar> Clone for Scalar<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: RealScalar> Copy for Scalar<T> {}

impl<T: RealScalar> Debug for Scalar<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl<T: RealScalar> PartialEq for Scalar<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: RealScalar> Add for Scalar<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<T: RealScalar> Sub for Scalar<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<T: RealScalar> Mul for Scalar<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<T: RealScalar> Div for Scalar<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl<T: RealScalar> Neg for Scalar<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

/// Owned dense vector wrapper parameterized by scalar and backend.
pub struct Vector<T: RealScalar, B: LinearAlgebra<T> = NalgebraBackend> {
    storage: B::VectorStorage,
    _marker: PhantomData<(T, B)>,
}

impl<T: RealScalar, B: LinearAlgebra<T>> Vector<T, B> {
    /// Wrap backend vector storage.
    pub const fn from_storage(storage: B::VectorStorage) -> Self {
        Self {
            storage,
            _marker: PhantomData,
        }
    }

    /// Consume the wrapper and return backend vector storage.
    pub fn into_storage(self) -> B::VectorStorage {
        self.storage
    }

    /// Borrow backend vector storage.
    pub const fn as_storage(&self) -> &B::VectorStorage {
        &self.storage
    }

    /// Mutably borrow backend vector storage.
    pub fn as_storage_mut(&mut self) -> &mut B::VectorStorage {
        &mut self.storage
    }

    /// Create a zero vector.
    pub fn zeros(len: usize) -> Self {
        Self::from_storage(B::vector_zeros(len))
    }

    /// Create a vector from owned values.
    pub fn from_vec(values: Vec<T>) -> Self {
        Self::from_storage(B::vector_from_vec(values))
    }

    /// Return vector length.
    pub fn len(&self) -> usize {
        B::vector_len(&self.storage)
    }

    /// Return `true` when the vector has no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get one element.
    pub fn get(&self, index: usize) -> T {
        B::vector_get(&self.storage, index)
    }

    /// Set one element.
    pub fn set(&mut self, index: usize, entry: T) {
        B::vector_set(&mut self.storage, index, entry);
    }

    /// Return entries as owned values.
    pub fn to_vec(&self) -> Vec<T> {
        B::vector_to_vec(&self.storage)
    }

    /// Dot product.
    pub fn dot(&self, rhs: &Self) -> T {
        B::vec_dot(&self.storage, &rhs.storage)
    }

    /// Euclidean norm.
    pub fn norm(&self) -> T {
        B::vec_norm(&self.storage)
    }

    /// Return `self * alpha`.
    pub fn scale(&self, alpha: T) -> Self {
        Self::from_storage(B::vec_scale(&self.storage, alpha))
    }

    /// Return `self + rhs`.
    pub fn add(&self, rhs: &Self) -> Self {
        self + rhs
    }

    /// Return `self - rhs`.
    pub fn sub(&self, rhs: &Self) -> Self {
        self - rhs
    }

    /// Return `-self`.
    pub fn neg(&self) -> Self {
        -self
    }

    /// Return `self + rhs * alpha`.
    pub fn add_scaled(&self, rhs: &Self, alpha: T) -> Self {
        self + &rhs.scale(alpha)
    }

    /// Return `true` when all entries are finite.
    pub fn all_finite(&self) -> bool {
        B::vec_all_finite(&self.storage)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Clone for Vector<T, B> {
    fn clone(&self) -> Self {
        Self::from_storage(self.storage.clone())
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Debug for Vector<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.storage.fmt(f)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> PartialEq for Vector<T, B> {
    fn eq(&self, other: &Self) -> bool {
        self.storage == other.storage
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Add<&Vector<T, B>> for &Vector<T, B> {
    type Output = Vector<T, B>;

    fn add(self, rhs: &Vector<T, B>) -> Self::Output {
        Vector::from_storage(B::vec_add(&self.storage, &rhs.storage))
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Add<&Self> for Vector<T, B> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Add<Vector<T, B>> for &Vector<T, B> {
    type Output = Vector<T, B>;

    fn add(self, rhs: Vector<T, B>) -> Self::Output {
        self + &rhs
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Add for Vector<T, B> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Sub<&Vector<T, B>> for &Vector<T, B> {
    type Output = Vector<T, B>;

    fn sub(self, rhs: &Vector<T, B>) -> Self::Output {
        Vector::from_storage(B::vec_sub(&self.storage, &rhs.storage))
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Sub<&Self> for Vector<T, B> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Sub<Vector<T, B>> for &Vector<T, B> {
    type Output = Vector<T, B>;

    fn sub(self, rhs: Vector<T, B>) -> Self::Output {
        self - &rhs
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Sub for Vector<T, B> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Neg for &Vector<T, B> {
    type Output = Vector<T, B>;

    fn neg(self) -> Self::Output {
        Vector::from_storage(B::vec_neg(&self.storage))
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Neg for Vector<T, B> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Mul<T> for &Vector<T, B> {
    type Output = Vector<T, B>;

    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Mul<T> for Vector<T, B> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Div<T> for &Vector<T, B> {
    type Output = Vector<T, B>;

    fn div(self, rhs: T) -> Self::Output {
        self.scale(T::one() / rhs)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Div<T> for Vector<T, B> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        self.scale(T::one() / rhs)
    }
}

/// Owned dense matrix wrapper parameterized by scalar and backend.
pub struct Matrix<T: RealScalar, B: LinearAlgebra<T> = NalgebraBackend> {
    storage: B::MatrixStorage,
    _marker: PhantomData<(T, B)>,
}

impl<T: RealScalar, B: LinearAlgebra<T>> Matrix<T, B> {
    /// Wrap backend matrix storage.
    pub const fn from_storage(storage: B::MatrixStorage) -> Self {
        Self {
            storage,
            _marker: PhantomData,
        }
    }

    /// Consume the wrapper and return backend matrix storage.
    pub fn into_storage(self) -> B::MatrixStorage {
        self.storage
    }

    /// Borrow backend matrix storage.
    pub const fn as_storage(&self) -> &B::MatrixStorage {
        &self.storage
    }

    /// Mutably borrow backend matrix storage.
    pub fn as_storage_mut(&mut self) -> &mut B::MatrixStorage {
        &mut self.storage
    }

    /// Create a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::from_storage(B::matrix_zeros(rows, cols))
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        Self::from_storage(B::matrix_identity(n))
    }

    /// Create a matrix from row-major values.
    pub fn from_vec(rows: usize, cols: usize, values: Vec<T>) -> Self {
        Self::from_storage(B::matrix_from_vec(rows, cols, values))
    }

    /// Return row count.
    pub fn rows(&self) -> usize {
        B::matrix_rows(&self.storage)
    }

    /// Return column count.
    pub fn cols(&self) -> usize {
        B::matrix_cols(&self.storage)
    }

    /// Get one element.
    pub fn get(&self, row: usize, col: usize) -> T {
        B::matrix_get(&self.storage, row, col)
    }

    /// Set one element.
    pub fn set(&mut self, row: usize, col: usize, entry: T) {
        B::matrix_set(&mut self.storage, row, col, entry);
    }

    /// Return `self * alpha`.
    pub fn scale(&self, alpha: T) -> Self {
        Self::from_storage(B::mat_scale(&self.storage, alpha))
    }

    /// Return the transpose.
    pub fn transpose(&self) -> Self {
        Self::from_storage(B::mat_transpose(&self.storage))
    }

    /// Matrix-vector product.
    pub fn mul_vec(&self, rhs: &Vector<T, B>) -> Vector<T, B> {
        Vector::from_storage(B::mat_vec(&self.storage, &rhs.storage))
    }

    /// Matrix-matrix product.
    pub fn mul_mat(&self, rhs: &Self) -> Self {
        Self::from_storage(B::mat_mul(&self.storage, &rhs.storage))
    }
}

impl<T, B> Matrix<T, B>
where
    T: RealScalar,
    B: LinearSolve<T>,
{
    /// Solve `self * x = rhs` using an LU-style solve.
    pub fn lu_solve(&self, rhs: &Vector<T, B>) -> Option<Vector<T, B>> {
        B::lu_solve(&self.storage, &rhs.storage).map(Vector::from_storage)
    }

    /// Solve `self * x = rhs` with a matrix right-hand side using an LU-style solve.
    pub fn lu_solve_matrix(&self, rhs: &Self) -> Option<Self> {
        B::lu_solve_matrix(&self.storage, &rhs.storage).map(Self::from_storage)
    }

    /// Compute the inverse using an LU-style operation.
    pub fn lu_inverse(&self) -> Option<Self> {
        B::lu_inverse(&self.storage).map(Self::from_storage)
    }
}

impl<T, B> Matrix<T, B>
where
    T: RealScalar,
    B: SymmetricEigen<T>,
{
    /// Compute eigenvalues and column-oriented eigenvectors of a symmetric matrix.
    pub fn symmetric_eigen(&self) -> Option<(Vector<T, B>, Self)> {
        B::symmetric_eigen(&self.storage)
            .map(|(values, vectors)| (Vector::from_storage(values), Self::from_storage(vectors)))
    }
}

impl<T, B> Matrix<T, B>
where
    T: RealScalar,
    B: PseudoInverse<T>,
{
    /// Compute the pseudoinverse.
    pub fn pseudo_inverse(&self, epsilon: T) -> Option<Self> {
        B::pseudo_inverse(&self.storage, epsilon).map(Self::from_storage)
    }
}

impl<T, B> Matrix<T, B>
where
    T: RealScalar,
    B: Determinant<T>,
{
    /// Compute the determinant.
    pub fn determinant(&self) -> Option<T> {
        B::determinant(&self.storage)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Clone for Matrix<T, B> {
    fn clone(&self) -> Self {
        Self::from_storage(self.storage.clone())
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Debug for Matrix<T, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.storage.fmt(f)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> PartialEq for Matrix<T, B> {
    fn eq(&self, other: &Self) -> bool {
        self.storage == other.storage
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Add<&Matrix<T, B>> for &Matrix<T, B> {
    type Output = Matrix<T, B>;

    fn add(self, rhs: &Matrix<T, B>) -> Self::Output {
        Matrix::from_storage(B::mat_add(&self.storage, &rhs.storage))
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Add for Matrix<T, B> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Sub<&Matrix<T, B>> for &Matrix<T, B> {
    type Output = Matrix<T, B>;

    fn sub(self, rhs: &Matrix<T, B>) -> Self::Output {
        Matrix::from_storage(B::mat_sub(&self.storage, &rhs.storage))
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Sub for Matrix<T, B> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Mul<T> for &Matrix<T, B> {
    type Output = Matrix<T, B>;

    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Mul<T> for Matrix<T, B> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        self.scale(rhs)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Div<T> for &Matrix<T, B> {
    type Output = Matrix<T, B>;

    fn div(self, rhs: T) -> Self::Output {
        self.scale(T::one() / rhs)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Div<T> for Matrix<T, B> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        self.scale(T::one() / rhs)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Mul<&Vector<T, B>> for &Matrix<T, B> {
    type Output = Vector<T, B>;

    fn mul(self, rhs: &Vector<T, B>) -> Self::Output {
        self.mul_vec(rhs)
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Mul<&Matrix<T, B>> for &Matrix<T, B> {
    type Output = Matrix<T, B>;

    fn mul(self, rhs: &Matrix<T, B>) -> Self::Output {
        self.mul_mat(rhs)
    }
}

/// Nalgebra dynamic vector/matrix backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NalgebraBackend;

impl<T: RealScalar + nalgebra::RealField> LinearAlgebra<T> for NalgebraBackend {
    type VectorStorage = DVector<T>;
    type MatrixStorage = DMatrix<T>;

    fn vector_zeros(len: usize) -> Self::VectorStorage {
        DVector::from_element(len, <T as RealScalar>::zero())
    }

    fn vector_from_vec(values: Vec<T>) -> Self::VectorStorage {
        DVector::from_vec(values)
    }

    fn vector_len(value: &Self::VectorStorage) -> usize {
        value.len()
    }

    fn vector_get(value: &Self::VectorStorage, index: usize) -> T {
        value[index]
    }

    fn vector_set(value: &mut Self::VectorStorage, index: usize, entry: T) {
        value[index] = entry;
    }

    fn vector_to_vec(value: &Self::VectorStorage) -> Vec<T> {
        value.iter().copied().collect()
    }

    fn vec_add(lhs: &Self::VectorStorage, rhs: &Self::VectorStorage) -> Self::VectorStorage {
        lhs + rhs
    }

    fn vec_sub(lhs: &Self::VectorStorage, rhs: &Self::VectorStorage) -> Self::VectorStorage {
        lhs - rhs
    }

    fn vec_neg(value: &Self::VectorStorage) -> Self::VectorStorage {
        -value
    }

    fn vec_scale(value: &Self::VectorStorage, alpha: T) -> Self::VectorStorage {
        value * alpha
    }

    fn vec_dot(lhs: &Self::VectorStorage, rhs: &Self::VectorStorage) -> T {
        lhs.dot(rhs)
    }

    fn vec_norm(value: &Self::VectorStorage) -> T {
        value.norm()
    }

    fn vec_all_finite(value: &Self::VectorStorage) -> bool {
        value.iter().all(|entry| entry.is_finite())
    }

    fn matrix_zeros(rows: usize, cols: usize) -> Self::MatrixStorage {
        DMatrix::from_element(rows, cols, <T as RealScalar>::zero())
    }

    fn matrix_identity(n: usize) -> Self::MatrixStorage {
        DMatrix::identity(n, n)
    }

    fn matrix_from_vec(rows: usize, cols: usize, values: Vec<T>) -> Self::MatrixStorage {
        DMatrix::from_row_slice(rows, cols, &values)
    }

    fn matrix_rows(value: &Self::MatrixStorage) -> usize {
        value.nrows()
    }

    fn matrix_cols(value: &Self::MatrixStorage) -> usize {
        value.ncols()
    }

    fn matrix_get(value: &Self::MatrixStorage, row: usize, col: usize) -> T {
        value[(row, col)]
    }

    fn matrix_set(value: &mut Self::MatrixStorage, row: usize, col: usize, entry: T) {
        value[(row, col)] = entry;
    }

    fn mat_add(lhs: &Self::MatrixStorage, rhs: &Self::MatrixStorage) -> Self::MatrixStorage {
        lhs + rhs
    }

    fn mat_sub(lhs: &Self::MatrixStorage, rhs: &Self::MatrixStorage) -> Self::MatrixStorage {
        lhs - rhs
    }

    fn mat_scale(value: &Self::MatrixStorage, alpha: T) -> Self::MatrixStorage {
        value * alpha
    }

    fn mat_transpose(value: &Self::MatrixStorage) -> Self::MatrixStorage {
        value.transpose()
    }

    fn mat_mul(lhs: &Self::MatrixStorage, rhs: &Self::MatrixStorage) -> Self::MatrixStorage {
        lhs * rhs
    }

    fn mat_vec(matrix: &Self::MatrixStorage, vector: &Self::VectorStorage) -> Self::VectorStorage {
        matrix * vector
    }
}

impl<T: RealScalar + nalgebra::RealField> LinearSolve<T> for NalgebraBackend {
    fn lu_solve(
        matrix: &Self::MatrixStorage,
        rhs: &Self::VectorStorage,
    ) -> Option<Self::VectorStorage> {
        LU::new(matrix.clone()).solve(rhs)
    }

    fn lu_solve_matrix(
        matrix: &Self::MatrixStorage,
        rhs: &Self::MatrixStorage,
    ) -> Option<Self::MatrixStorage> {
        LU::new(matrix.clone()).solve(rhs)
    }

    fn lu_inverse(matrix: &Self::MatrixStorage) -> Option<Self::MatrixStorage> {
        LU::new(matrix.clone()).try_inverse()
    }
}

impl<T: RealScalar + nalgebra::RealField> PseudoInverse<T> for NalgebraBackend {
    fn pseudo_inverse(matrix: &Self::MatrixStorage, epsilon: T) -> Option<Self::MatrixStorage> {
        matrix.clone().pseudo_inverse(epsilon).ok()
    }
}

impl<T: RealScalar + nalgebra::RealField> Determinant<T> for NalgebraBackend {
    fn determinant(matrix: &Self::MatrixStorage) -> Option<T> {
        matrix.is_square().then(|| matrix.clone().determinant())
    }
}

impl<T: RealScalar + nalgebra::RealField> SymmetricEigen<T> for NalgebraBackend {
    fn symmetric_eigen(
        matrix: &Self::MatrixStorage,
    ) -> Option<(Self::VectorStorage, Self::MatrixStorage)> {
        if !matrix.is_square() {
            return None;
        }
        let decomposition = nalgebra::linalg::SymmetricEigen::new(matrix.clone());
        Some((decomposition.eigenvalues, decomposition.eigenvectors))
    }
}

/// Ndarray `Array1`/`Array2` backend.
#[cfg(feature = "backend-ndarray")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NdArrayBackend;

#[cfg(feature = "backend-ndarray")]
impl<T> LinearAlgebra<T> for NdArrayBackend
where
    T: RealScalar + Lapack<Real = T> + ndarray::ScalarOperand,
{
    type VectorStorage = Array1<T>;
    type MatrixStorage = Array2<T>;

    fn vector_zeros(len: usize) -> Self::VectorStorage {
        Array1::zeros(len)
    }

    fn vector_from_vec(values: Vec<T>) -> Self::VectorStorage {
        Array1::from_vec(values)
    }

    fn vector_len(value: &Self::VectorStorage) -> usize {
        value.len()
    }

    fn vector_get(value: &Self::VectorStorage, index: usize) -> T {
        value[index]
    }

    fn vector_set(value: &mut Self::VectorStorage, index: usize, entry: T) {
        value[index] = entry;
    }

    fn vector_to_vec(value: &Self::VectorStorage) -> Vec<T> {
        value.iter().copied().collect()
    }

    fn vec_add(lhs: &Self::VectorStorage, rhs: &Self::VectorStorage) -> Self::VectorStorage {
        lhs + rhs
    }

    fn vec_sub(lhs: &Self::VectorStorage, rhs: &Self::VectorStorage) -> Self::VectorStorage {
        lhs - rhs
    }

    fn vec_neg(value: &Self::VectorStorage) -> Self::VectorStorage {
        value.mapv(|entry| -entry)
    }

    fn vec_scale(value: &Self::VectorStorage, alpha: T) -> Self::VectorStorage {
        value.mapv(|entry| entry * alpha)
    }

    fn vec_dot(lhs: &Self::VectorStorage, rhs: &Self::VectorStorage) -> T {
        lhs.dot(rhs)
    }

    fn vec_norm(value: &Self::VectorStorage) -> T {
        RealScalar::sqrt(value.dot(value))
    }

    fn vec_all_finite(value: &Self::VectorStorage) -> bool {
        value.iter().all(|entry| entry.is_finite())
    }

    fn matrix_zeros(rows: usize, cols: usize) -> Self::MatrixStorage {
        Array2::zeros((rows, cols))
    }

    fn matrix_identity(n: usize) -> Self::MatrixStorage {
        Array2::eye(n)
    }

    fn matrix_from_vec(rows: usize, cols: usize, values: Vec<T>) -> Self::MatrixStorage {
        Array2::from_shape_vec((rows, cols), values).expect("shape matches row-major values")
    }

    fn matrix_rows(value: &Self::MatrixStorage) -> usize {
        value.nrows()
    }

    fn matrix_cols(value: &Self::MatrixStorage) -> usize {
        value.ncols()
    }

    fn matrix_get(value: &Self::MatrixStorage, row: usize, col: usize) -> T {
        value[(row, col)]
    }

    fn matrix_set(value: &mut Self::MatrixStorage, row: usize, col: usize, entry: T) {
        value[(row, col)] = entry;
    }

    fn mat_add(lhs: &Self::MatrixStorage, rhs: &Self::MatrixStorage) -> Self::MatrixStorage {
        lhs + rhs
    }

    fn mat_sub(lhs: &Self::MatrixStorage, rhs: &Self::MatrixStorage) -> Self::MatrixStorage {
        lhs - rhs
    }

    fn mat_scale(value: &Self::MatrixStorage, alpha: T) -> Self::MatrixStorage {
        value.mapv(|entry| entry * alpha)
    }

    fn mat_transpose(value: &Self::MatrixStorage) -> Self::MatrixStorage {
        value.t().to_owned()
    }

    fn mat_mul(lhs: &Self::MatrixStorage, rhs: &Self::MatrixStorage) -> Self::MatrixStorage {
        lhs.dot(rhs)
    }

    fn mat_vec(matrix: &Self::MatrixStorage, vector: &Self::VectorStorage) -> Self::VectorStorage {
        matrix.dot(vector)
    }
}

#[cfg(feature = "backend-ndarray")]
impl<T> LinearSolve<T> for NdArrayBackend
where
    T: RealScalar + Lapack<Real = T> + ndarray::ScalarOperand,
{
    fn lu_solve(
        matrix: &Self::MatrixStorage,
        rhs: &Self::VectorStorage,
    ) -> Option<Self::VectorStorage> {
        matrix.solve(rhs).ok()
    }

    fn lu_solve_matrix(
        matrix: &Self::MatrixStorage,
        rhs: &Self::MatrixStorage,
    ) -> Option<Self::MatrixStorage> {
        let mut solved = Self::matrix_zeros(matrix.ncols(), rhs.ncols());
        for col in 0..rhs.ncols() {
            let rhs_col = rhs.column(col).to_owned();
            let x_col = matrix.solve(&rhs_col).ok()?;
            solved.column_mut(col).assign(&x_col);
        }
        Some(solved)
    }

    fn lu_inverse(matrix: &Self::MatrixStorage) -> Option<Self::MatrixStorage> {
        matrix.inv().ok()
    }
}

#[cfg(feature = "backend-ndarray")]
impl<T> PseudoInverse<T> for NdArrayBackend
where
    T: RealScalar + Lapack<Real = T> + ndarray::ScalarOperand,
{
    fn pseudo_inverse(_matrix: &Self::MatrixStorage, _epsilon: T) -> Option<Self::MatrixStorage> {
        None
    }
}

#[cfg(feature = "backend-ndarray")]
impl<T> Determinant<T> for NdArrayBackend
where
    T: RealScalar + Lapack<Real = T> + ndarray::ScalarOperand,
{
    fn determinant(matrix: &Self::MatrixStorage) -> Option<T> {
        matrix.det().ok()
    }
}

#[cfg(feature = "backend-ndarray")]
impl<T> SymmetricEigen<T> for NdArrayBackend
where
    T: RealScalar + Lapack<Real = T> + ndarray::ScalarOperand,
{
    fn symmetric_eigen(
        matrix: &Self::MatrixStorage,
    ) -> Option<(Self::VectorStorage, Self::MatrixStorage)> {
        matrix.clone().eigh(UPLO::Lower).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn vector_contract<B>()
    where
        B: LinearAlgebra<f64>,
    {
        let a = Vector::<f64, B>::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::<f64, B>::from_vec(vec![4.0, 5.0, 6.0]);

        assert_eq!(a.len(), 3);
        assert_eq!(a.dot(&b), 32.0);
        assert_relative_eq!(a.norm(), 14.0_f64.sqrt());
        assert_eq!((&a * 2.0).to_vec(), vec![2.0, 4.0, 6.0]);
        assert_eq!((&a + &b).to_vec(), vec![5.0, 7.0, 9.0]);
        assert_eq!((&b - &a).to_vec(), vec![3.0, 3.0, 3.0]);
        assert_eq!((-&a).to_vec(), vec![-1.0, -2.0, -3.0]);
    }

    fn matrix_contract<B>()
    where
        B: LinearAlgebra<f64> + LinearSolve<f64> + Determinant<f64>,
    {
        let mut matrix = Matrix::<f64, B>::identity(2);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);

        let x = Vector::<f64, B>::from_vec(vec![5.0, 7.0]);
        assert_eq!(matrix.mul_vec(&x).to_vec(), vec![19.0, 22.0]);
        assert_eq!(matrix.transpose().get(0, 1), 3.0);

        let a = Matrix::<f64, B>::from_vec(2, 2, vec![3.0, 2.0, 1.0, 2.0]);
        let rhs = Vector::<f64, B>::from_vec(vec![5.0, 5.0]);
        let Some(solved) = a.lu_solve(&rhs) else {
            panic!("matrix should solve");
        };
        assert_relative_eq!(solved.get(0), 0.0);
        assert_relative_eq!(solved.get(1), 2.5);

        let Some(inverse) = a.lu_solve_matrix(&Matrix::identity(2)) else {
            panic!("matrix RHS should solve");
        };
        assert_relative_eq!(inverse.get(0, 0), 0.5);
        assert_relative_eq!(inverse.get(0, 1), -0.5);
        assert_relative_eq!(inverse.get(1, 0), -0.25);
        assert_relative_eq!(inverse.get(1, 1), 0.75);

        let Some(inverse) = a.lu_inverse() else {
            panic!("matrix should invert");
        };
        assert_relative_eq!(inverse.get(0, 0), 0.5);
        assert_relative_eq!(inverse.get(0, 1), -0.5);
        assert_relative_eq!(inverse.get(1, 0), -0.25);
        assert_relative_eq!(inverse.get(1, 1), 0.75);

        let Some(determinant) = a.determinant() else {
            panic!("matrix should have determinant");
        };
        assert_relative_eq!(determinant, 4.0);
    }

    fn nalgebra_pseudo_inverse_contract() {
        let singular = Matrix::<f64, NalgebraBackend>::from_vec(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        let Some(pseudo_inverse) = singular.pseudo_inverse(f64::EPSILON.cbrt()) else {
            panic!("pseudoinverse");
        };
        assert_eq!(pseudo_inverse.rows(), 2);
        assert_eq!(pseudo_inverse.cols(), 2);
    }

    #[test]
    fn nalgebra_backend_satisfies_contracts() {
        vector_contract::<NalgebraBackend>();
        matrix_contract::<NalgebraBackend>();
        nalgebra_pseudo_inverse_contract();
    }

    #[cfg(feature = "backend-ndarray")]
    #[test]
    fn ndarray_backend_satisfies_contracts() {
        vector_contract::<NdArrayBackend>();
        matrix_contract::<NdArrayBackend>();
    }
}
