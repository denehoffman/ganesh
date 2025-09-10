use parking_lot::Mutex;
use std::{ops::ControlFlow, sync::Arc};

/// A trait for all kinds of terminators used in [`Algorithm`](`crate::traits::Algorithm`)s.
///
/// These can be implemented for different kinds of [`Algorithm`](`crate::traits::Algorithm`)s (`A`), problems (`P`), [`Status`](`crate::traits::Status`)es (`S`), and user data (`U`). This is the second least-restrictive type of callback, able to mutate both the [`Algorithm`](`crate::traits::Algorithm`) and its [`Status`](`crate::traits::Status`).
pub trait Terminator<A, P, S, U, E> {
    /// A termination check which is called on each step of an [`Algorithm`](`crate::traits::Algorithm`).
    fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        args: &U,
    ) -> ControlFlow<()>;
}

impl<T, A, P, S, U, E> Terminator<A, P, S, U, E> for Arc<Mutex<T>>
where
    T: Terminator<A, P, S, U, E>,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        args: &U,
    ) -> ControlFlow<()> {
        self.lock()
            .check_for_termination(current_step, algorithm, problem, status, args)
    }
}

/// A trait for all kinds of observers used in [`Algorithm`](`crate::traits::Algorithm`)s.
///
/// These can be implemented for different kinds of [`Algorithm`](`crate::traits::Algorithm`)s (`A`), problems (`P`), [`Status`](`crate::traits::Status`)es (`S`), and user data (`U`). This is the most restrictive type of callback and is not able to mutate any of its inputs aside from itself.
pub trait Observer<A, P, S, U, E> {
    /// An observation method which is called on each step of an [`Algorithm`](`crate::traits::Algorithm`).
    fn observe(&mut self, current_step: usize, algorithm: &A, problem: &P, status: &S, args: &U);
}

impl<O, A, P, S, U, E> Observer<A, P, S, U, E> for Arc<Mutex<O>>
where
    O: Observer<A, P, S, U, E>,
{
    fn observe(&mut self, current_step: usize, algorithm: &A, problem: &P, status: &S, args: &U) {
        self.lock()
            .observe(current_step, algorithm, problem, status, args)
    }
}
