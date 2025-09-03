use parking_lot::Mutex;
use std::{convert::Infallible, ops::ControlFlow, sync::Arc};

/// A trait for all kinds of callbacks used in [`Algorithm`](`crate::traits::Algorithm`)s.
///
/// These can be implemented for different kinds of [`Algorithm`](`crate::traits::Algorithm`)s (`A`), problems (`P`), [`Status`](`crate::traits::Status`)es (`S`), and user data (`U`). These are the least restrictive of all callbacks, able to mutate any of their inputs aside from the current step.
pub trait Callback<A, P, S, U = (), E = Infallible> {
    /// A callback method which is called on each step of an [`Algorithm`](`crate::traits::Algorithm`).
    fn callback(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &mut P,
        status: &mut S,
        user_data: &mut U,
    ) -> ControlFlow<()>;
}

impl<C, A, P, S, U, E> Callback<A, P, S, U, E> for Arc<Mutex<C>>
where
    C: Callback<A, P, S, U, E>,
{
    fn callback(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &mut P,
        status: &mut S,
        user_data: &mut U,
    ) -> ControlFlow<()> {
        self.lock()
            .callback(current_step, algorithm, problem, status, user_data)
    }
}

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
        user_data: &U,
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
        user_data: &U,
    ) -> ControlFlow<()> {
        self.lock()
            .check_for_termination(current_step, algorithm, problem, status, user_data)
    }
}

/// A trait for all kinds of observers used in [`Algorithm`](`crate::traits::Algorithm`)s.
///
/// These can be implemented for different kinds of [`Algorithm`](`crate::traits::Algorithm`)s (`A`), problems (`P`), [`Status`](`crate::traits::Status`)es (`S`), and user data (`U`). This is the most restrictive type of callback and is not able to mutate any of its inputs aside from itself.
pub trait Observer<A, P, S, U, E> {
    /// An observation method which is called on each step of an [`Algorithm`](`crate::traits::Algorithm`).
    fn observe(
        &mut self,
        current_step: usize,
        algorithm: &A,
        problem: &P,
        status: &S,
        user_data: &U,
    );
}

impl<O, A, P, S, U, E> Observer<A, P, S, U, E> for Arc<Mutex<O>>
where
    O: Observer<A, P, S, U, E>,
{
    fn observe(
        &mut self,
        current_step: usize,
        algorithm: &A,
        problem: &P,
        status: &S,
        user_data: &U,
    ) {
        self.lock()
            .observe(current_step, algorithm, problem, status, user_data)
    }
}
