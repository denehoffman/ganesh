use parking_lot::{Mutex, RwLock};
use std::{ops::ControlFlow, rc::Rc, sync::Arc};

use crate::traits::{Algorithm, Status};

/// A trait for all kinds of terminators used in [`Algorithm`](`crate::traits::Algorithm`)s.
///
/// These can be implemented for different kinds of [`Algorithm`](`crate::traits::Algorithm`)s (`A`), problems (`P`), [`Status`](`crate::traits::Status`)es (`S`), and user data (`U`). This is the second least-restrictive type of callback, able to mutate both the [`Algorithm`](`crate::traits::Algorithm`) and its [`Status`](`crate::traits::Status`).
pub trait Terminator<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    /// A termination check which is called on each step of an [`Algorithm`](`crate::traits::Algorithm`).
    fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        args: &U,
        config: &C,
    ) -> ControlFlow<()>;
}

impl<T, A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for Rc<RwLock<T>>
where
    T: Terminator<A, P, S, U, E, C>,
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        args: &U,
        config: &C,
    ) -> ControlFlow<()> {
        self.write()
            .check_for_termination(current_step, algorithm, problem, status, args, config)
    }
}
impl<T, A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for Rc<Mutex<T>>
where
    T: Terminator<A, P, S, U, E, C>,
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        args: &U,
        config: &C,
    ) -> ControlFlow<()> {
        self.lock()
            .check_for_termination(current_step, algorithm, problem, status, args, config)
    }
}
impl<T, A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for Arc<RwLock<T>>
where
    T: Terminator<A, P, S, U, E, C>,
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        args: &U,
        config: &C,
    ) -> ControlFlow<()> {
        self.write()
            .check_for_termination(current_step, algorithm, problem, status, args, config)
    }
}
impl<T, A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for Arc<Mutex<T>>
where
    T: Terminator<A, P, S, U, E, C>,
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn check_for_termination(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        args: &U,
        config: &C,
    ) -> ControlFlow<()> {
        self.lock()
            .check_for_termination(current_step, algorithm, problem, status, args, config)
    }
}

/// A trait for all kinds of observers used in [`Algorithm`](`crate::traits::Algorithm`)s.
///
/// These can be implemented for different kinds of [`Algorithm`](`crate::traits::Algorithm`)s (`A`), problems (`P`), [`Status`](`crate::traits::Status`)es (`S`), and user data (`U`). This is the most restrictive type of callback and is not able to mutate any of its inputs aside from itself.
pub trait Observer<A, P, S, U, E, C>
where
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    /// An observation method which is called on each step of an [`Algorithm`](`crate::traits::Algorithm`).
    fn observe(
        &mut self,
        current_step: usize,
        algorithm: &A,
        problem: &P,
        status: &S,
        args: &U,
        config: &C,
    );
}

impl<O, A, P, S, U, E, C> Observer<A, P, S, U, E, C> for Rc<Mutex<O>>
where
    O: Observer<A, P, S, U, E, C>,
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn observe(
        &mut self,
        current_step: usize,
        algorithm: &A,
        problem: &P,
        status: &S,
        args: &U,
        config: &C,
    ) {
        self.lock()
            .observe(current_step, algorithm, problem, status, args, config)
    }
}
impl<O, A, P, S, U, E, C> Observer<A, P, S, U, E, C> for Rc<RwLock<O>>
where
    O: Observer<A, P, S, U, E, C>,
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn observe(
        &mut self,
        current_step: usize,
        algorithm: &A,
        problem: &P,
        status: &S,
        args: &U,
        config: &C,
    ) {
        self.write()
            .observe(current_step, algorithm, problem, status, args, config)
    }
}
impl<O, A, P, S, U, E, C> Observer<A, P, S, U, E, C> for Arc<Mutex<O>>
where
    O: Observer<A, P, S, U, E, C>,
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn observe(
        &mut self,
        current_step: usize,
        algorithm: &A,
        problem: &P,
        status: &S,
        args: &U,
        config: &C,
    ) {
        self.lock()
            .observe(current_step, algorithm, problem, status, args, config)
    }
}
impl<O, A, P, S, U, E, C> Observer<A, P, S, U, E, C> for Arc<RwLock<O>>
where
    O: Observer<A, P, S, U, E, C>,
    A: Algorithm<P, S, U, E, Config = C>,
    S: Status,
{
    fn observe(
        &mut self,
        current_step: usize,
        algorithm: &A,
        problem: &P,
        status: &S,
        args: &U,
        config: &C,
    ) {
        self.write()
            .observe(current_step, algorithm, problem, status, args, config)
    }
}
