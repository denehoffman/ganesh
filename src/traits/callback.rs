use parking_lot::{Mutex, RwLock};
use std::{cell::RefCell, ops::ControlFlow, rc::Rc, sync::Arc};

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
impl<T, A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for Rc<RefCell<T>>
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
        self.borrow_mut().check_for_termination(
            current_step,
            algorithm,
            problem,
            status,
            args,
            config,
        )
    }
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
impl<T, A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for Arc<RefCell<T>>
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
        self.borrow_mut().check_for_termination(
            current_step,
            algorithm,
            problem,
            status,
            args,
            config,
        )
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
impl<O, A, P, S, U, E, C> Observer<A, P, S, U, E, C> for Rc<RefCell<O>>
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
        self.borrow_mut()
            .observe(current_step, algorithm, problem, status, args, config)
    }
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
impl<O, A, P, S, U, E, C> Observer<A, P, S, U, E, C> for Arc<RefCell<O>>
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
        self.borrow_mut()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        algorithms::gradient::{LBFGSBConfig, LBFGSB},
        core::{summary::HasParameterNames, MaxSteps},
        test_functions::Rosenbrock,
    };

    #[derive(Default)]
    struct Trivial(usize);
    impl<A, P, S, U, E, C> Terminator<A, P, S, U, E, C> for Trivial
    where
        A: Algorithm<P, S, U, E, Config = C>,
        S: Status,
    {
        fn check_for_termination(
            &mut self,
            _current_step: usize,
            _algorithm: &mut A,
            _problem: &P,
            _status: &mut S,
            _args: &U,
            _config: &C,
        ) -> ControlFlow<()> {
            self.0 += 1;
            ControlFlow::Continue(())
        }
    }
    impl<A, P, S, U, E, C> Observer<A, P, S, U, E, C> for Trivial
    where
        A: Algorithm<P, S, U, E, Config = C>,
        S: Status,
    {
        fn observe(
            &mut self,
            _current_step: usize,
            _algorithm: &A,
            _problem: &P,
            _status: &S,
            _args: &U,
            _config: &C,
        ) {
            self.0 += 1;
        }
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn check_all_terminator_wrappers() {
        let rc_refcel = Rc::new(RefCell::new(Trivial::default()));
        let rc_rwlock = Rc::new(RwLock::new(Trivial::default()));
        let rc_mutex = Rc::new(Mutex::new(Trivial::default()));
        let arc_refcel = Arc::new(RefCell::new(Trivial::default()));
        let arc_rwlock = Arc::new(RwLock::new(Trivial::default()));
        let arc_mutex = Arc::new(Mutex::new(Trivial::default()));
        let res = LBFGSB::default()
            .process(
                &Rosenbrock { n: 2 },
                &(),
                LBFGSBConfig::new([2.0, 3.0]),
                LBFGSB::default_callbacks()
                    .with_terminator(rc_refcel.clone())
                    .with_terminator(rc_rwlock.clone())
                    .with_terminator(rc_mutex.clone())
                    .with_terminator(arc_refcel.clone())
                    .with_terminator(arc_rwlock.clone())
                    .with_terminator(arc_mutex.clone())
                    .with_observer(rc_refcel.clone())
                    .with_observer(rc_rwlock.clone())
                    .with_observer(rc_mutex.clone())
                    .with_observer(arc_refcel.clone())
                    .with_observer(arc_rwlock.clone())
                    .with_observer(arc_mutex.clone())
                    .with_terminator(MaxSteps(5)),
            )
            .unwrap()
            .with_parameter_names(["a", "b"]);
        assert_eq!(rc_refcel.borrow().0, 10); // 5 * 2 = 10 because each is called as both an
                                              // observer and a terminator
        assert_eq!(rc_rwlock.read().0, 10);
        assert_eq!(rc_mutex.lock().0, 10);
        assert_eq!(arc_refcel.borrow().0, 10);
        assert_eq!(arc_rwlock.read().0, 10);
        assert_eq!(arc_mutex.lock().0, 10);
        assert_eq!(res.message, "Maximum number of steps reached (5)!");
        assert_eq!(
            res.parameter_names,
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }
}
