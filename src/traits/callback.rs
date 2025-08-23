use std::{convert::Infallible, ops::ControlFlow, sync::Arc};

use parking_lot::RwLock;

use crate::traits::{cost_function::Updatable, Algorithm, Status};

pub trait Callback<A, P, S, U = (), E = Infallible>
where
    A: Algorithm<P, S, U, E>,
    S: Status,
    P: Updatable<U, E>,
{
    fn callback(
        &mut self,
        current_step: usize,
        algorithm: &mut A,
        problem: &P,
        status: &mut S,
        user_data: &mut U,
    ) -> ControlFlow<()>; // TODO: return a break value?

    fn build(self) -> Arc<RwLock<dyn Callback<A, P, S, U, E>>>
    where
        Self: Sized + 'static,
    {
        Arc::new(RwLock::new(self))
    }
}

pub struct MaxSteps(pub usize);
impl Default for MaxSteps {
    fn default() -> Self {
        Self(4000)
    }
}
impl<A, P, S, U, E> Callback<A, P, S, U, E> for MaxSteps
where
    A: Algorithm<P, S, U, E>,
    P: Updatable<U, E>,
    S: Status,
{
    fn callback(
        &mut self,
        current_step: usize,
        _algorithm: &mut A,
        _problem: &P,
        _status: &mut S,
        _user_data: &mut U,
    ) -> ControlFlow<()> {
        if current_step >= self.0 {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}
