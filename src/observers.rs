use std::fmt::Debug;

use crate::{Observer, Status};

pub struct DebugObserver;
impl<T, U> Observer<T, U> for DebugObserver
where
    T: Debug,
    U: Debug,
{
    fn callback(&mut self, step: usize, status: &Status<T>, user_data: &mut U) {
        println!("{step}, {:?}, {:?}", status, user_data)
    }
}
