// use std::fmt::Debug;
//
// use fastrand::Rng;
// use fastrand_contrib::RngExt;
// use nalgebra::{DVector, Scalar};
//
// use crate::{algorithms::Point, convert, Algorithm, Bound, Function, Status};
//
// #[derive(Clone)]
// pub struct Swarm<T: Float + Debug + 'static> {
//     pcurrent: Vec<Point<T>>,
//     gbest: Point<T>,
//     pbest: Vec<Point<T>>,
//     velocity: Vec<DVector<T>>,
// }
//
// pub enum SwarmPositionConstructor<T: Float + Debug + 'static> {
//     RandomInBounds(Vec<Bound<T>>),
//     Custom(Vec<Point<T>>),
// }
//
// impl<T: Float + Debug + 'static> SwarmPositionConstructor<T> {
//     fn get_positions(&self, rng: &mut Rng, n: usize, dim: usize) -> Vec<Point<T>> {
//         match self {
//             SwarmPositionConstructor::RandomInBounds(bounds) => (0..n)
//                 .map(|_| {
//                     DVector::from_iterator(dim, bounds.iter().map(|bound| bound.get_uniform(rng)))
//                 })
//                 .map(|pos| pos.into())
//                 .collect(),
//             SwarmPositionConstructor::Custom(points) => (0..n).map(|i| points[i].clone()).collect(),
//         }
//     }
// }
//
// pub enum SwarmVelocityConstructor<T> {
//     RandomInLimits(Vec<Bound<T>>),
//     Zero,
// }
//
// #[derive(Clone, Copy)]
// pub enum BoundaryMethod {
//     Inf,
//     Nearest,
//     Random,
//     Shr,
//     Transform,
// }
//
// #[derive(Clone)]
// pub struct PSO<T: Float + Debug + 'static> {
//     swarm: Swarm<T>,
//     omega: T,
//     c1: T,
//     c2: T,
//     rng: Rng,
//     boundary_method: BoundaryMethod,
// }
//
// impl<T: Float + Clone + Debug + 'static> PSO<T> {
//     fn generate_random_vector(mut self, dimension: usize, lb: f64, ub: f64) -> DVector<T> {
//         DVector::from_iterator(
//             dimension,
//             std::iter::repeat_with(|| convert!(self.rng.f64_range(lb..ub), T)),
//         )
//     }
// }
//
// impl<T, U, E> Algorithm<T, U, E> for PSO<T>
// where
//     T: Float + Debug + 'static,
// {
//     fn initialize(
//         &mut self,
//         func: &dyn Function<T, U, E>,
//         x0: &[T],
//         bounds: Option<&Vec<Bound<T>>>,
//         user_data: &mut U,
//         status: &mut Status<T>,
//     ) -> Result<(), E> {
//         let dim = x0.len();
//         todo!()
//     }
//
//     fn step(
//         &mut self,
//         i_step: usize,
//         func: &dyn Function<T, U, E>,
//         bounds: Option<&Vec<Bound<T>>>,
//         user_data: &mut U,
//         status: &mut Status<T>,
//     ) -> Result<(), E> {
//         todo!()
//     }
//
//     fn check_for_termination(
//         &mut self,
//         func: &dyn Function<T, U, E>,
//         bounds: Option<&Vec<Bound<T>>>,
//         user_data: &mut U,
//         status: &mut Status<T>,
//     ) -> Result<bool, E> {
//         todo!()
//     }
// }
