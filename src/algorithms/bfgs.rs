use core::f32;
use std::collections::VecDeque;

use nalgebra::{DMatrix, DVector, RealField};
use num::Float;
use typed_builder::TypedBuilder;

use crate::{
    algorithms::line_search::TwoWayBacktrackingLineSearch,
    core::{convert, Field, Function, LineSearch, Minimizer},
};

#[derive(Debug, Default)]
pub enum InverseHessianApproximation {
    #[default]
    Identity,
    Exact,
}

impl InverseHessianApproximation {
    fn gradient_and_inverse_hessian<F, A, E>(
        &self,
        func: &dyn Function<F, A, E>,
        x: &DVector<F>,
        args: Option<&A>,
        ndim: usize,
    ) -> Result<(DVector<F>, DMatrix<F>), E>
    where
        F: Field + RealField,
    {
        match self {
            InverseHessianApproximation::Exact => func.gradient_and_inverse_hessian(x, args),
            InverseHessianApproximation::Identity => {
                Ok((func.gradient(x, args)?, DMatrix::identity(ndim, ndim)))
            }
        }
    }
}

#[derive(TypedBuilder)]
pub struct BFGSOptions<F, A, E>
where
    F: Field + 'static,
{
    #[builder(default = Box::new(TwoWayBacktrackingLineSearch::builder().build()))]
    pub line_search_method: Box<dyn LineSearch<F, A, E>>,
    /// The minimum absolute difference between evaluations that will terminate the
    /// algorithm (default = 1e-8)
    #[builder(default = convert!(1e-8, F))]
    pub tolerance: F,
    #[builder(default = None)]
    pub memory_limit: Option<usize>,
    #[builder(default = InverseHessianApproximation::default())]
    pub initial_hessian: InverseHessianApproximation,
}

pub struct BFGS<F, A, E>
where
    F: Field + 'static,
{
    function: Box<dyn Function<F, A, E>>,
    options: BFGSOptions<F, A, E>,
    x: DVector<F>,
    x_prev: Option<DVector<F>>,
    g: DVector<F>,
    fx: F,
    fx_old: F,
    x_best: DVector<F>,
    fx_best: F,
    current_step: usize,
    learning_rate: Option<F>,
    s_history: VecDeque<DVector<F>>,
    y_history: VecDeque<DVector<F>>,
    b0_inv: DMatrix<F>,
    p: DVector<F>,
    p_prev: Option<DVector<F>>,
}
impl<F, A, E> BFGS<F, A, E>
where
    F: Field,
{
    pub fn new<Func: Function<F, A, E> + 'static>(
        function: Func,
        x0: &[F],
        options: Option<BFGSOptions<F, A, E>>,
    ) -> Self {
        Self {
            function: Box::new(function),
            options: options.unwrap_or_else(|| BFGSOptions::builder().build()),
            x: DVector::from_row_slice(x0),
            x_prev: None,
            g: DVector::default(),
            fx: F::nan(),
            fx_old: F::nan(),
            x_best: DVector::from_element(x0.len(), F::nan()),
            fx_best: F::infinity(),
            current_step: 0,
            learning_rate: None,
            s_history: VecDeque::default(),
            y_history: VecDeque::default(),
            b0_inv: DMatrix::default(),
            p: DVector::default(),
            p_prev: None,
        }
    }
}

impl<F, A, E> Minimizer<F, A, E> for BFGS<F, A, E>
where
    F: Field + RealField,
{
    fn initialize(&mut self, args: Option<&A>) -> Result<(), E> {
        self.learning_rate = Some(self.options.line_search_method.get_base_learning_rate());
        (self.g, self.b0_inv) = self.options.initial_hessian.gradient_and_inverse_hessian(
            &*self.function,
            &self.x,
            args,
            self.x.len(),
        )?;
        self.p = &self.b0_inv * &self.g;
        Ok(())
    }
    fn step(&mut self, args: Option<&A>) -> Result<(), E> {
        self.current_step += 1;
        self.fx_old = self.fx;
        let (x_next, fx_next, learning_rate_next) = self.options.line_search_method.search(
            &*self.function,
            args,
            &self.x,
            &self.x_prev,
            &self.p,
            &self.p_prev,
            self.learning_rate,
        )?;
        let g_next = self.function.gradient(&x_next, args)?;
        let n_steps = self.s_history.len();
        if let Some(memory_limit) = self.options.memory_limit {
            if n_steps >= memory_limit {
                self.s_history.pop_front();
                self.y_history.pop_front();
            }
        }
        self.s_history.push_back(&x_next - &self.x);
        self.y_history.push_back(&g_next - &self.g);
        let mut rho_history: Vec<F> = vec![F::zero(); n_steps];
        let mut gamma_history: Vec<F> = vec![F::zero(); n_steps];
        let mut p_next = self.p.clone_owned();
        for i in (0..n_steps).rev() {
            rho_history[i] = F::one() / (self.y_history[i].dot(&self.s_history[i]));
            gamma_history[i] = rho_history[i] * (self.s_history[i].dot(&p_next));
            p_next -= self.y_history[i].scale(gamma_history[i]);
        }
        p_next = &self.b0_inv * p_next;
        for i in 0..n_steps {
            let beta = rho_history[i] * (self.y_history[i].dot(&p_next));
            p_next += self.s_history[i].scale(gamma_history[n_steps - i] - beta);
        }

        self.learning_rate = Some(learning_rate_next);
        self.x = x_next;
        self.fx = fx_next;
        self.g = g_next;
        self.p = p_next;
        Ok(())
    }

    fn check_for_termination(&self) -> bool {
        Float::abs(self.fx - self.fx_old) <= Float::abs(self.options.tolerance)
    }

    fn best(&self) -> (&DVector<F>, &F) {
        (&self.x_best, &self.fx_best)
    }

    fn update_best(&mut self) {
        if self.fx < self.fx_best {
            self.x_best = self.x.clone();
            self.fx_best = self.fx;
        }
    }
}
