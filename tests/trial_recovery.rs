use std::cell::Cell;

use ganesh::{
    algorithms::line_search::{
        BacktrackingLineSearch, HagerZhangLineSearch, MoreThuenteLineSearch, StrongWolfeLineSearch,
    },
    core::EvalCounts,
    traits::{CostFunction, Gradient, LineSearch, TransformedProblem},
    NalgebraProvider, Vector,
};

#[derive(Debug, PartialEq)]
enum EvaluationError {
    OutsideDomain,
    Fatal,
}

struct Restricted {
    ceiling: f64,
    reject_gradient: bool,
    recoverable: bool,
    fatal_trial: bool,
    invalid_gradient_calls: Cell<usize>,
}

impl Restricted {
    fn new(ceiling: f64, reject_gradient: bool) -> Self {
        Self {
            ceiling,
            reject_gradient,
            recoverable: true,
            fatal_trial: false,
            invalid_gradient_calls: Cell::new(0),
        }
    }
}

impl CostFunction<f64, NalgebraProvider, (), EvaluationError> for Restricted {
    fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, EvaluationError> {
        if x.get(0) >= self.ceiling && !self.reject_gradient {
            Err(if self.fatal_trial {
                EvaluationError::Fatal
            } else {
                EvaluationError::OutsideDomain
            })
        } else {
            Ok((x.get(0) - 0.25).powi(2))
        }
    }

    fn is_recoverable_trial_error(&self, error: &EvaluationError) -> bool {
        self.recoverable && *error == EvaluationError::OutsideDomain
    }
}

impl Gradient<f64, NalgebraProvider, (), EvaluationError> for Restricted {
    fn gradient(&self, x: &Vector, _: &()) -> Result<Vector, EvaluationError> {
        if x.get(0) >= self.ceiling {
            self.invalid_gradient_calls
                .set(self.invalid_gradient_calls.get() + 1);
            Err(if self.reject_gradient {
                EvaluationError::OutsideDomain
            } else {
                EvaluationError::Fatal
            })
        } else {
            Ok(Vector::from_vec(vec![2.0 * (x.get(0) - 0.25)]))
        }
    }
}

fn exercise(
    search: &mut impl for<'a> LineSearch<
        f64,
        NalgebraProvider,
        TransformedProblem<'a, Restricted, f64, NalgebraProvider>,
        (),
        EvaluationError,
    >,
) {
    let x = Vector::from_vec(vec![0.0]);
    let direction = Vector::from_vec(vec![1.0]);
    for reject_gradient in [false, true] {
        let problem = Restricted::new(if reject_gradient { 0.2 } else { 0.4 }, reject_gradient);
        let transformed = TransformedProblem::new(&problem, None);
        let output = search
            .search(
                &x,
                &direction,
                Some(1.0),
                &transformed,
                &(),
                &mut EvalCounts::default(),
            )
            .unwrap()
            .unwrap();
        assert!(output.alpha > 0.0 && output.alpha < problem.ceiling);
        assert_eq!(problem.invalid_gradient_calls.get() > 0, reject_gradient);
    }

    for (ceiling, recoverable, fatal_trial, expected) in [
        (0.0, true, false, EvaluationError::OutsideDomain),
        (
            f64::MIN_POSITIVE,
            true,
            false,
            EvaluationError::OutsideDomain,
        ),
        (0.4, false, false, EvaluationError::OutsideDomain),
        (0.4, true, true, EvaluationError::Fatal),
    ] {
        let mut problem = Restricted::new(ceiling, false);
        problem.recoverable = recoverable;
        problem.fatal_trial = fatal_trial;
        let transformed = TransformedProblem::new(&problem, None);
        let error = search
            .search(
                &x,
                &direction,
                Some(1.0),
                &transformed,
                &(),
                &mut EvalCounts::default(),
            )
            .unwrap_err();
        assert_eq!(error, expected);
        assert_eq!(problem.invalid_gradient_calls.get(), 0);
    }
}

#[test]
fn line_searches_recover_only_from_opted_in_trial_errors() {
    exercise(&mut BacktrackingLineSearch::default());
    exercise(&mut HagerZhangLineSearch::default());
    exercise(&mut MoreThuenteLineSearch::default());
    exercise(&mut StrongWolfeLineSearch::default());
    exercise(&mut StrongWolfeLineSearch::from(
        HagerZhangLineSearch::default(),
    ));
}
