use std::cell::Cell;

use ganesh::{
    algorithms::{
        gradient::{Adam, ConjugateGradient, GradientStatus, TrustRegion, LBFGSB},
        gradient_free::{
            DifferentialEvolution, GradientFreeStatus, NelderMead, SimulatedAnnealing, CMAES,
        },
        line_search::{BacktrackingLineSearch, HagerZhangLineSearch, MoreThuenteLineSearch},
        particles::PSO,
    },
    core::EvalCounts,
    traits::{
        Algorithm, CostFunction, Gradient, LineSearch, LineSearchFailure, Status, StatusType,
    },
    Matrix, Vector,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum EvaluationError {
    Domain,
    Fatal,
}

#[derive(Clone, Copy)]
enum Mode {
    Valid,
    RejectNext,
    RejectAll,
    Fatal,
}

struct Quadratic(Cell<Mode>);

impl CostFunction<f64, ganesh::NalgebraProvider, (), EvaluationError> for Quadratic {
    fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, EvaluationError> {
        match self.0.get() {
            Mode::Valid => Ok(x.dot(x)),
            Mode::RejectNext => {
                self.0.set(Mode::Valid);
                Err(EvaluationError::Domain)
            }
            Mode::RejectAll => Err(EvaluationError::Domain),
            Mode::Fatal => Err(EvaluationError::Fatal),
        }
    }

    fn trial_rejection(&self, error: &EvaluationError) -> Option<String> {
        (*error == EvaluationError::Domain).then(|| "test domain".to_owned())
    }
}

impl Gradient<f64, ganesh::NalgebraProvider, (), EvaluationError> for Quadratic {
    fn gradient(&self, x: &Vector, _: &()) -> Result<Vector, EvaluationError> {
        Ok(x.scale(2.0))
    }

    fn hessian(&self, x: &Vector, _: &()) -> Result<Matrix, EvaluationError> {
        Ok(Matrix::identity(x.len()).scale(2.0))
    }
}

macro_rules! exercise {
    ($solver:expr, $status:ty) => {{
        let mut solver = $solver;
        let mut status: $status = Default::default();
        let config = Default::default();
        let init = Vector::from_vec(vec![1.0, 2.0]);
        let problem = Quadratic(Cell::new(Mode::Valid));

        problem.0.set(Mode::RejectAll);
        assert_eq!(
            solver.initialize(&problem, &mut status, &(), &init, &config),
            Err(EvaluationError::Domain)
        );

        problem.0.set(Mode::Valid);
        solver
            .initialize(&problem, &mut status, &(), &init, &config)
            .unwrap();
        problem.0.set(Mode::RejectNext);
        solver.step(0, &problem, &mut status, &(), &config).unwrap();
        assert!(matches!(problem.0.get(), Mode::Valid));
        assert!(status.fx.is_finite());
        assert!(!matches!(status.message().status_type, StatusType::Failed));

        problem.0.set(Mode::Valid);
        solver
            .initialize(&problem, &mut status, &(), &init, &config)
            .unwrap();
        let last = status.x.to_vec();
        let value = status.fx;
        problem.0.set(Mode::RejectAll);
        for step in 0..16 {
            solver
                .step(step, &problem, &mut status, &(), &config)
                .unwrap();
            if status.check_invariants().is_break() {
                break;
            }
        }
        assert!(matches!(status.message().status_type, StatusType::Failed));
        assert_eq!(status.x.to_vec(), last);
        assert_eq!(status.fx, value);
        assert!(status.message().text_or_empty().contains("test domain"));

        problem.0.set(Mode::Valid);
        solver
            .initialize(&problem, &mut status, &(), &init, &config)
            .unwrap();
        problem.0.set(Mode::Fatal);
        assert_eq!(
            solver.step(0, &problem, &mut status, &(), &config),
            Err(EvaluationError::Fatal)
        );
    }};
}

#[test]
fn gradient_minimizers_reject_trials() {
    exercise!(LBFGSB::<f64>::default(), GradientStatus);
    exercise!(ConjugateGradient::<f64>::default(), GradientStatus);
    exercise!(TrustRegion::<f64>::default(), GradientStatus);
    exercise!(Adam::<f64>::default(), GradientStatus);
}

#[test]
fn derivative_free_minimizers_reject_trials() {
    exercise!(NelderMead::<f64>::default(), GradientFreeStatus);
    exercise!(SimulatedAnnealing::<f64>::default(), GradientFreeStatus);
    exercise!(CMAES::<f64>::default(), GradientFreeStatus);
    exercise!(DifferentialEvolution::<f64>::default(), GradientFreeStatus);
    exercise!(PSO::<f64>::default(), GradientFreeStatus);
}

struct Restricted {
    value_domain: bool,
    ceiling: f64,
}

impl CostFunction<f64, ganesh::NalgebraProvider, (), EvaluationError> for Restricted {
    fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, EvaluationError> {
        if self.value_domain && x.get(0) >= self.ceiling {
            Err(EvaluationError::Domain)
        } else {
            Ok((x.get(0) - 0.25).powi(2))
        }
    }

    fn trial_rejection(&self, error: &EvaluationError) -> Option<String> {
        (*error == EvaluationError::Domain).then(|| "test domain".to_owned())
    }
}

impl Gradient<f64, ganesh::NalgebraProvider, (), EvaluationError> for Restricted {
    fn gradient(&self, x: &Vector, _: &()) -> Result<Vector, EvaluationError> {
        if x.get(0) >= self.ceiling {
            Err(if self.value_domain {
                EvaluationError::Fatal
            } else {
                EvaluationError::Domain
            })
        } else {
            Ok(Vector::from_vec(vec![2.0 * (x.get(0) - 0.25)]))
        }
    }
}

fn line_search(
    search: &mut impl LineSearch<f64, ganesh::NalgebraProvider, Restricted, (), EvaluationError>,
) {
    let x = Vector::from_vec(vec![0.0]);
    let direction = Vector::from_vec(vec![1.0]);
    for value_domain in [true, false] {
        let problem = Restricted {
            value_domain,
            ceiling: 0.4,
        };
        let result = search
            .search(
                &x,
                &direction,
                Some(1.0),
                &problem,
                &(),
                &mut EvalCounts::default(),
            )
            .unwrap()
            .unwrap();
        assert!(result.alpha > 0.0 && result.alpha < problem.ceiling);
    }
    let problem = Restricted {
        value_domain: true,
        ceiling: f64::MIN_POSITIVE,
    };
    let result = search
        .search(
            &x,
            &direction,
            Some(1.0),
            &problem,
            &(),
            &mut EvalCounts::default(),
        )
        .unwrap();
    assert!(matches!(result, Err(LineSearchFailure::Rejected(_))));
}

#[test]
fn searches_contract_on_value_or_gradient_rejection() {
    line_search(&mut MoreThuenteLineSearch::default());
    line_search(&mut HagerZhangLineSearch::default());
    line_search(&mut BacktrackingLineSearch::default());
}
