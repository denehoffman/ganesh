use ganesh::{
    algorithms::{
        gradient::{
            ConjugateGradient, ConjugateGradientConfig, ConjugateGradientUpdate, LBFGSBConfig,
            LBFGSBErrorMode, LBFGSBInfNormGTerminator, LBFGSB,
        },
        line_search::{
            BacktrackingLineSearch, HagerZhangLineSearch, MoreThuenteLineSearch,
            StrongWolfeLineSearch,
        },
    },
    core::{Callbacks, EvalCounts, MaxSteps},
    traits::{
        Algorithm, CostFunction, Gradient, LineSearch, LineSearchResult, LineSearchStart,
        Transform, TransformedProblem,
    },
    Bounds, LinearAlgebra, Matrix, NalgebraProvider, RealScalar, ScaleTransform, Vector,
};
use std::{
    cell::{Cell, RefCell},
    convert::Infallible,
};

#[derive(Default)]
struct CountingQuadratic {
    f_points: RefCell<Vec<f64>>,
    g_points: RefCell<Vec<f64>>,
    fused_calls: Cell<usize>,
}

impl<T: RealScalar, B: LinearAlgebra<T>> CostFunction<T, B> for CountingQuadratic {
    fn evaluate(&self, x: &Vector<T, B>, _: &()) -> Result<T, Infallible> {
        self.f_points.borrow_mut().push(x.get(0).to_f64().unwrap());
        Ok(x.dot(x) * T::literal(0.5))
    }
}

impl<T: RealScalar, B: LinearAlgebra<T>> Gradient<T, B> for CountingQuadratic {
    fn gradient(&self, x: &Vector<T, B>, _: &()) -> Result<Vector<T, B>, Infallible> {
        self.g_points.borrow_mut().push(x.get(0).to_f64().unwrap());
        Ok(x.clone())
    }

    fn hessian(&self, x: &Vector<T, B>, _: &()) -> Result<Matrix<T, B>, Infallible> {
        Ok(Matrix::identity(x.len()))
    }

    fn evaluate_with_gradient(
        &self,
        x: &Vector<T, B>,
        _: &(),
    ) -> Result<(T, Vector<T, B>), Infallible> {
        self.fused_calls.set(self.fused_calls.get() + 1);
        self.f_points.borrow_mut().push(x.get(0).to_f64().unwrap());
        self.g_points.borrow_mut().push(x.get(0).to_f64().unwrap());
        Ok((x.dot(x) * T::literal(0.5), x.clone()))
    }
}

fn assert_seed_reused<L>(mut cold: L, mut cached: L)
where
    L: LineSearch<f64, NalgebraProvider, CountingQuadratic>,
{
    let cold_problem = CountingQuadratic::default();
    let cached_problem = CountingQuadratic::default();
    let x = Vector::from_vec(vec![2.0]);
    let direction = x.neg();
    let gradient = x.clone();
    let mut cold_evals = EvalCounts::default();
    let mut cached_evals = EvalCounts::default();
    let expected = cold
        .search(
            &x,
            &direction,
            Some(1.0),
            &cold_problem,
            &(),
            &mut cold_evals,
        )
        .unwrap()
        .unwrap();
    let actual = cached
        .search_from(
            LineSearchStart {
                x: &x,
                fx: 2.0,
                gradient: &gradient,
            },
            &direction,
            Some(1.0),
            &cached_problem,
            &(),
            &mut cached_evals,
        )
        .unwrap()
        .unwrap();
    assert_eq!(actual.alpha, expected.alpha);
    assert_eq!(actual.fx, expected.fx);
    assert_eq!(actual.gradient, expected.gradient);
    assert_eq!(cached_evals.f() + 1, cold_evals.f());
    assert_eq!(cached_evals.g() + 1, cold_evals.g());
    assert_eq!(cached_evals.f(), cached_problem.f_points.borrow().len());
    assert_eq!(cached_evals.g(), cached_problem.g_points.borrow().len());
    assert!(!cached_problem.f_points.borrow().contains(&2.0));
    assert!(!cached_problem.g_points.borrow().contains(&2.0));
}

#[test]
fn more_thuente_respects_the_step_cap_and_scaled_armijo_condition() {
    let problem = CountingQuadratic::default();
    let x: Vector = vec![1.0].into();
    let mut evals = EvalCounts::default();
    let output = MoreThuenteLineSearch::default()
        .with_c1(0.4)
        .unwrap()
        .search_from(
            LineSearchStart {
                x: &x,
                fx: 0.5,
                gradient: &x,
            },
            &Vector::from_vec(vec![-1.0]),
            Some(0.25),
            &problem,
            &(),
            &mut evals,
        )
        .unwrap()
        .unwrap();
    assert!(problem.f_points.borrow().iter().all(|&point| point >= 0.75));
    assert_eq!(output.alpha, 0.25);
    assert_eq!(output.fx, 0.5 * 0.75_f64.powi(2));
    assert_eq!(evals, EvalCounts::new(1, 1, 0));
}

#[test]
fn more_thuente_expands_locally_with_a_large_step_ceiling() {
    let problem = CountingQuadratic::default();
    let x: Vector = vec![1.0].into();
    let direction: Vector = vec![-0.01].into();
    let mut evals = EvalCounts::default();
    let output = MoreThuenteLineSearch::default()
        .search_from(
            LineSearchStart {
                x: &x,
                fx: 0.5,
                gradient: &x,
            },
            &direction,
            Some(1e8),
            &problem,
            &(),
            &mut evals,
        )
        .unwrap()
        .unwrap();
    assert!(evals.f() <= 8, "{} objective evaluations", evals.f());
    assert!(output.fx <= 0.5 - 1e-4 * output.alpha * 0.01);
    assert!(output.gradient.dot(&direction).abs() <= 0.9 * 0.01);
}

#[test]
fn more_thuente_zoom_limit_does_not_report_unmet_curvature_as_success() {
    let problem = CountingQuadratic::default();
    let x: Vector = vec![1.0].into();
    let mut evals = EvalCounts::default();
    let output = MoreThuenteLineSearch::default()
        .with_c2(0.1)
        .unwrap()
        .with_max_zoom(0)
        .search_from(
            LineSearchStart {
                x: &x,
                fx: 0.5,
                gradient: &x,
            },
            &Vector::from_vec(vec![-3.0]),
            None,
            &problem,
            &(),
            &mut evals,
        )
        .unwrap();
    assert!(
        output.is_err(),
        "zoom exhausted before satisfying curvature"
    );
}

#[test]
fn more_thuente_failure_values_belong_to_the_returned_step() {
    let problem = CountingQuadratic::default();
    let x: Vector = vec![1.0].into();
    let direction: Vector = vec![-0.01].into();
    let mut evals = EvalCounts::default();
    let output = MoreThuenteLineSearch::default()
        .with_max_iterations(0)
        .search_from(
            LineSearchStart {
                x: &x,
                fx: 0.5,
                gradient: &x,
            },
            &direction,
            Some(1e8),
            &problem,
            &(),
            &mut evals,
        )
        .unwrap()
        .unwrap_err();
    let trial = x.add_scaled(&direction, output.alpha);
    assert_eq!(output.alpha, 1.0);
    assert_eq!(output.fx, 0.5 * trial.dot(&trial));
    assert_eq!(output.gradient, trial);
    assert_eq!(evals, EvalCounts::new(1, 1, 0));
}

#[test]
fn more_thuente_rejects_invalid_limits_and_non_descent_without_evaluating() {
    let x: Vector = vec![1.0].into();
    for (limit, d) in [
        (0.0, -1.0),
        (-1.0, -1.0),
        (f64::NAN, -1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    ] {
        let problem = CountingQuadratic::default();
        let mut evals = EvalCounts::default();
        let output = MoreThuenteLineSearch::default()
            .search_from(
                LineSearchStart {
                    x: &x,
                    fx: 0.5,
                    gradient: &x,
                },
                &Vector::from_vec(vec![d]),
                Some(limit),
                &problem,
                &(),
                &mut evals,
            )
            .unwrap()
            .unwrap_err();
        assert_eq!(output.alpha, 0.0);
        assert_eq!(output.fx, 0.5);
        assert_eq!(output.gradient, x);
        assert_eq!(evals, EvalCounts::default());
        assert!(problem.f_points.borrow().is_empty());
        assert!(problem.g_points.borrow().is_empty());
    }
}

#[test]
fn more_thuente_accepts_a_decreasing_feasible_endpoint_before_curvature() {
    let problem = CountingQuadratic::default();
    let result = LBFGSB::<f64>::default()
        .process(
            &problem,
            &(),
            Vector::from_vec(vec![1.0]),
            LBFGSBConfig::default()
                .with_bounds([(0.95, 2.0)])
                .unwrap()
                .with_error_mode(LBFGSBErrorMode::Skip),
            Callbacks::empty()
                .with_terminator(LBFGSBInfNormGTerminator::new(1e-8).unwrap())
                .with_terminator(MaxSteps(5)),
        )
        .unwrap();
    assert!(result
        .message
        .to_string()
        .contains("PROJECTED GRADIENT WITHIN TOLERANCE"));
    assert_eq!(result.x.get(0), 0.95);
    assert_eq!(result.fx, 0.5 * 0.95_f64.powi(2));
    assert_eq!(result.evals, EvalCounts::new(2, 2, 0));
    assert!(problem
        .f_points
        .borrow()
        .iter()
        .all(|&point| (0.95..=2.0).contains(&point)));
    assert!(problem
        .g_points
        .borrow()
        .iter()
        .all(|&point| (0.95..=2.0).contains(&point)));
}

fn check_quadratic_step_conditions<T: RealScalar, B: LinearAlgebra<T>>() {
    let x = Vector::<T, B>::from_vec(vec![T::one(), T::literal(2.0)]);
    let fx = T::literal(0.5) * x.dot(&x);
    for (c1, c2) in [(1e-4, 0.9), (0.4, 0.8), (0.05, 0.1)] {
        for scale in [1e-3, 0.1, 1.0, 4.0, 100.0] {
            for cap in [None, Some(0.01), Some(0.25), Some(1.0), Some(1e8)] {
                let problem = CountingQuadratic::default();
                let direction = x.scale(-T::literal(scale));
                let mut evals = EvalCounts::default();
                let output = MoreThuenteLineSearch::<T, B>::default()
                    .with_c1_c2(T::literal(c1), T::literal(c2))
                    .unwrap()
                    .search_from(
                        LineSearchStart {
                            x: &x,
                            fx,
                            gradient: &x,
                        },
                        &direction,
                        cap.map(T::literal),
                        &problem,
                        &(),
                        &mut evals,
                    )
                    .unwrap()
                    .unwrap();
                assert!(output.alpha > T::zero());
                assert!(output.alpha <= cap.map(T::literal).unwrap_or_else(T::infinity));
                let trial = x.add_scaled(&direction, output.alpha);
                assert_eq!(output.fx, T::literal(0.5) * trial.dot(&trial));
                assert_eq!(output.gradient, trial);
                let derivative0 = x.dot(&direction);
                let derivative = output.gradient.dot(&direction);
                let tolerance = T::literal(32.0) * T::epsilon();
                assert!(
                    output.fx <= fx + T::literal(c1) * output.alpha * derivative0 + tolerance * fx
                );
                if derivative.abs() > (T::literal(c2) + tolerance) * derivative0.abs() {
                    assert_eq!(Some(output.alpha), cap.map(T::literal));
                    assert!(derivative < T::zero() && output.fx < fx);
                }
                assert_eq!(evals.f(), problem.f_points.borrow().len());
                assert_eq!(evals.g(), problem.g_points.borrow().len());
            }
        }
    }
}

#[test]
fn more_thuente_step_conditions_hold_across_scales_and_limits() {
    check_quadratic_step_conditions::<f64, NalgebraProvider>();
    check_quadratic_step_conditions::<f32, NalgebraProvider>();
}

#[cfg(feature = "backend-ndarray")]
#[test]
fn ndarray_more_thuente_step_conditions_hold_across_scales_and_limits() {
    check_quadratic_step_conditions::<f64, ganesh::NdArrayProvider>();
    check_quadratic_step_conditions::<f32, ganesh::NdArrayProvider>();
}

#[test]
fn built_in_searches_and_dispatch_reuse_the_starting_evaluation() {
    assert_seed_reused(
        MoreThuenteLineSearch::default(),
        MoreThuenteLineSearch::default(),
    );
    assert_seed_reused(
        HagerZhangLineSearch::default(),
        HagerZhangLineSearch::default(),
    );
    assert_seed_reused(
        BacktrackingLineSearch::default(),
        BacktrackingLineSearch::default(),
    );
    for search in [
        StrongWolfeLineSearch::default(),
        StrongWolfeLineSearch::from(HagerZhangLineSearch::default()),
    ] {
        assert_seed_reused(search.clone(), search);
    }
}

#[test]
fn zoom_reuses_endpoint_values() {
    let problem = CountingQuadratic::default();
    let x: Vector = Vector::from_vec(vec![1.0]);
    let gradient = x.clone();
    let mut evals = EvalCounts::default();
    let output = MoreThuenteLineSearch::default()
        .search_from(
            LineSearchStart {
                x: &x,
                fx: 0.5,
                gradient: &gradient,
            },
            &Vector::from_vec(vec![-8.0]),
            None,
            &problem,
            &(),
            &mut evals,
        )
        .unwrap()
        .unwrap();
    assert_eq!(output.alpha, 0.125);
    assert_eq!(output.fx, 0.0);
    assert_eq!(*problem.f_points.borrow(), vec![-7.0, -3.0, -1.0, 0.0]);
    assert_eq!(*problem.g_points.borrow(), vec![0.0]);
    assert_eq!(evals, EvalCounts::new(4, 1, 0));
}

#[test]
fn zoom_reuses_gradient_at_iteration_limit() {
    let problem = CountingQuadratic::default();
    let x: Vector = Vector::from_vec(vec![1.0]);
    let gradient = x.clone();
    let mut evals = EvalCounts::default();
    let output = MoreThuenteLineSearch::default()
        .with_c2(0.1)
        .unwrap()
        .with_max_zoom(0)
        .search_from(
            LineSearchStart {
                x: &x,
                fx: 0.5,
                gradient: &gradient,
            },
            &Vector::from_vec(vec![-3.0]),
            None,
            &problem,
            &(),
            &mut evals,
        )
        .unwrap()
        .unwrap_err();
    assert_eq!(output.alpha, 0.5);
    assert_eq!(output.fx, 0.125);
    assert_eq!(*problem.g_points.borrow(), vec![-0.5]);
    assert_eq!(evals, EvalCounts::new(2, 1, 0));
}

#[test]
fn zoom_reuses_values_when_the_lower_endpoint_changes() {
    let problem = CountingQuadratic::default();
    let x: Vector = vec![1.0].into();
    let gradient = x.clone();
    let mut evals = EvalCounts::default();
    let output = MoreThuenteLineSearch::default()
        .with_c2(0.1)
        .unwrap()
        .search_from(
            LineSearchStart {
                x: &x,
                fx: 0.5,
                gradient: &gradient,
            },
            &Vector::from_vec(vec![-1.5]),
            None,
            &problem,
            &(),
            &mut evals,
        )
        .unwrap()
        .unwrap();
    assert_eq!(output.alpha, 0.625);
    assert_eq!(*problem.f_points.borrow(), vec![-0.5, 0.25, -0.125, 0.0625]);
    assert_eq!(*problem.g_points.borrow(), vec![-0.5, 0.25, -0.125, 0.0625]);
    assert_eq!(evals, EvalCounts::new(4, 4, 0));
}

struct FusedFailure;

impl CostFunction<f64, NalgebraProvider, (), &'static str> for FusedFailure {
    fn evaluate(&self, _: &Vector, _: &()) -> Result<f64, &'static str> {
        Ok(0.0)
    }
}

impl Gradient<f64, NalgebraProvider, (), &'static str> for FusedFailure {
    fn gradient(&self, x: &Vector, _: &()) -> Result<Vector, &'static str> {
        Ok(x.clone())
    }
    fn evaluate_with_gradient(&self, _: &Vector, _: &()) -> Result<(f64, Vector), &'static str> {
        Err("fused evaluation failed")
    }
}

#[test]
fn fused_evaluation_errors_propagate_through_lbfgsb_and_transforms() {
    let scale = ScaleTransform::<f64>::from_parameter_scales([2.0]).unwrap();
    let transformed = TransformedProblem::new(&FusedFailure, Some(&scale));
    let x: Vector = vec![2.0].into();
    assert_eq!(
        transformed.evaluate_with_gradient(&x, &()).unwrap_err(),
        "fused evaluation failed"
    );
    let result = LBFGSB::<f64>::default().process(
        &FusedFailure,
        &(),
        x,
        LBFGSBConfig::default().with_error_mode(LBFGSBErrorMode::Skip),
        Callbacks::empty().with_terminator(MaxSteps(1)),
    );
    assert_eq!(result.unwrap_err(), "fused evaluation failed");
}

#[test]
fn lbfgsb_evaluates_only_initial_and_accepted_points_in_one_step() {
    let problem = CountingQuadratic::default();
    let result = LBFGSB::<f64>::default()
        .process(
            &problem,
            &(),
            Vector::from_vec(vec![2.0]),
            LBFGSBConfig::default().with_error_mode(LBFGSBErrorMode::Skip),
            Callbacks::empty().with_terminator(MaxSteps(1)),
        )
        .unwrap();
    assert_eq!(result.fx, 0.0);
    assert_eq!(result.evals, EvalCounts::new(2, 2, 0));
    assert_eq!(*problem.f_points.borrow(), vec![2.0, 0.0]);
    assert_eq!(*problem.g_points.borrow(), vec![2.0, 0.0]);
    assert_eq!(problem.fused_calls.get(), 1);
}

#[test]
fn cached_start_supports_f32() {
    let problem = CountingQuadratic::default();
    let result = LBFGSB::<f32>::default()
        .process(
            &problem,
            &(),
            Vector::from_vec(vec![2.0_f32]),
            LBFGSBConfig::default().with_error_mode(LBFGSBErrorMode::Skip),
            Callbacks::empty().with_terminator(MaxSteps(1)),
        )
        .unwrap();
    assert_eq!(result.fx, 0.0);
    assert_eq!(result.evals, EvalCounts::new(2, 2, 0));
    assert_eq!(problem.fused_calls.get(), 1);
}

#[test]
fn fused_evaluation_is_forwarded_with_and_without_transforms() {
    let internal: Vector = vec![0.3, -0.4].into();
    let scale = ScaleTransform::<f64>::from_parameter_scales([2.0, 0.5]).unwrap();
    let bounds = Bounds::<f64>::new([(-1.0, 3.0), (0.0, f64::INFINITY)]).unwrap();
    for transform in [
        None,
        Some(&scale as &dyn Transform<f64, NalgebraProvider>),
        Some(&bounds),
    ] {
        let problem = CountingQuadratic::default();
        let transformed = TransformedProblem::new(&problem, transform);
        let (fx, gradient) = transformed.evaluate_with_gradient(&internal, &()).unwrap();
        let external = transformed.to_external(&internal);
        let expected_gradient = transform.map_or_else(
            || external.clone(),
            |t| t.pullback_gradient(&internal, &external),
        );
        assert_eq!(fx, 0.5 * external.dot(&external));
        assert_eq!(gradient, expected_gradient);
        assert_eq!(problem.fused_calls.get(), 1);
        assert_eq!(problem.f_points.borrow().len(), 1);
        assert_eq!(problem.g_points.borrow().len(), 1);
    }
}

#[derive(Clone, Default)]
struct ReevaluatingSearch;

impl<P, U, E> LineSearch<f64, NalgebraProvider, P, U, E> for ReevaluatingSearch
where
    P: Gradient<f64, NalgebraProvider, U, E> + ?Sized,
{
    fn search_from(
        &mut self,
        start: LineSearchStart<'_, f64>,
        direction: &Vector,
        max_step: Option<f64>,
        problem: &P,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<LineSearchResult<f64, NalgebraProvider>, E> {
        // Emulate the previous optimizer behavior for the numerical-equivalence tests.
        MoreThuenteLineSearch::default().search(start.x, direction, max_step, problem, args, evals)
    }
}

#[derive(Default)]
struct SeededSearch;

impl<P, U, E> LineSearch<f64, NalgebraProvider, P, U, E> for SeededSearch
where
    P: Gradient<f64, NalgebraProvider, U, E> + ?Sized,
{
    fn search_from(
        &mut self,
        start: LineSearchStart<'_, f64>,
        direction: &Vector,
        max_step: Option<f64>,
        problem: &P,
        args: &U,
        evals: &mut EvalCounts,
    ) -> Result<LineSearchResult<f64, NalgebraProvider>, E> {
        MoreThuenteLineSearch::default()
            .search_from(start, direction, max_step, problem, args, evals)
    }
}

#[test]
fn custom_searches_only_need_to_implement_the_seeded_method() {
    assert_seed_reused(SeededSearch, SeededSearch);
}

#[test]
fn cached_and_reevaluating_searches_produce_the_same_bounded_and_transformed_fits() {
    for bounded in [false, true] {
        for scaled in [false, true] {
            let config = || {
                let mut config =
                    LBFGSBConfig::<f64>::default().with_error_mode(LBFGSBErrorMode::Skip);
                if bounded {
                    config = config.with_bounds([(-1.0, 3.0), (-2.0, 4.0)]).unwrap();
                }
                if scaled {
                    config = config
                        .with_transform(ScaleTransform::from_parameter_scales([2.0, 0.5]).unwrap())
                        .unwrap();
                }
                config
            };
            let x: Vector = vec![2.0, 2.0].into();
            let problem = ganesh::test_functions::Rosenbrock { n: 2 };
            let cached = LBFGSB::<f64>::default()
                .process(
                    &problem,
                    &(),
                    x.clone(),
                    config(),
                    Callbacks::empty().with_terminator(MaxSteps(20)),
                )
                .unwrap();
            let reevaluating = LBFGSB::<f64, NalgebraProvider, ReevaluatingSearch>::default()
                .process(
                    &problem,
                    &(),
                    x,
                    config().with_line_search(ReevaluatingSearch),
                    Callbacks::empty().with_terminator(MaxSteps(20)),
                )
                .unwrap();
            assert_eq!(
                cached.x, reevaluating.x,
                "bounded={bounded}, scaled={scaled}"
            );
            assert_eq!(cached.fx, reevaluating.fx);
            assert!(cached.evals.f() < reevaluating.evals.f());
            assert!(cached.evals.g() < reevaluating.evals.g());
        }
    }
}

#[test]
fn conjugate_gradient_reuses_the_initial_evaluation() {
    let problem = CountingQuadratic::default();
    let result = ConjugateGradient::<f64>::default()
        .process(
            &problem,
            &(),
            Vector::from_vec(vec![2.0]),
            ConjugateGradientConfig::default(),
            Callbacks::empty().with_terminator(MaxSteps(1)),
        )
        .unwrap();
    assert_eq!(result.fx, 0.0);
    assert_eq!(result.evals, EvalCounts::new(2, 2, 1));
    assert_eq!(*problem.f_points.borrow(), vec![2.0, 0.0]);
    assert_eq!(*problem.g_points.borrow(), vec![2.0, 0.0]);
    assert_eq!(problem.fused_calls.get(), 1);
}

#[test]
fn conjugate_gradient_cached_start_supports_f32() {
    let problem = CountingQuadratic::default();
    let result = ConjugateGradient::<f32>::default()
        .process(
            &problem,
            &(),
            Vector::from_vec(vec![2.0_f32]),
            ConjugateGradientConfig::default(),
            Callbacks::empty().with_terminator(MaxSteps(1)),
        )
        .unwrap();
    assert_eq!(result.fx, 0.0);
    assert_eq!(result.evals, EvalCounts::new(2, 2, 1));
    assert_eq!(problem.fused_calls.get(), 1);
}

#[test]
fn conjugate_gradient_updates_match_reevaluating_searches() {
    for update in [
        ConjugateGradientUpdate::FletcherReeves,
        ConjugateGradientUpdate::PolakRibierePlus,
        ConjugateGradientUpdate::HestenesStiefelPlus,
        ConjugateGradientUpdate::DaiYuan,
        ConjugateGradientUpdate::HagerZhang,
    ] {
        for transform in 0..3 {
            let config = || {
                let config = ConjugateGradientConfig::<f64>::default().with_update(update);
                match transform {
                    0 => config,
                    1 => config
                        .with_transform(ScaleTransform::from_parameter_scales([2.0, 0.5]).unwrap()),
                    _ => config.with_transform(Bounds::new([(-1.0, 3.0), (-2.0, 4.0)]).unwrap()),
                }
            };
            let problem = ganesh::test_functions::Rosenbrock { n: 2 };
            let x: Vector = vec![2.0, 2.0].into();
            let cached = ConjugateGradient::<f64>::default()
                .process(
                    &problem,
                    &(),
                    x.clone(),
                    config(),
                    Callbacks::empty().with_terminator(MaxSteps(20)),
                )
                .unwrap();
            let reevaluating =
                ConjugateGradient::<f64, NalgebraProvider, ReevaluatingSearch>::default()
                    .process(
                        &problem,
                        &(),
                        x,
                        config().with_line_search(ReevaluatingSearch),
                        Callbacks::empty().with_terminator(MaxSteps(20)),
                    )
                    .unwrap();
            assert_eq!(
                cached.x, reevaluating.x,
                "update={update:?}, transform={transform}"
            );
            assert_eq!(cached.fx, reevaluating.fx);
            assert!(cached.evals.f() < reevaluating.evals.f());
            assert!(cached.evals.g() < reevaluating.evals.g());
        }
    }
}

#[test]
fn default_search_propagates_fused_evaluation_errors() {
    let x: Vector = vec![2.0].into();
    let mut evals = EvalCounts::default();
    let result = SeededSearch.search(&x, &x.neg(), None, &FusedFailure, &(), &mut evals);
    assert_eq!(result.unwrap_err(), "fused evaluation failed");
    assert_eq!(evals, EvalCounts::new(1, 1, 0));
}
