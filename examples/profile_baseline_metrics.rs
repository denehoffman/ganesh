use ganesh::{
    algorithms::{
        gradient::{lbfgsb::LBFGSBConfig, LBFGSB},
        gradient_free::{nelder_mead::NelderMeadConfig, NelderMead},
        mcmc::{AIESConfig, ESSConfig, ESSMove, AIES, ESS},
        particles::{PSOConfig, Swarm, SwarmPositionInitializer, PSO},
    },
    core::MaxSteps,
    test_functions::{rastrigin::Rastrigin, rosenbrock::Rosenbrock},
    traits::Algorithm,
    DVector, Float,
};
use std::{env, process};

fn parse_usize_arg(args: &[String], index: usize, default: usize, name: &str) -> usize {
    args.get(index)
        .map(|value| {
            value.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("invalid {}: {}", name, value);
                process::exit(2);
            })
        })
        .unwrap_or(default)
}

fn make_walkers(n_dim: usize, n_walkers: usize) -> Vec<DVector<Float>> {
    (0..n_walkers)
        .map(|i| {
            DVector::from_fn(n_dim, |j, _| {
                let center = if (i + j) % 2 == 0 { -1.5 } else { 1.5 };
                center + 0.02 * i as Float + 0.01 * j as Float
            })
        })
        .collect()
}

fn print_metrics(
    algorithm: &str,
    problem: &str,
    dim: usize,
    cost_evals: usize,
    gradient_evals: usize,
    success: bool,
    message: &str,
) {
    println!("algorithm={algorithm}");
    println!("problem={problem}");
    println!("dim={dim}");
    println!("cost_evals={cost_evals}");
    println!("gradient_evals={gradient_evals}");
    println!("success={success}");
    println!("message={message}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let algorithm = args.get(1).map(String::as_str).unwrap_or("lbfgsb");
    let dim = parse_usize_arg(&args, 2, 2, "dimension");

    match algorithm {
        "lbfgsb" => {
            let problem = Rosenbrock { n: dim };
            let mut solver = LBFGSB::default();
            let summary = solver
                .process(
                    &problem,
                    &(),
                    LBFGSBConfig::new(vec![5.0; dim]),
                    LBFGSB::default_callbacks().with_terminator(MaxSteps(80)),
                )
                .unwrap();
            print_metrics(
                "lbfgsb",
                "rosenbrock",
                dim,
                summary.cost_evals,
                summary.gradient_evals,
                summary.message.success(),
                &summary.message.to_string(),
            );
        }
        "nelder_mead" => {
            let problem = Rosenbrock { n: dim };
            let mut solver = NelderMead::default();
            let summary = solver
                .process(
                    &problem,
                    &(),
                    NelderMeadConfig::new(vec![5.0; dim]),
                    NelderMead::default_callbacks().with_terminator(MaxSteps(120)),
                )
                .unwrap();
            print_metrics(
                "nelder_mead",
                "rosenbrock",
                dim,
                summary.cost_evals,
                summary.gradient_evals,
                summary.message.success(),
                &summary.message.to_string(),
            );
        }
        "pso" => {
            let problem = Rastrigin { n: dim };
            let mut solver = PSO::default();
            let bounds = vec![(-5.12, 5.12); dim];
            let summary = solver
                .process(
                    &problem,
                    &(),
                    PSOConfig::new(Swarm::new(SwarmPositionInitializer::RandomInLimits {
                        bounds,
                        n_particles: 24,
                    }))
                    .with_c1(0.1)
                    .unwrap()
                    .with_c2(0.1)
                    .unwrap()
                    .with_omega(0.8)
                    .unwrap(),
                    PSO::default_callbacks().with_terminator(MaxSteps(60)),
                )
                .unwrap();
            print_metrics(
                "pso",
                "rastrigin",
                dim,
                summary.cost_evals,
                summary.gradient_evals,
                summary.message.success(),
                &summary.message.to_string(),
            );
        }
        "aies" => {
            let n_walkers = parse_usize_arg(&args, 3, 12, "walker count");
            let steps = parse_usize_arg(&args, 4, 40, "step count");
            let problem = Rosenbrock { n: dim };
            let mut solver = AIES::default();
            let summary = solver
                .process(
                    &problem,
                    &(),
                    AIESConfig::new(make_walkers(dim, n_walkers)).unwrap(),
                    AIES::default_callbacks().with_terminator(MaxSteps(steps)),
                )
                .unwrap();
            print_metrics(
                "aies",
                "rosenbrock_log_density",
                dim,
                summary.cost_evals,
                summary.gradient_evals,
                summary.message.success(),
                &summary.message.to_string(),
            );
        }
        "ess" => {
            let n_walkers = parse_usize_arg(&args, 3, 12, "walker count");
            let steps = parse_usize_arg(&args, 4, 40, "step count");
            let problem = Rosenbrock { n: dim };
            let mut solver = ESS::default();
            let summary = solver
                .process(
                    &problem,
                    &(),
                    ESSConfig::new(make_walkers(dim, n_walkers))
                        .unwrap()
                        .with_moves([ESSMove::gaussian(0.2), ESSMove::differential(0.8)])
                        .unwrap()
                        .with_n_adaptive(5)
                        .with_max_steps(64),
                    ESS::default_callbacks().with_terminator(MaxSteps(steps)),
                )
                .unwrap();
            print_metrics(
                "ess",
                "rosenbrock_log_density",
                dim,
                summary.cost_evals,
                summary.gradient_evals,
                summary.message.success(),
                &summary.message.to_string(),
            );
        }
        other => {
            eprintln!(
                "unknown algorithm: {} (expected lbfgsb, nelder_mead, pso, aies, or ess)",
                other
            );
            process::exit(2);
        }
    }
}
