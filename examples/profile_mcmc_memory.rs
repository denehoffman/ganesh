use ganesh::{
    DVector, Float,
    algorithms::mcmc::{AIES, AIESConfig, ESS, ESSConfig, ESSMove},
    core::MaxSteps,
    test_functions::rosenbrock::Rosenbrock,
    traits::Algorithm,
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

fn walkers(n_dim: usize, n_walkers: usize) -> Vec<DVector<Float>> {
    (0..n_walkers)
        .map(|i| {
            DVector::from_fn(n_dim, |j, _| {
                let center = if (i + j) % 2 == 0 { -1.5 } else { 1.5 };
                center + 0.02 * i as Float + 0.01 * j as Float
            })
        })
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let sampler = args.get(1).map(String::as_str).unwrap_or("aies");
    let n_dim = parse_usize_arg(&args, 2, 4, "dimension");
    let n_walkers = parse_usize_arg(&args, 3, 32, "walker count");
    let n_steps = parse_usize_arg(&args, 4, 2000, "step count");

    let problem = Rosenbrock { n: n_dim };
    let x0 = walkers(n_dim, n_walkers);

    match sampler {
        "aies" => {
            let mut solver = AIES::default();
            let summary = solver
                .process(
                    &problem,
                    &(),
                    AIESConfig::new(x0),
                    AIES::default_callbacks().with_terminator(MaxSteps(n_steps)),
                )
                .unwrap();
            println!(
                "sampler=aies walkers={} steps={} dim={} evals={}",
                n_walkers,
                n_steps,
                n_dim,
                summary.cost_evals
            );
        }
        "ess" => {
            let mut solver = ESS::default();
            let summary = solver
                .process(
                    &problem,
                    &(),
                    ESSConfig::new(x0)
                        .with_moves([ESSMove::gaussian(0.2), ESSMove::differential(0.8)])
                        .unwrap()
                        .with_n_adaptive(10)
                        .with_max_steps(128),
                    ESS::default_callbacks().with_terminator(MaxSteps(n_steps)),
                )
                .unwrap();
            println!(
                "sampler=ess walkers={} steps={} dim={} evals={}",
                n_walkers,
                n_steps,
                n_dim,
                summary.cost_evals
            );
        }
        other => {
            eprintln!("unknown sampler: {} (expected 'aies' or 'ess')", other);
            process::exit(2);
        }
    }
}
