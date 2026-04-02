use ganesh::{test_functions::rosenbrock::Rosenbrock, traits::Gradient, DVector, Float};
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let n_dim = parse_usize_arg(&args, 1, 200, "dimension");
    let x = DVector::from_fn(n_dim, |i, _| 2.0 + 0.001 * i as Float);
    let problem = Rosenbrock { n: n_dim };
    let hessian = problem.hessian(&x, &()).unwrap();
    println!(
        "computed hessian for dim={} shape={}x{} trace={}",
        n_dim,
        hessian.nrows(),
        hessian.ncols(),
        hessian.trace()
    );
}
