//! Run the same objective with multiple scalar representations and linear algebra backends.

use ganesh::{
    algorithms::gradient::TrustRegion, test_functions::Rosenbrock, traits::Algorithm,
    NalgebraBackend, Vector,
};

fn main() {
    let problem = Rosenbrock { n: 2 };

    let result64 = TrustRegion::<f64, NalgebraBackend>::default()
        .process_default(&problem, &(), Vector::from_vec(vec![-1.2_f64, 1.0]))
        .expect("f64 trust-region run");
    let result32 = TrustRegion::<f32, NalgebraBackend>::default()
        .process_default(&problem, &(), Vector::from_vec(vec![-1.2_f32, 1.0]))
        .expect("f32 trust-region run");

    println!("f64: {}", result64.fx);
    println!("f32: {}", result32.fx);

    #[cfg(feature = "backend-ndarray")]
    {
        use ganesh::NdArrayBackend;

        let ndarray_result = TrustRegion::<f64, NdArrayBackend>::default()
            .process_default(&problem, &(), Vector::from_vec(vec![-1.2_f64, 1.0]))
            .expect("ndarray trust-region run");
        println!("ndarray: {}", ndarray_result.fx);
    }
}
