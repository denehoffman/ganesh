//! Run the same objective with multiple scalar and linear algebra representations.

use ganesh::{
    algorithms::gradient::TrustRegion, test_functions::Rosenbrock, traits::Algorithm, Vector,
};

fn main() {
    let problem = Rosenbrock { n: 2 };

    let init64: Vector = [-1.2, 1.0].into();
    let result64 = TrustRegion::default()
        .process_default(&problem, &(), init64)
        .expect("f64 trust-region run");
    let init32: Vector<f32> = [-1.2, 1.0].into();
    let result32 = TrustRegion::<f32>::default()
        .process_default(&problem, &(), init32)
        .expect("f32 trust-region run");

    println!("f64: {}", result64.fx);
    println!("f32: {}", result32.fx);

    #[cfg(feature = "backend-ndarray")]
    {
        use ganesh::NdArrayProvider;

        let ndarray_result = TrustRegion::<f64, NdArrayProvider>::default()
            .process_default(
                &problem,
                &(),
                Vector::<f64, NdArrayProvider>::from([-1.2, 1.0]),
            )
            .expect("ndarray trust-region run");
        println!("ndarray: {}", ndarray_result.fx);
    }
}
