use fastrand::Rng;
use ganesh::{
    algorithms::{
        gradient::{LBFGSBConfig, LBFGSB},
        gradient_free::{NelderMead, NelderMeadConfig},
        mcmc::{AutocorrelationTerminator, ESSConfig, ESS},
    },
    core::{summary::HasParameterNames, utils::SampleFloat, Bounds},
    traits::{
        algorithm::SupportsTransform, Algorithm, CostFunction, Gradient, LogDensity, Transform,
        TransformExt,
    },
    PI,
};
use nalgebra::{dmatrix, dvector, DMatrix, DVector, Matrix2, Vector2};
use parking_lot::Mutex;
use std::{borrow::Cow, convert::Infallible, error::Error, fs::File, io::BufWriter, sync::Arc};

fn generate_data(
    n: usize,
    mu: &DVector<f64>,
    cov: &DMatrix<f64>,
    rng: &mut Rng,
) -> Vec<DVector<f64>> {
    (0..n).map(|_| rng.mv_normal(mu, cov)).collect()
}

fn write_fit_result(
    truths: &[f64],
    data: &DVector<f64>,
    stderr: &DVector<f64>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    serde_pickle::to_writer(
        &mut writer,
        &(truths, data.data.as_vec(), stderr.data.as_vec()),
        Default::default(),
    )?;
    Ok(())
}

fn write_data(data: &[DVector<f64>], path: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let data: Vec<Vec<f64>> = data.iter().map(|x| x.data.as_vec()).cloned().collect();
    serde_pickle::to_writer(&mut writer, &data, Default::default())?;
    Ok(())
}

fn write_data_chain(
    data: &[Vec<DVector<f64>>],
    burn: usize,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let data: Vec<Vec<Vec<f64>>> = data
        .iter()
        .map(|w| w.iter().map(|x| x.data.as_vec()).cloned().collect())
        .collect();
    serde_pickle::to_writer(&mut writer, &(data, burn), Default::default())?;
    Ok(())
}

struct Problem;
impl CostFunction<Vec<DVector<f64>>> for Problem {
    fn evaluate(&self, x: &DVector<f64>, args: &Vec<DVector<f64>>) -> Result<f64, Infallible> {
        let mu = Vector2::new(x[0], x[1]);
        let sigma = Matrix2::new(x[2], x[3], x[3], x[4]);

        // Assume positive-definite, so this always works
        let chol = sigma.cholesky().unwrap();
        let l = chol.l();

        let ln_det = 2.0 * l.diagonal().iter().map(|v| v.ln()).sum::<f64>();

        let n = args.len() as f64;
        let quad_sum = args
            .iter()
            .map(|datum| {
                let d = datum - mu;
                d.dot(&chol.solve(&d))
            })
            .sum::<f64>();

        Ok((n * (2.0 * (2.0 * PI).ln() + ln_det)) + quad_sum)
    }
}
impl Gradient<Vec<DVector<f64>>> for Problem {}

impl LogDensity<Vec<DVector<f64>>> for Problem {
    fn log_density(&self, x: &DVector<f64>, args: &Vec<DVector<f64>>) -> Result<f64, Infallible> {
        Ok(-self.evaluate(x, args)?)
    }
}

#[derive(Clone)]
struct SPDTransform {
    i_00: usize,
    i_01: usize,
    i_11: usize,
}
impl Transform for SPDTransform {
    fn to_external<'a>(&'a self, z: &'a DVector<f64>) -> Cow<'a, DVector<f64>> {
        let (l00, l01, l11) = (z[self.i_00], z[self.i_01], z[self.i_11]);
        let (s00, s01, s11) = (l00 * l00, l00 * l01, l01 * l01 + l11 * l11);
        let mut x = z.clone();
        x[self.i_00] = s00;
        x[self.i_01] = s01;
        x[self.i_11] = s11;
        Cow::Owned(x)
    }

    fn to_internal<'a>(&'a self, x: &'a DVector<f64>) -> Cow<'a, DVector<f64>> {
        let (s00, s01, s11) = (x[self.i_00], x[self.i_01], x[self.i_11]);
        let l00 = s00.sqrt();
        let l01 = s01 / l00;
        let l11 = (s11 - l01 * l01).sqrt();
        let mut z = x.clone();
        z[self.i_00] = l00;
        z[self.i_01] = l01;
        z[self.i_11] = l11;
        Cow::Owned(z)
    }

    fn to_external_jacobian(&self, z: &DVector<f64>) -> DMatrix<f64> {
        let n = z.len();
        let (l00, l01, l11) = (z[self.i_00], z[self.i_01], z[self.i_11]);
        let mut j = DMatrix::identity(n, n);
        j[(self.i_00, self.i_00)] = 2.0 * l00;
        j[(self.i_01, self.i_00)] = l01;
        j[(self.i_01, self.i_01)] = l00;
        j[(self.i_11, self.i_01)] = 2.0 * l01;
        j[(self.i_11, self.i_11)] = 2.0 * l11;
        j
    }

    fn to_external_component_hessian(&self, a: usize, z: &DVector<f64>) -> DMatrix<f64> {
        let n = z.len();
        let mut h = DMatrix::zeros(n, n);
        if a == self.i_00 {
            h[(self.i_00, self.i_00)] = 2.0;
        } else if a == self.i_01 {
            h[(self.i_00, self.i_01)] = 1.0;
            h[(self.i_01, self.i_00)] = 1.0;
        } else if a == self.i_11 {
            h[(self.i_01, self.i_01)] = 2.0;
            h[(self.i_11, self.i_11)] = 2.0;
        }
        h
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut rng = Rng::with_seed(0);
    let truths = [1.2, 2.3, 0.6, 0.5, 0.7];
    let mu_truth = dvector![truths[0], truths[1]];
    let sigma_truth = dmatrix![truths[2], truths[3]; truths[3], truths[4]];
    let data = generate_data(10_000, &mu_truth, &sigma_truth, &mut rng);
    write_data(&data, "data.pkl")?;
    let spd_transform = SPDTransform {
        i_00: 2,
        i_01: 3,
        i_11: 4,
    };
    let internal_bounds = Bounds::from([
        (None, None),
        (None, None),
        (Some(0.0), None),
        (None, None),
        (Some(0.0), None),
    ]);
    let transform = internal_bounds.compose(spd_transform);

    println!("Running fit (Nelder-Mead)...");
    let res = NelderMead::default()
        .process_default(
            &Problem,
            &data,
            NelderMeadConfig::new([0.5, 1.0, 0.7, 0.1, 0.7]).with_transform(&transform),
        )?
        .with_parameter_names(["μ₀", "μ₁", "Σ₀₀", "Σ₀₁", "Σ₁₁"]);
    println!("{}", res);

    println!("Running fit (L-BFGS-B)...");
    let res = LBFGSB::default()
        .process_default(
            &Problem,
            &data,
            LBFGSBConfig::new([0.5, 1.0, 0.7, 0.1, 0.7]).with_transform(&transform),
        )?
        .with_parameter_names(["μ₀", "μ₁", "Σ₀₀", "Σ₀₁", "Σ₁₁"]);
    println!("{}", res);
    write_fit_result(&truths, &res.x, &res.std, "fit.pkl")?;

    let mut rng = Rng::with_seed(0);
    let aco = Arc::new(Mutex::new(AutocorrelationTerminator::default()));
    let x_min_int = transform.to_internal(&res.x);
    let n_walkers = 100;
    let walkers = Vec::from_iter((0..n_walkers).map(|_| {
        transform.to_owned_external(&DVector::from_fn(5, |i, _| rng.normal(x_min_int[i], 0.2)))
    }));
    println!("Running MCMC (ESS)...");
    let sample = ESS::default().process(
        &Problem,
        &data,
        ESSConfig::new(walkers).with_transform(&transform),
        ESS::default_callbacks().with_terminator(aco.clone()),
    )?;

    let burn = (aco.lock().taus.last().unwrap() * 10.0) as usize;

    let chain: Vec<Vec<DVector<f64>>> = sample.get_chain(None, None);
    write_data_chain(&chain, burn, "chain.pkl")?;

    let flat_chain: Vec<DVector<f64>> = sample.get_flat_chain(Some(burn), None);
    write_data(&flat_chain, "flat_chain.pkl")?;

    Ok(())
}
