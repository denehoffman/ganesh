//! Python-facing config extraction traits for downstream wrapper crates.

use pyo3::{types::PyAnyMethods, Borrowed, Bound, FromPyObject, PyAny, PyResult};

use crate::{
    algorithms::{
        gradient::{
            lbfgsb::LBFGSBErrorMode, AdamConfig, ConjugateGradientConfig, ConjugateGradientUpdate,
            LBFGSBConfig, TrustRegionConfig, TrustRegionSubproblem,
        },
        gradient_free::{
            nelder_mead::{NelderMeadInit, SimplexConstructionMethod, SimplexExpansionMethod},
            CMAESConfig, CMAESInit, DifferentialEvolutionConfig, DifferentialEvolutionInit,
            NelderMeadConfig, SimulatedAnnealingConfig,
        },
        line_search::{HagerZhangLineSearch, MoreThuenteLineSearch, StrongWolfeLineSearch},
        mcmc::{
            aies::{AIESInit, WeightedAIESMove},
            ess::{ESSInit, WeightedESSMove},
            AIESConfig, AIESMove, ChainStorageMode, ESSConfig, ESSMove,
        },
        particles::{
            PSOConfig, Swarm, SwarmBoundaryMethod, SwarmPositionInitializer, SwarmTopology,
            SwarmUpdateMethod,
        },
    },
    error::GaneshError,
    python::{
        extract::{extract_optional_field, extract_required_field, resolve_protocol},
        numeric::{extract_matrix, extract_vector},
    },
    traits::{
        algorithm::BoundsHandlingMode, Bound as GaneshBound, SupportsBounds, SupportsParameterNames,
    },
    DVector, Float,
};

fn apply_python_bounds<C>(mut config: C, bounds: Option<Vec<(Option<Float>, Option<Float>)>>) -> C
where
    C: SupportsBounds,
{
    if let Some(bounds) = bounds {
        config = config.with_bounds(bounds.into_iter().map(GaneshBound::from));
    }
    config
}

fn apply_python_parameter_names<C>(mut config: C, parameter_names: Option<Vec<String>>) -> C
where
    C: SupportsParameterNames,
{
    if let Some(parameter_names) = parameter_names {
        config = config.with_parameter_names(parameter_names);
    }
    config
}

fn vectors_to_dvectors(vectors: &[Vec<Float>]) -> Vec<DVector<Float>> {
    vectors.iter().cloned().map(DVector::from_vec).collect()
}

fn config_error(message: impl Into<String>) -> pyo3::PyErr {
    GaneshError::ConfigError(message.into()).into()
}

fn normalize_choice(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn parse_bounds_handling(value: &str) -> PyResult<BoundsHandlingMode> {
    match normalize_choice(value).as_str() {
        "auto" => Ok(BoundsHandlingMode::Auto),
        "native_bounds" => Ok(BoundsHandlingMode::NativeBounds),
        "transform_bounds" => Ok(BoundsHandlingMode::TransformBounds),
        _ => Err(config_error(format!(
            "unknown bounds_handling `{value}`; expected one of auto, native_bounds, transform_bounds"
        ))),
    }
}

fn parse_lbfgsb_error_mode(value: &str) -> PyResult<LBFGSBErrorMode> {
    match normalize_choice(value).as_str() {
        "exact_hessian" => Ok(LBFGSBErrorMode::ExactHessian),
        "skip" => Ok(LBFGSBErrorMode::Skip),
        _ => Err(config_error(format!(
            "unknown LBFGSB error_mode `{value}`; expected one of exact_hessian, skip"
        ))),
    }
}

fn parse_swarm_topology(value: &str) -> PyResult<SwarmTopology> {
    match normalize_choice(value).as_str() {
        "global" => Ok(SwarmTopology::Global),
        "ring" => Ok(SwarmTopology::Ring),
        _ => Err(config_error(format!(
            "unknown swarm topology `{value}`; expected one of global, ring"
        ))),
    }
}

fn parse_swarm_update_method(value: &str) -> PyResult<SwarmUpdateMethod> {
    match normalize_choice(value).as_str() {
        "synchronous" => Ok(SwarmUpdateMethod::Synchronous),
        "asynchronous" => Ok(SwarmUpdateMethod::Asynchronous),
        _ => Err(config_error(format!(
            "unknown swarm update_method `{value}`; expected one of synchronous, asynchronous"
        ))),
    }
}

fn parse_swarm_boundary_method(value: &str) -> PyResult<SwarmBoundaryMethod> {
    match normalize_choice(value).as_str() {
        "inf" => Ok(SwarmBoundaryMethod::Inf),
        "shr" => Ok(SwarmBoundaryMethod::Shr),
        _ => Err(config_error(format!(
            "unknown swarm boundary_method `{value}`; expected one of inf, shr"
        ))),
    }
}

fn parse_cg_update(value: &str) -> PyResult<ConjugateGradientUpdate> {
    match normalize_choice(value).as_str() {
        "fletcher_reeves" => Ok(ConjugateGradientUpdate::FletcherReeves),
        "polak_ribiere_plus" => Ok(ConjugateGradientUpdate::PolakRibierePlus),
        "hestenes_stiefel_plus" => Ok(ConjugateGradientUpdate::HestenesStiefelPlus),
        "dai_yuan" => Ok(ConjugateGradientUpdate::DaiYuan),
        "hager_zhang" => Ok(ConjugateGradientUpdate::HagerZhang),
        _ => Err(config_error(format!(
            "unknown conjugate-gradient update `{value}`; expected one of fletcher_reeves, polak_ribiere_plus, hestenes_stiefel_plus, dai_yuan, hager_zhang"
        ))),
    }
}

fn parse_trust_region_subproblem(value: &str) -> PyResult<TrustRegionSubproblem> {
    match normalize_choice(value).as_str() {
        "cauchy_point" => Ok(TrustRegionSubproblem::CauchyPoint),
        "dogleg" => Ok(TrustRegionSubproblem::Dogleg),
        _ => Err(config_error(format!(
            "unknown trust-region subproblem `{value}`; expected one of cauchy_point, dogleg"
        ))),
    }
}

fn parse_simplex_expansion_method(value: &str) -> PyResult<SimplexExpansionMethod> {
    match normalize_choice(value).as_str() {
        "greedy_minimization" => Ok(SimplexExpansionMethod::GreedyMinimization),
        "greedy_expansion" => Ok(SimplexExpansionMethod::GreedyExpansion),
        _ => Err(config_error(format!(
            "unknown simplex expansion_method `{value}`; expected one of greedy_minimization, greedy_expansion"
        ))),
    }
}

fn parse_line_search<'py>(obj: &Bound<'py, PyAny>) -> PyResult<StrongWolfeLineSearch> {
    if let Ok(kind) = obj.extract::<String>() {
        return match normalize_choice(&kind).as_str() {
            "more_thuente" => Ok(StrongWolfeLineSearch::MoreThuente(
                MoreThuenteLineSearch::default(),
            )),
            "hager_zhang" => Ok(StrongWolfeLineSearch::HagerZhang(
                HagerZhangLineSearch::default(),
            )),
            _ => Err(config_error(format!(
                "unknown line_search `{kind}`; expected one of more_thuente, hager_zhang"
            ))),
        };
    }

    let obj = resolve_protocol(obj, "__ganesh_line_search__")?;
    let kind: String = extract_required_field(&obj, "kind")?;
    match normalize_choice(&kind).as_str() {
        "more_thuente" => {
            let max_iterations: Option<usize> = extract_optional_field(&obj, "max_iterations")?;
            let max_zoom: Option<usize> = extract_optional_field(&obj, "max_zoom")?;
            let c1: Option<Float> = extract_optional_field(&obj, "c1")?;
            let c2: Option<Float> = extract_optional_field(&obj, "c2")?;
            let mut line_search = MoreThuenteLineSearch::default()
                .with_c1_c2(c1.unwrap_or(1e-4), c2.unwrap_or(0.9))?;
            if let Some(max_iterations) = max_iterations {
                line_search = line_search.with_max_iterations(max_iterations);
            }
            if let Some(max_zoom) = max_zoom {
                line_search = line_search.with_max_zoom(max_zoom);
            }
            Ok(StrongWolfeLineSearch::MoreThuente(line_search))
        }
        "hager_zhang" => {
            let max_iterations: Option<usize> = extract_optional_field(&obj, "max_iterations")?;
            let delta: Option<Float> = extract_optional_field(&obj, "delta")?;
            let sigma: Option<Float> = extract_optional_field(&obj, "sigma")?;
            let epsilon: Option<Float> = extract_optional_field(&obj, "epsilon")?;
            let theta: Option<Float> = extract_optional_field(&obj, "theta")?;
            let gamma: Option<Float> = extract_optional_field(&obj, "gamma")?;
            let max_bisects: Option<usize> = extract_optional_field(&obj, "max_bisects")?;
            let mut line_search = HagerZhangLineSearch::default()
                .with_delta_sigma(delta.unwrap_or(0.1), sigma.unwrap_or(0.9))?
                .with_epsilon(epsilon.unwrap_or(Float::EPSILON.cbrt()))?
                .with_theta(theta.unwrap_or(0.5))?
                .with_gamma(gamma.unwrap_or(0.66))?;
            if let Some(max_iterations) = max_iterations {
                line_search = line_search.with_max_iterations(max_iterations);
            }
            if let Some(max_bisects) = max_bisects {
                line_search = line_search.with_max_bisects(max_bisects);
            }
            Ok(StrongWolfeLineSearch::HagerZhang(line_search))
        }
        _ => Err(config_error(format!(
            "unknown line_search kind `{kind}`; expected one of more_thuente, hager_zhang"
        ))),
    }
}

fn parse_simplex_construction<'py>(obj: &Bound<'py, PyAny>) -> PyResult<SimplexConstructionMethod> {
    let obj = resolve_protocol(obj, "__ganesh_simplex_construction__")?;
    let kind: String = extract_required_field(&obj, "kind")?;
    match normalize_choice(&kind).as_str() {
        "scaled_orthogonal" => {
            let x0 = extract_vector(&extract_required_field::<Bound<'py, PyAny>>(&obj, "x0")?)?;
            let orthogonal_multiplier: Option<Float> =
                extract_optional_field(&obj, "orthogonal_multiplier")?;
            let orthogonal_zero_step: Option<Float> =
                extract_optional_field(&obj, "orthogonal_zero_step")?;
            SimplexConstructionMethod::custom_scaled_orthogonal(
                &x0,
                orthogonal_multiplier.unwrap_or(1.05),
                orthogonal_zero_step.unwrap_or(0.00025),
            )
            .map_err(Into::into)
        }
        "orthogonal" => {
            let x0 = extract_vector(&extract_required_field::<Bound<'py, PyAny>>(&obj, "x0")?)?;
            let simplex_size: Option<Float> = extract_optional_field(&obj, "simplex_size")?;
            SimplexConstructionMethod::custom_orthogonal(&x0, simplex_size.unwrap_or(1.0))
                .map_err(Into::into)
        }
        "custom" => {
            let simplex =
                extract_matrix(&extract_required_field::<Bound<'py, PyAny>>(&obj, "simplex")?)?;
            SimplexConstructionMethod::custom(vectors_to_dvectors(&simplex)).map_err(Into::into)
        }
        _ => Err(config_error(format!(
            "unknown simplex construction kind `{kind}`; expected one of scaled_orthogonal, orthogonal, custom"
        ))),
    }
}

fn parse_chain_storage<'py>(obj: &Bound<'py, PyAny>) -> PyResult<ChainStorageMode> {
    if let Ok(kind) = obj.extract::<String>() {
        return match normalize_choice(&kind).as_str() {
            "full" => Ok(ChainStorageMode::Full),
            _ => Err(config_error(format!(
                "unknown chain_storage `{kind}`; expected full or a chain storage helper object"
            ))),
        };
    }

    let obj = resolve_protocol(obj, "__ganesh_chain_storage__")?;
    let kind: String = extract_required_field(&obj, "kind")?;
    match normalize_choice(&kind).as_str() {
        "full" => Ok(ChainStorageMode::Full),
        "rolling" => {
            let window: usize = extract_required_field(&obj, "window")?;
            Ok(ChainStorageMode::Rolling { window })
        }
        "sampled" => {
            let keep_every: usize = extract_required_field(&obj, "keep_every")?;
            let max_samples: Option<usize> = extract_optional_field(&obj, "max_samples")?;
            Ok(ChainStorageMode::Sampled {
                keep_every,
                max_samples,
            })
        }
        _ => Err(config_error(format!(
            "unknown chain storage kind `{kind}`; expected one of full, rolling, sampled"
        ))),
    }
}

fn parse_aies_move<'py>(obj: &Bound<'py, PyAny>) -> PyResult<WeightedAIESMove> {
    let obj = resolve_protocol(obj, "__ganesh_move__")?;
    let kind: String = extract_required_field(&obj, "kind")?;
    let weight: Option<Float> = extract_optional_field(&obj, "weight")?;
    let weight = weight.unwrap_or(1.0);
    match normalize_choice(&kind).as_str() {
        "stretch" => {
            let a: Option<Float> = extract_optional_field(&obj, "a")?;
            match a {
                Some(a) => AIESMove::custom_stretch(a, weight).map_err(Into::into),
                None => Ok(AIESMove::stretch(weight)),
            }
        }
        "walk" => Ok(AIESMove::walk(weight)),
        _ => Err(config_error(format!(
            "unknown AIES move kind `{kind}`; expected one of stretch, walk"
        ))),
    }
}

fn parse_aies_moves<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<WeightedAIESMove>> {
    let mut moves = Vec::new();
    for item in obj.try_iter()? {
        moves.push(parse_aies_move(&item?)?);
    }
    Ok(moves)
}

fn parse_ess_move<'py>(obj: &Bound<'py, PyAny>) -> PyResult<WeightedESSMove> {
    let obj = resolve_protocol(obj, "__ganesh_move__")?;
    let kind: String = extract_required_field(&obj, "kind")?;
    let weight: Option<Float> = extract_optional_field(&obj, "weight")?;
    let weight = weight.unwrap_or(1.0);
    match normalize_choice(&kind).as_str() {
        "differential" => Ok(ESSMove::differential(weight)),
        "gaussian" => Ok(ESSMove::gaussian(weight)),
        "global" => {
            let scale: Option<Float> = extract_optional_field(&obj, "scale")?;
            let rescale_cov: Option<Float> = extract_optional_field(&obj, "rescale_cov")?;
            let n_components: Option<usize> = extract_optional_field(&obj, "n_components")?;
            let is_default = scale.is_none() && rescale_cov.is_none() && n_components.is_none();
            if is_default {
                Ok(ESSMove::global(weight))
            } else {
                ESSMove::custom_global(weight, scale, rescale_cov, n_components).map_err(Into::into)
            }
        }
        _ => Err(config_error(format!(
            "unknown ESS move kind `{kind}`; expected one of differential, gaussian, global"
        ))),
    }
}

fn parse_ess_moves<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Vec<WeightedESSMove>> {
    let mut moves = Vec::new();
    for item in obj.try_iter()? {
        moves.push(parse_ess_move(&item?)?);
    }
    Ok(moves)
}

impl<'a, 'py> FromPyObject<'a, 'py> for LBFGSBConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_init__")?;
        let _x0 = extract_vector(&extract_required_field::<Bound<'py, PyAny>>(&obj, "x0")?)?;
        let memory_limit: Option<usize> = extract_optional_field(&obj, "memory_limit")?;
        let bounds: Option<Vec<(Option<Float>, Option<Float>)>> =
            extract_optional_field(&obj, "bounds")?;
        let parameter_names: Option<Vec<String>> = extract_optional_field(&obj, "parameter_names")?;
        let bounds_handling: Option<String> = extract_optional_field(&obj, "bounds_handling")?;
        let line_search: Option<Bound<'py, PyAny>> = extract_optional_field(&obj, "line_search")?;
        let error_mode: Option<String> = extract_optional_field(&obj, "error_mode")?;

        let mut native = LBFGSBConfig::default();
        if let Some(memory_limit) = memory_limit {
            native = native.with_memory_limit(memory_limit)?;
        }
        if let Some(bounds_handling) = bounds_handling {
            native = native.with_bounds_handling(parse_bounds_handling(&bounds_handling)?);
        }
        if let Some(line_search) = line_search {
            native = native.with_line_search(parse_line_search(&line_search)?);
        }
        if let Some(error_mode) = error_mode {
            native = native.with_error_mode(parse_lbfgsb_error_mode(&error_mode)?);
        }
        let native = apply_python_bounds(native, bounds);
        Ok(apply_python_parameter_names(native, parameter_names))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for NelderMeadConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_init__")?;
        let bounds: Option<Vec<(Option<Float>, Option<Float>)>> =
            extract_optional_field(&obj, "bounds")?;
        let parameter_names: Option<Vec<String>> = extract_optional_field(&obj, "parameter_names")?;
        let alpha: Option<Float> = extract_optional_field(&obj, "alpha")?;
        let beta: Option<Float> = extract_optional_field(&obj, "beta")?;
        let gamma: Option<Float> = extract_optional_field(&obj, "gamma")?;
        let delta: Option<Float> = extract_optional_field(&obj, "delta")?;
        let adaptive_dimension: Option<usize> = extract_optional_field(&obj, "adaptive_dimension")?;
        let expansion_method: Option<String> = extract_optional_field(&obj, "expansion_method")?;
        let bounds_handling: Option<String> = extract_optional_field(&obj, "bounds_handling")?;

        let mut native = NelderMeadConfig::default();
        if let Some(alpha) = alpha {
            native = native.with_alpha(alpha)?;
        }
        if let Some(beta) = beta {
            native = native.with_beta(beta)?;
        }
        if let Some(gamma) = gamma {
            native = native.with_gamma(gamma)?;
        }
        if let Some(delta) = delta {
            native = native.with_delta(delta)?;
        }
        if let Some(adaptive_dimension) = adaptive_dimension {
            native = native.with_adaptive(adaptive_dimension)?;
        }
        if let Some(expansion_method) = expansion_method {
            native =
                native.with_expansion_method(parse_simplex_expansion_method(&expansion_method)?);
        }
        if let Some(bounds_handling) = bounds_handling {
            native = native.with_bounds_handling(parse_bounds_handling(&bounds_handling)?);
        }
        let native = apply_python_bounds(native, bounds);
        Ok(apply_python_parameter_names(native, parameter_names))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for NelderMeadInit {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_init__")?;
        let x0: Option<Bound<'py, PyAny>> = extract_optional_field(&obj, "x0")?;
        let construction_method: Option<Bound<'py, PyAny>> =
            extract_optional_field(&obj, "construction_method")?;
        if let Some(construction_method) = construction_method {
            Ok(NelderMeadInit::new_with_method(parse_simplex_construction(
                &construction_method,
            )?))
        } else {
            let x0 = x0.ok_or_else(|| {
                config_error("NelderMeadConfig requires either `x0` or `construction_method`")
            })?;
            Ok(NelderMeadInit::new(extract_vector(&x0)?))
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PSOConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_init__")?;
        let bounds: Option<Vec<(Option<Float>, Option<Float>)>> =
            extract_optional_field(&obj, "bounds")?;
        let parameter_names: Option<Vec<String>> = extract_optional_field(&obj, "parameter_names")?;
        let omega: Option<Float> = extract_optional_field(&obj, "omega")?;
        let c1: Option<Float> = extract_optional_field(&obj, "c1")?;
        let c2: Option<Float> = extract_optional_field(&obj, "c2")?;
        let bounds_handling: Option<String> = extract_optional_field(&obj, "bounds_handling")?;
        let mut native = PSOConfig::default();
        if let Some(omega) = omega {
            native = native.with_omega(omega)?;
        }
        if let Some(c1) = c1 {
            native = native.with_c1(c1)?;
        }
        if let Some(c2) = c2 {
            native = native.with_c2(c2)?;
        }
        if let Some(bounds_handling) = bounds_handling {
            native = native.with_bounds_handling(parse_bounds_handling(&bounds_handling)?);
        }
        let native = apply_python_bounds(native, bounds);
        Ok(apply_python_parameter_names(native, parameter_names))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Swarm {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_init__")?;
        let positions = extract_matrix(&extract_required_field::<Bound<'py, PyAny>>(
            &obj,
            "positions",
        )?)?;
        let topology: Option<String> = extract_optional_field(&obj, "topology")?;
        let update_method: Option<String> = extract_optional_field(&obj, "update_method")?;
        let boundary_method: Option<String> = extract_optional_field(&obj, "boundary_method")?;
        let mut swarm = Swarm::new(SwarmPositionInitializer::Custom(vectors_to_dvectors(
            &positions,
        )));
        if let Some(topology) = topology {
            swarm = swarm.with_topology(parse_swarm_topology(&topology)?);
        }
        if let Some(update_method) = update_method {
            swarm = swarm.with_update_method(parse_swarm_update_method(&update_method)?);
        }
        if let Some(boundary_method) = boundary_method {
            swarm = swarm.with_boundary_method(parse_swarm_boundary_method(&boundary_method)?);
        }
        Ok(swarm)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for AIESConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_init__")?;
        let parameter_names: Option<Vec<String>> = extract_optional_field(&obj, "parameter_names")?;
        let moves: Option<Bound<'py, PyAny>> = extract_optional_field(&obj, "moves")?;
        let chain_storage: Option<Bound<'py, PyAny>> =
            extract_optional_field(&obj, "chain_storage")?;

        let mut native = AIESConfig::default();
        if let Some(moves) = moves {
            native = native.with_moves(parse_aies_moves(&moves)?)?;
        }
        if let Some(chain_storage) = chain_storage {
            native = native.with_chain_storage(parse_chain_storage(&chain_storage)?);
        }
        Ok(apply_python_parameter_names(native, parameter_names))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for AIESInit {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_config__")?;
        let walkers = extract_matrix(&extract_required_field::<Bound<'py, PyAny>>(
            &obj, "walkers",
        )?)?;
        AIESInit::new(vectors_to_dvectors(&walkers)).map_err(Into::into)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ESSConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_config__")?;
        let parameter_names: Option<Vec<String>> = extract_optional_field(&obj, "parameter_names")?;
        let n_adaptive: Option<usize> = extract_optional_field(&obj, "n_adaptive")?;
        let max_steps: Option<usize> = extract_optional_field(&obj, "max_steps")?;
        let mu: Option<Float> = extract_optional_field(&obj, "mu")?;
        let moves: Option<Bound<'py, PyAny>> = extract_optional_field(&obj, "moves")?;
        let chain_storage: Option<Bound<'py, PyAny>> =
            extract_optional_field(&obj, "chain_storage")?;
        let mut native = ESSConfig::default();
        if let Some(moves) = moves {
            native = native.with_moves(parse_ess_moves(&moves)?)?;
        }
        if let Some(n_adaptive) = n_adaptive {
            native = native.with_n_adaptive(n_adaptive);
        }
        if let Some(max_steps) = max_steps {
            native = native.with_max_steps(max_steps);
        }
        if let Some(mu) = mu {
            native = native.with_mu(mu)?;
        }
        if let Some(chain_storage) = chain_storage {
            native = native.with_chain_storage(parse_chain_storage(&chain_storage)?);
        }
        Ok(apply_python_parameter_names(native, parameter_names))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ESSInit {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_config__")?;
        let walkers = extract_matrix(&extract_required_field::<Bound<'py, PyAny>>(
            &obj, "walkers",
        )?)?;
        ESSInit::new(vectors_to_dvectors(&walkers)).map_err(Into::into)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for DifferentialEvolutionConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_config__")?;
        let population_size: Option<usize> = extract_optional_field(&obj, "population_size")?;
        let differential_weight: Option<Float> =
            extract_optional_field(&obj, "differential_weight")?;
        let crossover_probability: Option<Float> =
            extract_optional_field(&obj, "crossover_probability")?;
        let bounds: Option<Vec<(Option<Float>, Option<Float>)>> =
            extract_optional_field(&obj, "bounds")?;
        let parameter_names: Option<Vec<String>> = extract_optional_field(&obj, "parameter_names")?;

        let mut native = DifferentialEvolutionConfig::default();
        if let Some(population_size) = population_size {
            native = native.with_population_size(population_size)?;
        }
        if let Some(differential_weight) = differential_weight {
            native = native.with_differential_weight(differential_weight)?;
        }
        if let Some(crossover_probability) = crossover_probability {
            native = native.with_crossover_probability(crossover_probability)?;
        }
        let native = apply_python_bounds(native, bounds);
        Ok(apply_python_parameter_names(native, parameter_names))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for DifferentialEvolutionInit {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_config__")?;
        let x0 = extract_vector(&extract_required_field::<Bound<'py, PyAny>>(&obj, "x0")?)?;
        let initial_scale: Option<Float> = extract_optional_field(&obj, "initial_scale")?;

        let mut native = DifferentialEvolutionInit::new(&x0)?;
        if let Some(initial_scale) = initial_scale {
            native = native.with_initial_scale(initial_scale)?;
        }
        Ok(native)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for CMAESConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_config__")?;
        let population_size: Option<usize> = extract_optional_field(&obj, "population_size")?;
        let bounds: Option<Vec<(Option<Float>, Option<Float>)>> =
            extract_optional_field(&obj, "bounds")?;
        let parameter_names: Option<Vec<String>> = extract_optional_field(&obj, "parameter_names")?;

        let mut native = CMAESConfig::default();
        if let Some(population_size) = population_size {
            native = native.with_population_size(population_size)?;
        }
        let native = apply_python_bounds(native, bounds);
        Ok(apply_python_parameter_names(native, parameter_names))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for CMAESInit {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_config__")?;
        let x0 = extract_vector(&extract_required_field::<Bound<'py, PyAny>>(&obj, "x0")?)?;
        let sigma: Float = extract_required_field(&obj, "sigma")?;
        CMAESInit::new(&x0, sigma).map_err(Into::into)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for SimulatedAnnealingConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_config__")?;
        let initial_temperature: Option<Float> =
            extract_optional_field(&obj, "initial_temperature")?;
        let cooling_rate: Option<Float> = extract_optional_field(&obj, "cooling_rate")?;

        SimulatedAnnealingConfig::new(
            initial_temperature.unwrap_or(1.0),
            cooling_rate.unwrap_or(0.999),
        )
        .map_err(Into::into)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for AdamConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_config__")?;
        let parameter_names: Option<Vec<String>> = extract_optional_field(&obj, "parameter_names")?;
        let alpha: Option<Float> = extract_optional_field(&obj, "alpha")?;
        let beta_1: Option<Float> = extract_optional_field(&obj, "beta_1")?;
        let beta_2: Option<Float> = extract_optional_field(&obj, "beta_2")?;
        let epsilon: Option<Float> = extract_optional_field(&obj, "epsilon")?;

        let mut native = AdamConfig::default();
        if let Some(alpha) = alpha {
            native = native.with_alpha(alpha)?;
        }
        if let Some(beta_1) = beta_1 {
            native = native.with_beta_1(beta_1)?;
        }
        if let Some(beta_2) = beta_2 {
            native = native.with_beta_2(beta_2)?;
        }
        if let Some(epsilon) = epsilon {
            native = native.with_epsilon(epsilon)?;
        }
        Ok(apply_python_parameter_names(native, parameter_names))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for ConjugateGradientConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_config__")?;
        let parameter_names: Option<Vec<String>> = extract_optional_field(&obj, "parameter_names")?;
        let line_search: Option<Bound<'py, PyAny>> = extract_optional_field(&obj, "line_search")?;
        let update: Option<String> = extract_optional_field(&obj, "update")?;

        let mut native = ConjugateGradientConfig::default();
        if let Some(line_search) = line_search {
            native = native.with_line_search(parse_line_search(&line_search)?);
        }
        if let Some(update) = update {
            native = native.with_update(parse_cg_update(&update)?);
        }
        Ok(apply_python_parameter_names(native, parameter_names))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for TrustRegionConfig {
    type Error = pyo3::PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let obj = resolve_protocol(&obj.to_owned(), "__ganesh_config__")?;
        let parameter_names: Option<Vec<String>> = extract_optional_field(&obj, "parameter_names")?;
        let subproblem: Option<String> = extract_optional_field(&obj, "subproblem")?;
        let initial_radius: Option<Float> = extract_optional_field(&obj, "initial_radius")?;
        let max_radius: Option<Float> = extract_optional_field(&obj, "max_radius")?;
        let eta: Option<Float> = extract_optional_field(&obj, "eta")?;

        let mut native = TrustRegionConfig::default();
        if let Some(subproblem) = subproblem {
            native = native.with_subproblem(parse_trust_region_subproblem(&subproblem)?);
        }
        if let Some(initial_radius) = initial_radius {
            native = native.with_initial_radius(initial_radius)?;
        }
        if let Some(max_radius) = max_radius {
            native = native.with_max_radius(max_radius)?;
        }
        if let Some(eta) = eta {
            native = native.with_eta(eta)?;
        }
        Ok(apply_python_parameter_names(native, parameter_names))
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{
        types::{PyDict, PyList, PyListMethods, PyModule},
        Bound, PyAny, Python,
    };

    use super::*;
    use crate::{
        algorithms::{
            gradient::{Adam, ConjugateGradient, TrustRegion, LBFGSB},
            gradient_free::{
                nelder_mead::NelderMeadInit, CMAESInit, DifferentialEvolution,
                DifferentialEvolutionInit, NelderMead, CMAES,
            },
            mcmc::{aies::AIESInit, ess::ESSInit, AIES, ESS},
            particles::{PSO, Swarm},
        },
        core::{Callbacks, MaxSteps},
        traits::{Algorithm, CostFunction, Gradient, LogDensity},
        DVector,
    };
    use std::{convert::Infallible, ffi::CString};

    fn package_root() -> &'static str {
        concat!(env!("CARGO_MANIFEST_DIR"), "/python")
    }

    fn py_vector<'py>(py: Python<'py>, values: &[Float]) -> Bound<'py, PyAny> {
        PyList::new(py, values).unwrap().into_any()
    }

    fn import_ganesh<'py>(py: Python<'py>) -> Bound<'py, PyAny> {
        let sys = py.import("sys").unwrap();
        sys.getattr("path")
            .unwrap()
            .call_method1("insert", (0, package_root()))
            .unwrap();
        py.import("ganesh").unwrap().into_any()
    }

    struct Quadratic;
    struct GaussianLogDensity;

    impl CostFunction for Quadratic {
        fn evaluate(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(x.dot(x))
        }
    }

    impl Gradient for Quadratic {
        fn gradient(&self, x: &DVector<Float>, _args: &()) -> Result<DVector<Float>, Infallible> {
            Ok(x * 2.0)
        }
    }

    impl LogDensity<(), Infallible> for GaussianLogDensity {
        fn log_density(&self, x: &DVector<Float>, _args: &()) -> Result<Float, Infallible> {
            Ok(-0.5 * x.dot(x))
        }
    }

    #[test]
    fn pure_python_lbfgsb_config_extracts_and_runs() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let config_like = ganesh
                .getattr("LBFGSBConfig")
                .unwrap()
                .call1((py_vector(py, &[2.0, -1.0]),))
                .unwrap();
            config_like.setattr("memory_limit", 5).unwrap();
            config_like
                .setattr(
                    "bounds",
                    vec![(Some(-5.0), Some(5.0)), (Some(-5.0), Some(5.0))],
                )
                .unwrap();
            config_like
                .setattr("parameter_names", vec!["x".to_string(), "y".to_string()])
                .unwrap();

            let config: LBFGSBConfig = config_like.extract().unwrap();
            let summary = LBFGSB::default()
                .process(
                    &Quadratic,
                    &(),
                    DVector::from_row_slice(&[2.0, -1.0]),
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(2)),
                )
                .unwrap();
            assert_eq!(summary.parameter_names.as_ref().unwrap(), &vec!["x", "y"]);
        });
    }

    #[test]
    fn pure_python_lbfgsb_config_extracts_extended_options() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "bounds",
                    vec![(Some(-5.0), Some(5.0)), (Some(-5.0), Some(5.0))],
                )
                .unwrap();
            kwargs
                .set_item("parameter_names", vec!["x".to_string(), "y".to_string()])
                .unwrap();
            kwargs
                .set_item("bounds_handling", "transform_bounds")
                .unwrap();
            kwargs.set_item("error_mode", "skip").unwrap();
            kwargs
                .set_item(
                    "line_search",
                    ganesh
                        .getattr("HagerZhangLineSearch")
                        .unwrap()
                        .call0()
                        .unwrap(),
                )
                .unwrap();
            let config_like = ganesh
                .getattr("LBFGSBConfig")
                .unwrap()
                .call((py_vector(py, &[1.5, -1.0]),), Some(&kwargs))
                .unwrap();

            let config: LBFGSBConfig = config_like.extract().unwrap();
            let summary = LBFGSB::default()
                .process(
                    &Quadratic,
                    &(),
                    DVector::from_row_slice(&[1.5, -1.0]),
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(2)),
                )
                .unwrap();
            assert_eq!(
                summary.parameter_names.as_ref().unwrap(),
                &vec!["x".to_string(), "y".to_string()]
            );
        });
    }

    #[test]
    fn differential_evolution_accepts_dictionary_fallback() {
        crate::python::attach_for_tests(|py| {
            let dict = PyDict::new(py);
            dict.set_item("x0", vec![1.0, -1.0]).unwrap();
            dict.set_item("population_size", 8).unwrap();
            dict.set_item("initial_scale", 0.5).unwrap();

            let init: DifferentialEvolutionInit = dict.as_any().extract().unwrap();
            let config: DifferentialEvolutionConfig = dict.as_any().extract().unwrap();
            let _summary = DifferentialEvolution::default()
                .process(
                    &Quadratic,
                    &(),
                    init,
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(1)),
                )
                .unwrap();
        });
    }

    #[test]
    fn pure_python_cmaes_config_extracts_and_runs() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs.set_item("population_size", 8).unwrap();
            kwargs
                .set_item(
                    "bounds",
                    vec![(Some(-2.0), Some(2.0)), (Some(-2.0), Some(2.0))],
                )
                .unwrap();
            kwargs
                .set_item("parameter_names", vec!["u".to_string(), "v".to_string()])
                .unwrap();
            let config_like = ganesh
                .getattr("CMAESConfig")
                .unwrap()
                .call((py_vector(py, &[0.5, -0.5]), 0.3), Some(&kwargs))
                .unwrap();

            let init: CMAESInit = config_like.extract().unwrap();
            let config: CMAESConfig = config_like.extract().unwrap();
            let summary = CMAES::default()
                .process(
                    &Quadratic,
                    &(),
                    init,
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(1)),
                )
                .unwrap();
            assert_eq!(
                summary.parameter_names.as_ref().unwrap(),
                &vec!["u".to_string(), "v".to_string()]
            );
        });
    }

    #[test]
    fn pure_python_nelder_mead_config_extracts_extended_options() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "construction_method",
                    ganesh
                        .getattr("OrthogonalSimplex")
                        .unwrap()
                        .call1((py_vector(py, &[1.0, -1.0]), 0.5))
                        .unwrap(),
                )
                .unwrap();
            kwargs
                .set_item(
                    "bounds",
                    vec![(Some(-4.0), Some(4.0)), (Some(-4.0), Some(4.0))],
                )
                .unwrap();
            kwargs
                .set_item("parameter_names", vec!["a".to_string(), "b".to_string()])
                .unwrap();
            kwargs.set_item("alpha", 1.2).unwrap();
            kwargs.set_item("beta", 2.4).unwrap();
            kwargs.set_item("gamma", 0.45).unwrap();
            kwargs.set_item("delta", 0.4).unwrap();
            kwargs
                .set_item("expansion_method", "greedy_expansion")
                .unwrap();
            kwargs
                .set_item("bounds_handling", "transform_bounds")
                .unwrap();
            let config_like = ganesh
                .getattr("NelderMeadConfig")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap();

            let init: NelderMeadInit = config_like.extract().unwrap();
            let config: NelderMeadConfig = config_like.extract().unwrap();
            let summary = NelderMead::default()
                .process(
                    &Quadratic,
                    &(),
                    init,
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(1)),
                )
                .unwrap();
            assert_eq!(
                summary.parameter_names.as_ref().unwrap(),
                &vec!["a".to_string(), "b".to_string()]
            );
        });
    }

    #[test]
    fn pure_python_pso_config_extracts_extended_options() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(
                    "bounds",
                    vec![(Some(-2.0), Some(2.0)), (Some(-2.0), Some(2.0))],
                )
                .unwrap();
            kwargs
                .set_item("parameter_names", vec!["x".to_string(), "y".to_string()])
                .unwrap();
            kwargs.set_item("omega", 0.7).unwrap();
            kwargs.set_item("c1", 0.2).unwrap();
            kwargs.set_item("c2", 0.3).unwrap();
            kwargs
                .set_item("bounds_handling", "transform_bounds")
                .unwrap();
            kwargs.set_item("topology", "ring").unwrap();
            kwargs.set_item("update_method", "synchronous").unwrap();
            kwargs.set_item("boundary_method", "shr").unwrap();
            let positions = vec![
                vec![1.0, -1.0],
                vec![0.5, -0.5],
                vec![-1.0, 1.0],
                vec![0.25, 0.75],
            ];
            let config_like = ganesh
                .getattr("PSOConfig")
                .unwrap()
                .call((positions,), Some(&kwargs))
                .unwrap();

            let init: Swarm = config_like.extract().unwrap();
            let config: PSOConfig = config_like.extract().unwrap();
            let summary = PSO::default()
                .process(
                    &Quadratic,
                    &(),
                    init,
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(1)),
                )
                .unwrap();
            assert_eq!(
                summary.parameter_names.as_ref().unwrap(),
                &vec!["x".to_string(), "y".to_string()]
            );
        });
    }

    #[test]
    fn pure_python_aies_config_extracts_moves_and_chain_storage() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            let moves = PyList::empty(py);
            moves
                .append(
                    ganesh
                        .getattr("AIESStretchMove")
                        .unwrap()
                        .call(
                            (),
                            Some(&{
                                let kwargs = PyDict::new(py);
                                kwargs.set_item("weight", 0.4).unwrap();
                                kwargs.set_item("a", 2.5).unwrap();
                                kwargs
                            }),
                        )
                        .unwrap(),
                )
                .unwrap();
            moves
                .append(
                    ganesh
                        .getattr("AIESWalkMove")
                        .unwrap()
                        .call(
                            (),
                            Some(&{
                                let kwargs = PyDict::new(py);
                                kwargs.set_item("weight", 0.6).unwrap();
                                kwargs
                            }),
                        )
                        .unwrap(),
                )
                .unwrap();
            kwargs.set_item("moves", moves).unwrap();
            kwargs
                .set_item(
                    "chain_storage",
                    ganesh
                        .getattr("ChainStorageRolling")
                        .unwrap()
                        .call1((4,))
                        .unwrap(),
                )
                .unwrap();
            kwargs
                .set_item("parameter_names", vec!["a".to_string(), "b".to_string()])
                .unwrap();
            let walkers = vec![
                vec![0.0, 0.0],
                vec![0.1, 0.0],
                vec![0.0, 0.1],
                vec![0.1, 0.1],
            ];
            let config_like = ganesh
                .getattr("AIESConfig")
                .unwrap()
                .call((walkers,), Some(&kwargs))
                .unwrap();

            let init: AIESInit = config_like.extract().unwrap();
            let config: AIESConfig = config_like.extract().unwrap();
            let summary = AIES::default()
                .process(
                    &GaussianLogDensity,
                    &(),
                    init,
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(1)),
                )
                .unwrap();
            assert_eq!(
                summary.parameter_names.as_ref().unwrap(),
                &vec!["a".to_string(), "b".to_string()]
            );
            assert!(matches!(
                summary.chain_storage,
                crate::algorithms::mcmc::ChainStorageMode::Rolling { window: 4 }
            ));
        });
    }

    #[test]
    fn duck_typed_ess_config_extracts() {
        crate::python::attach_for_tests(|py| {
            let code = CString::new(
                "\
class DuckESS:
    def __init__(self):
        self.walkers = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]]
        self.parameter_names = ['a', 'b']
        self.n_adaptive = 2
        self.max_steps = 20
        self.mu = 1.5
",
            )
            .unwrap();
            let filename = CString::new("duck_ess.py").unwrap();
            let module_name = CString::new("duck_ess").unwrap();
            let module = PyModule::from_code(py, &code, &filename, &module_name).unwrap();
            let obj = module.getattr("DuckESS").unwrap().call0().unwrap();
            let init: ESSInit = obj.extract().unwrap();
            let config: ESSConfig = obj.extract().unwrap();
            let _summary = ESS::default()
                .process(
                    &GaussianLogDensity,
                    &(),
                    init,
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(1)),
                )
                .unwrap();
        });
    }

    #[test]
    fn pure_python_ess_config_extracts_moves_chain_storage_and_steps() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            let moves = PyList::empty(py);
            moves
                .append(
                    ganesh
                        .getattr("ESSDifferentialMove")
                        .unwrap()
                        .call(
                            (),
                            Some(&{
                                let kwargs = PyDict::new(py);
                                kwargs.set_item("weight", 0.5).unwrap();
                                kwargs
                            }),
                        )
                        .unwrap(),
                )
                .unwrap();
            moves
                .append(
                    ganesh
                        .getattr("ESSGaussianMove")
                        .unwrap()
                        .call(
                            (),
                            Some(&{
                                let kwargs = PyDict::new(py);
                                kwargs.set_item("weight", 0.5).unwrap();
                                kwargs
                            }),
                        )
                        .unwrap(),
                )
                .unwrap();
            kwargs.set_item("moves", moves).unwrap();
            kwargs.set_item("n_adaptive", 1).unwrap();
            kwargs.set_item("max_steps", 8).unwrap();
            kwargs.set_item("mu", 1.5).unwrap();
            kwargs
                .set_item(
                    "chain_storage",
                    ganesh
                        .getattr("ChainStorageSampled")
                        .unwrap()
                        .call(
                            (2,),
                            Some(&{
                                let kwargs = PyDict::new(py);
                                kwargs.set_item("max_samples", 4).unwrap();
                                kwargs
                            }),
                        )
                        .unwrap(),
                )
                .unwrap();
            kwargs
                .set_item("parameter_names", vec!["a".to_string(), "b".to_string()])
                .unwrap();
            let walkers = vec![
                vec![0.0, 0.0],
                vec![0.1, 0.0],
                vec![0.0, 0.1],
                vec![0.1, 0.1],
            ];
            let config_like = ganesh
                .getattr("ESSConfig")
                .unwrap()
                .call((walkers,), Some(&kwargs))
                .unwrap();

            let init: ESSInit = config_like.extract().unwrap();
            let config: ESSConfig = config_like.extract().unwrap();
            let summary = ESS::default()
                .process(
                    &GaussianLogDensity,
                    &(),
                    init,
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(1)),
                )
                .unwrap();
            assert_eq!(
                summary.parameter_names.as_ref().unwrap(),
                &vec!["a".to_string(), "b".to_string()]
            );
            assert!(matches!(
                summary.chain_storage,
                crate::algorithms::mcmc::ChainStorageMode::Sampled {
                    keep_every: 2,
                    max_samples: Some(4)
                }
            ));
        });
    }

    #[test]
    fn pure_python_adam_config_extracts_and_runs() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("parameter_names", vec!["u".to_string(), "v".to_string()])
                .unwrap();
            kwargs.set_item("alpha", 0.01).unwrap();
            kwargs.set_item("beta_1", 0.8).unwrap();
            kwargs.set_item("beta_2", 0.95).unwrap();
            kwargs.set_item("epsilon", 1e-7).unwrap();
            let config_like = ganesh
                .getattr("AdamConfig")
                .unwrap()
                .call((py_vector(py, &[1.0, -1.0]),), Some(&kwargs))
                .unwrap();

            let x0_obj: Bound<'_, PyAny> = extract_required_field(&config_like, "x0").unwrap();
            let x0 = DVector::from_vec(extract_vector(&x0_obj).unwrap());
            let config: AdamConfig = config_like.extract().unwrap();
            let summary = Adam::default()
                .process(
                    &Quadratic,
                    &(),
                    x0,
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(1)),
                )
                .unwrap();
            assert_eq!(
                summary.parameter_names.as_ref().unwrap(),
                &vec!["u".to_string(), "v".to_string()]
            );
        });
    }

    #[test]
    fn pure_python_conjugate_gradient_config_extracts_and_runs() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("parameter_names", vec!["u".to_string(), "v".to_string()])
                .unwrap();
            kwargs.set_item("update", "hager_zhang").unwrap();
            kwargs
                .set_item(
                    "line_search",
                    ganesh
                        .getattr("MoreThuenteLineSearch")
                        .unwrap()
                        .call0()
                        .unwrap(),
                )
                .unwrap();
            let config_like = ganesh
                .getattr("ConjugateGradientConfig")
                .unwrap()
                .call((py_vector(py, &[1.0, -1.0]),), Some(&kwargs))
                .unwrap();

            let x0_obj: Bound<'_, PyAny> = extract_required_field(&config_like, "x0").unwrap();
            let x0 = DVector::from_vec(extract_vector(&x0_obj).unwrap());
            let config: ConjugateGradientConfig = config_like.extract().unwrap();
            let summary = ConjugateGradient::default()
                .process(
                    &Quadratic,
                    &(),
                    x0,
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(1)),
                )
                .unwrap();
            assert_eq!(
                summary.parameter_names.as_ref().unwrap(),
                &vec!["u".to_string(), "v".to_string()]
            );
        });
    }

    #[test]
    fn pure_python_trust_region_config_extracts_and_runs() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("parameter_names", vec!["u".to_string(), "v".to_string()])
                .unwrap();
            kwargs.set_item("subproblem", "cauchy_point").unwrap();
            kwargs.set_item("initial_radius", 0.5).unwrap();
            kwargs.set_item("max_radius", 2.0).unwrap();
            kwargs.set_item("eta", 1e-3).unwrap();
            let config_like = ganesh
                .getattr("TrustRegionConfig")
                .unwrap()
                .call((py_vector(py, &[1.0, -1.0]),), Some(&kwargs))
                .unwrap();

            let x0_obj: Bound<'_, PyAny> = extract_required_field(&config_like, "x0").unwrap();
            let x0 = DVector::from_vec(extract_vector(&x0_obj).unwrap());
            let config: TrustRegionConfig = config_like.extract().unwrap();
            let summary = TrustRegion::default()
                .process(
                    &Quadratic,
                    &(),
                    x0,
                    config,
                    Callbacks::empty().with_terminator(MaxSteps(1)),
                )
                .unwrap();
            assert_eq!(
                summary.parameter_names.as_ref().unwrap(),
                &vec!["u".to_string(), "v".to_string()]
            );
        });
    }

    #[test]
    fn pure_python_package_imports_without_native_module() {
        crate::python::attach_for_tests(|py| {
            let ganesh = import_ganesh(py);
            assert!(ganesh.hasattr("LBFGSBConfig").unwrap());
            assert!(ganesh.hasattr("CMAESOptions").unwrap());
        });
    }
}
