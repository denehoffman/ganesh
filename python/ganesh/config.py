from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import BoundsLike, FloatMatrixLike, FloatVectorLike


class _GaneshConfigMixin:
    def __ganesh_config__(self):
        return self


class _GaneshInitMixin:
    def __ganesh_init__(self):
        return self


class _GaneshLineSearchMixin:
    def __ganesh_line_search__(self):
        return self


class _GaneshSimplexConstructionMixin:
    def __ganesh_simplex_construction__(self):
        return self


class _GaneshMoveMixin:
    def __ganesh_move__(self):
        return self


class _GaneshChainStorageMixin:
    def __ganesh_chain_storage__(self):
        return self


class MoreThuenteLineSearch(_GaneshLineSearchMixin):
    """
    More-Thuente strong-Wolfe line search setup.

    Parameters
    ----------
    max_iterations : int, default=100
        Maximum number of outer line-search iterations.
    max_zoom : int, default=100
        Maximum number of zoom-phase iterations.
    c1 : float, default=1e-4
        Armijo sufficient-decrease constant.
    c2 : float, default=0.9
        Curvature constant for the strong-Wolfe condition.

    """

    __slots__ = ('c1', 'c2', 'kind', 'max_iterations', 'max_zoom')

    def __init__(
        self,
        max_iterations: int = 100,
        max_zoom: int = 100,
        c1: float = 1e-4,
        c2: float = 0.9,
    ) -> None:
        self.kind = 'more_thuente'
        self.max_iterations = max_iterations
        self.max_zoom = max_zoom
        self.c1 = c1
        self.c2 = c2


class HagerZhangLineSearch(_GaneshLineSearchMixin):
    """
    Hager-Zhang approximate-Wolfe line search setup.

    Parameters
    ----------
    max_iterations : int, default=100
        Maximum number of line-search iterations.
    delta : float, default=0.1
        Lower Wolfe/approximate-Wolfe parameter.
    sigma : float, default=0.9
        Upper Wolfe/approximate-Wolfe parameter.
    epsilon : float, default=cbrt(machine epsilon)
        Approximate-Wolfe tolerance parameter.
    theta : float, default=0.5
        Safeguard parameter used while shrinking the uncertainty interval.
    gamma : float, default=0.66
        Bisection trigger parameter.
    max_bisects : int, default=50
        Maximum number of bisection steps.

    """

    __slots__ = (
        'delta',
        'epsilon',
        'gamma',
        'kind',
        'max_bisects',
        'max_iterations',
        'sigma',
        'theta',
    )

    def __init__(
        self,
        max_iterations: int = 100,
        delta: float = 0.1,
        sigma: float = 0.9,
        epsilon: float = 2.220446049250313e-16 ** (1.0 / 3.0),
        theta: float = 0.5,
        gamma: float = 0.66,
        max_bisects: int = 50,
    ) -> None:
        self.kind = 'hager_zhang'
        self.max_iterations = max_iterations
        self.delta = delta
        self.sigma = sigma
        self.epsilon = epsilon
        self.theta = theta
        self.gamma = gamma
        self.max_bisects = max_bisects


class ScaledOrthogonalSimplex(_GaneshSimplexConstructionMixin):
    """
    Scaled orthogonal simplex construction for Nelder-Mead.

    Parameters
    ----------
    x0 : FloatVectorLike
        Initial point around which the simplex is constructed.
    orthogonal_multiplier : float, default=1.05
        Multiplicative perturbation applied to nonzero coordinates.
    orthogonal_zero_step : float, default=0.00025
        Additive perturbation used when a coordinate of ``x0`` is zero.

    """

    __slots__ = ('kind', 'orthogonal_multiplier', 'orthogonal_zero_step', 'x0')

    def __init__(
        self,
        x0: FloatVectorLike,
        orthogonal_multiplier: float = 1.05,
        orthogonal_zero_step: float = 0.00025,
    ) -> None:
        self.kind = 'scaled_orthogonal'
        self.x0 = x0
        self.orthogonal_multiplier = orthogonal_multiplier
        self.orthogonal_zero_step = orthogonal_zero_step


class OrthogonalSimplex(_GaneshSimplexConstructionMixin):
    """
    Fixed-size orthogonal simplex construction for Nelder-Mead.

    Parameters
    ----------
    x0 : FloatVectorLike
        Initial point around which the simplex is constructed.
    simplex_size : float, default=1.0
        Edge scale used to build the orthogonal simplex.

    """

    __slots__ = ('kind', 'simplex_size', 'x0')

    def __init__(self, x0: FloatVectorLike, simplex_size: float = 1.0) -> None:
        self.kind = 'orthogonal'
        self.x0 = x0
        self.simplex_size = simplex_size


class CustomSimplex(_GaneshSimplexConstructionMixin):
    """
    Explicit simplex construction for Nelder-Mead.

    Parameters
    ----------
    simplex : FloatMatrixLike
        Full simplex with shape ``(n + 1, n)`` in external coordinates.

    """

    __slots__ = ('kind', 'simplex')

    def __init__(self, simplex: FloatMatrixLike) -> None:
        self.kind = 'custom'
        self.simplex = simplex


class ChainStorageFull(_GaneshChainStorageMixin):
    """Retain every stored MCMC state for every walker."""

    __slots__ = ('kind',)

    def __init__(self) -> None:
        self.kind = 'full'


class ChainStorageRolling(_GaneshChainStorageMixin):
    """
    Rolling MCMC chain storage.

    Parameters
    ----------
    window : int
        Number of most recent samples retained for each walker.

    """

    __slots__ = ('kind', 'window')

    def __init__(self, window: int) -> None:
        self.kind = 'rolling'
        self.window = window


class ChainStorageSampled(_GaneshChainStorageMixin):
    """
    Subsampled MCMC chain storage.

    Parameters
    ----------
    keep_every : int
        Retain every ``keep_every``-th sample.
    max_samples : int | None, default=None
        Optional cap on the number of retained samples per walker.

    """

    __slots__ = ('keep_every', 'kind', 'max_samples')

    def __init__(self, keep_every: int, max_samples: int | None = None) -> None:
        self.kind = 'sampled'
        self.keep_every = keep_every
        self.max_samples = max_samples


class AIESStretchMove(_GaneshMoveMixin):
    """
    Stretch move for the affine-invariant ensemble sampler.

    Parameters
    ----------
    weight : float, default=1.0
        Relative selection weight when multiple moves are provided.
    a : float, default=2.0
        Stretch parameter controlling the proposal scale.

    """

    __slots__ = ('a', 'kind', 'weight')

    def __init__(self, weight: float = 1.0, a: float = 2.0) -> None:
        self.kind = 'stretch'
        self.weight = weight
        self.a = a


class AIESWalkMove(_GaneshMoveMixin):
    """
    Walk move for the affine-invariant ensemble sampler.

    Parameters
    ----------
    weight : float, default=1.0
        Relative selection weight when multiple moves are provided.

    """

    __slots__ = ('kind', 'weight')

    def __init__(self, weight: float = 1.0) -> None:
        self.kind = 'walk'
        self.weight = weight


class ESSDifferentialMove(_GaneshMoveMixin):
    """
    Differential move for ensemble slice sampling.

    Parameters
    ----------
    weight : float, default=1.0
        Relative selection weight when multiple moves are provided.

    """

    __slots__ = ('kind', 'weight')

    def __init__(self, weight: float = 1.0) -> None:
        self.kind = 'differential'
        self.weight = weight


class ESSGaussianMove(_GaneshMoveMixin):
    """
    Gaussian move for ensemble slice sampling.

    Parameters
    ----------
    weight : float, default=1.0
        Relative selection weight when multiple moves are provided.

    """

    __slots__ = ('kind', 'weight')

    def __init__(self, weight: float = 1.0) -> None:
        self.kind = 'gaussian'
        self.weight = weight


class ESSGlobalMove(_GaneshMoveMixin):
    """
    Global Gaussian-mixture move for ensemble slice sampling.

    Parameters
    ----------
    weight : float, default=1.0
        Relative selection weight when multiple moves are provided.
    scale : float, default=1.0
        Global scale factor applied to the proposal covariance.
    rescale_cov : float, default=0.001
        Diagonal covariance regularization term.
    n_components : int, default=5
        Number of Gaussian mixture components used in the proposal model.

    """

    __slots__ = ('kind', 'n_components', 'rescale_cov', 'scale', 'weight')

    def __init__(
        self,
        weight: float = 1.0,
        scale: float = 1.0,
        rescale_cov: float = 0.001,
        n_components: int = 5,
    ) -> None:
        self.kind = 'global'
        self.weight = weight
        self.scale = scale
        self.rescale_cov = rescale_cov
        self.n_components = n_components


class LBFGSBConfig(_GaneshConfigMixin):
    """
    Configuration for the L-BFGS-B optimizer.

    Parameters
    ----------
    memory_limit : int, default=10
        Number of correction pairs retained by the limited-memory update.
    bounds : BoundsLike | None, default=None
        Optional parameter bounds as ``(lower, upper)`` pairs. Use ``None`` for
        an unbounded side.
    parameter_names : list[str] | None, default=None
        Optional names used in summaries and diagnostics.
    bounds_handling : str | None, default=None
        Optional bounds-handling mode. ``None`` uses the library default.
    line_search : MoreThuenteLineSearch | HagerZhangLineSearch | None, default=None
        Optional line-search setup. ``None`` uses the Rust-side default line
        search.
    error_mode : str | None, default=None
        Optional Hessian/covariance error-estimation mode.

    """

    __slots__ = (
        'bounds',
        'bounds_handling',
        'error_mode',
        'line_search',
        'memory_limit',
        'parameter_names',
    )

    def __init__(
        self,
        memory_limit: int = 10,
        bounds: BoundsLike | None = None,
        parameter_names: list[str] | None = None,
        bounds_handling: str | None = None,
        line_search: MoreThuenteLineSearch | HagerZhangLineSearch | None = None,
        error_mode: str | None = None,
    ) -> None:
        self.memory_limit = memory_limit
        self.bounds = bounds
        self.parameter_names = parameter_names
        self.bounds_handling = bounds_handling
        self.line_search = line_search
        self.error_mode = error_mode


class NelderMeadInit(_GaneshInitMixin):
    """
    Initialization payload for the Nelder-Mead optimizer.

    Parameters
    ----------
    x0 : FloatVectorLike | None, default=None
        Initial point used when ``construction_method`` is not provided.
    construction_method : ScaledOrthogonalSimplex | OrthogonalSimplex | CustomSimplex | None, default=None
        Optional simplex construction strategy.

    """

    __slots__ = ('construction_method', 'x0')

    def __init__(
        self,
        x0: FloatVectorLike | None = None,
        construction_method: ScaledOrthogonalSimplex | OrthogonalSimplex | CustomSimplex | None = None,
    ) -> None:
        if x0 is not None and construction_method is not None:
            msg = 'NelderMeadInit accepts either x0 or construction_method, not both'
            raise ValueError(msg)
        if x0 is None and construction_method is None:
            msg = 'NelderMeadInit requires either x0 or construction_method'
            raise ValueError(msg)
        self.x0 = x0
        self.construction_method = construction_method


class NelderMeadConfig(_GaneshConfigMixin):
    """
    Configuration for the Nelder-Mead optimizer.

    Parameters
    ----------
    bounds : BoundsLike | None, default=None
        Optional parameter bounds as ``(lower, upper)`` pairs.
    parameter_names : list[str] | None, default=None
        Optional names used in summaries and diagnostics.
    alpha, beta, gamma, delta : float | None, default=None
        Optional reflection, contraction, expansion, and shrink coefficients.
        ``None`` uses the library defaults.
    adaptive_dimension : int | None, default=None
        Optional dimension override for adaptive coefficient schemes.
    expansion_method : str | None, default=None
        Optional expansion rule selector.
    bounds_handling : str | None, default=None
        Optional bounds-handling mode.

    """

    __slots__ = (
        'adaptive_dimension',
        'alpha',
        'beta',
        'bounds',
        'bounds_handling',
        'delta',
        'expansion_method',
        'gamma',
        'parameter_names',
    )

    def __init__(
        self,
        bounds: BoundsLike | None = None,
        parameter_names: list[str] | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        delta: float | None = None,
        adaptive_dimension: int | None = None,
        expansion_method: str | None = None,
        bounds_handling: str | None = None,
    ) -> None:
        self.bounds = bounds
        self.parameter_names = parameter_names
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.adaptive_dimension = adaptive_dimension
        self.expansion_method = expansion_method
        self.bounds_handling = bounds_handling


class PSOInit(_GaneshInitMixin):
    """
    Initialization payload for particle swarm optimization.

    Parameters
    ----------
    positions : FloatMatrixLike
        Initial particle positions with shape ``(n_particles, n_dim)``.
    topology : str | None, default=None
        Optional swarm-topology selector.
    update_method : str | None, default=None
        Optional velocity update strategy.
    boundary_method : str | None, default=None
        Optional particle-boundary response strategy.

    """

    __slots__ = ('boundary_method', 'positions', 'topology', 'update_method')

    def __init__(
        self,
        positions: FloatMatrixLike,
        topology: str | None = None,
        update_method: str | None = None,
        boundary_method: str | None = None,
    ) -> None:
        self.positions = positions
        self.topology = topology
        self.update_method = update_method
        self.boundary_method = boundary_method


class PSOConfig(_GaneshConfigMixin):
    """
    Configuration for particle swarm optimization.

    Parameters
    ----------
    bounds : BoundsLike | None, default=None
        Optional parameter bounds as ``(lower, upper)`` pairs.
    parameter_names : list[str] | None, default=None
        Optional names used in summaries and diagnostics.
    omega : float, default=0.8
        Inertia coefficient.
    c1 : float, default=0.1
        Cognitive acceleration coefficient.
    c2 : float, default=0.1
        Social acceleration coefficient.
    bounds_handling : str | None, default=None
        Optional bounds-handling mode.

    """

    __slots__ = ('bounds', 'bounds_handling', 'c1', 'c2', 'omega', 'parameter_names')

    def __init__(
        self,
        bounds: BoundsLike | None = None,
        parameter_names: list[str] | None = None,
        omega: float = 0.8,
        c1: float = 0.1,
        c2: float = 0.1,
        bounds_handling: str | None = None,
    ) -> None:
        self.bounds = bounds
        self.parameter_names = parameter_names
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.bounds_handling = bounds_handling


class AIESInit(_GaneshInitMixin):
    """
    Initialization payload for the affine-invariant ensemble sampler.

    Parameters
    ----------
    walkers : FloatMatrixLike
        Initial walker positions with shape ``(n_walkers, n_dim)``.

    """

    __slots__ = ('walkers',)

    def __init__(self, walkers: FloatMatrixLike) -> None:
        self.walkers = walkers


class AIESConfig(_GaneshConfigMixin):
    """
    Configuration for the affine-invariant ensemble sampler.

    Parameters
    ----------
    parameter_names : list[str] | None, default=None
        Optional names used in summaries and diagnostics.
    moves : list[AIESStretchMove | AIESWalkMove] | None, default=None
        Optional move mixture. ``None`` uses the library default move set.
    chain_storage : ChainStorageFull | ChainStorageRolling | ChainStorageSampled | None, default=None
        Optional retained-chain storage policy.

    """

    __slots__ = ('chain_storage', 'moves', 'parameter_names')

    def __init__(
        self,
        parameter_names: list[str] | None = None,
        moves: list[AIESStretchMove | AIESWalkMove] | None = None,
        chain_storage: ChainStorageFull | ChainStorageRolling | ChainStorageSampled | None = None,
    ) -> None:
        self.parameter_names = parameter_names
        self.moves = moves
        self.chain_storage = chain_storage


class ESSInit(_GaneshInitMixin):
    """
    Initialization payload for ensemble slice sampling.

    Parameters
    ----------
    walkers : FloatMatrixLike
        Initial walker positions with shape ``(n_walkers, n_dim)``.

    """

    __slots__ = ('walkers',)

    def __init__(self, walkers: FloatMatrixLike) -> None:
        self.walkers = walkers


class ESSConfig(_GaneshConfigMixin):
    """
    Configuration for ensemble slice sampling.

    Parameters
    ----------
    parameter_names : list[str] | None, default=None
        Optional names used in summaries and diagnostics.
    moves : list[ESSDifferentialMove | ESSGaussianMove | ESSGlobalMove] | None, default=None
        Optional move mixture. ``None`` uses the library default move set.
    n_adaptive : int, default=0
        Number of adaptive warmup steps for the internal proposal machinery.
    max_steps : int, default=10000
        Internal step budget used by the ESS transition machinery.
    mu : float, default=1.0
        Scale parameter for the slice expansion.
    chain_storage : ChainStorageFull | ChainStorageRolling | ChainStorageSampled | None, default=None
        Optional retained-chain storage policy.

    """

    __slots__ = (
        'chain_storage',
        'max_steps',
        'moves',
        'mu',
        'n_adaptive',
        'parameter_names',
    )

    def __init__(
        self,
        parameter_names: list[str] | None = None,
        moves: list[ESSDifferentialMove | ESSGaussianMove | ESSGlobalMove] | None = None,
        n_adaptive: int = 0,
        max_steps: int = 10000,
        mu: float = 1.0,
        chain_storage: ChainStorageFull | ChainStorageRolling | ChainStorageSampled | None = None,
    ) -> None:
        self.parameter_names = parameter_names
        self.moves = moves
        self.n_adaptive = n_adaptive
        self.max_steps = max_steps
        self.mu = mu
        self.chain_storage = chain_storage


class DifferentialEvolutionInit(_GaneshInitMixin):
    """
    Initialization payload for differential evolution.

    Parameters
    ----------
    x0 : FloatVectorLike
        Central point used to initialize the population in external
        coordinates.
    initial_scale : float, default=1.0
        Initial population spread in external coordinates.

    """

    __slots__ = ('initial_scale', 'x0')

    def __init__(self, x0: FloatVectorLike, initial_scale: float = 1.0) -> None:
        self.x0 = x0
        self.initial_scale = initial_scale


class DifferentialEvolutionConfig(_GaneshConfigMixin):
    """
    Configuration for differential evolution.

    Parameters
    ----------
    population_size : int | None, default=None
        Optional population size. ``None`` uses the library default.
    differential_weight : float, default=0.8
        Mutation scaling factor.
    crossover_probability : float, default=0.9
        Binomial crossover probability.
    bounds : BoundsLike | None, default=None
        Optional parameter bounds as ``(lower, upper)`` pairs.
    parameter_names : list[str] | None, default=None
        Optional names used in summaries and diagnostics.

    """

    __slots__ = (
        'bounds',
        'crossover_probability',
        'differential_weight',
        'parameter_names',
        'population_size',
    )

    def __init__(
        self,
        population_size: int | None = None,
        differential_weight: float = 0.8,
        crossover_probability: float = 0.9,
        bounds: BoundsLike | None = None,
        parameter_names: list[str] | None = None,
    ) -> None:
        self.population_size = population_size
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.bounds = bounds
        self.parameter_names = parameter_names


class CMAESInit(_GaneshInitMixin):
    """
    Initialization payload for CMA-ES.

    Parameters
    ----------
    x0 : FloatVectorLike
        Initial mean of the search distribution in external coordinates.
    sigma : float
        Initial global step size.

    """

    __slots__ = ('sigma', 'x0')

    def __init__(self, x0: FloatVectorLike, sigma: float) -> None:
        self.x0 = x0
        self.sigma = sigma


class CMAESConfig(_GaneshConfigMixin):
    """
    Configuration for CMA-ES.

    Parameters
    ----------
    population_size : int | None, default=None
        Optional offspring population size. ``None`` uses the library default.
    bounds : BoundsLike | None, default=None
        Optional parameter bounds as ``(lower, upper)`` pairs.
    parameter_names : list[str] | None, default=None
        Optional names used in summaries and diagnostics.

    """

    __slots__ = ('bounds', 'parameter_names', 'population_size')

    def __init__(
        self,
        population_size: int | None = None,
        bounds: BoundsLike | None = None,
        parameter_names: list[str] | None = None,
    ) -> None:
        self.population_size = population_size
        self.bounds = bounds
        self.parameter_names = parameter_names


class SimulatedAnnealingConfig(_GaneshConfigMixin):
    """
    Configuration for simulated annealing.

    Parameters
    ----------
    initial_temperature : float, default=1.0
        Starting temperature of the annealing schedule.
    cooling_rate : float, default=0.999
        Per-step multiplicative cooling factor.

    """

    __slots__ = ('cooling_rate', 'initial_temperature')

    def __init__(
        self,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.999,
    ) -> None:
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate


class AdamConfig(_GaneshConfigMixin):
    """
    Configuration for the Adam optimizer.

    Parameters
    ----------
    parameter_names : list[str] | None, default=None
        Optional names used in summaries and diagnostics.
    alpha : float, default=0.001
        Base learning rate.
    beta_1 : float, default=0.9
        Exponential moving-average coefficient for first moments.
    beta_2 : float, default=0.999
        Exponential moving-average coefficient for second moments.
    epsilon : float, default=1e-8
        Numerical stabilizer added to the denominator.

    """

    __slots__ = ('alpha', 'beta_1', 'beta_2', 'epsilon', 'parameter_names')

    def __init__(
        self,
        parameter_names: list[str] | None = None,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.parameter_names = parameter_names
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon


class ConjugateGradientConfig(_GaneshConfigMixin):
    """
    Configuration for nonlinear conjugate gradient.

    Parameters
    ----------
    parameter_names : list[str] | None, default=None
        Optional names used in summaries and diagnostics.
    line_search : MoreThuenteLineSearch | HagerZhangLineSearch | None, default=None
        Optional line-search setup. ``None`` uses the library default.
    update : str | None, default=None
        Optional beta-update formula selector.

    """

    __slots__ = ('line_search', 'parameter_names', 'update')

    def __init__(
        self,
        parameter_names: list[str] | None = None,
        line_search: MoreThuenteLineSearch | HagerZhangLineSearch | None = None,
        update: str | None = None,
    ) -> None:
        self.parameter_names = parameter_names
        self.line_search = line_search
        self.update = update


class TrustRegionConfig(_GaneshConfigMixin):
    """
    Configuration for trust-region optimization.

    Parameters
    ----------
    parameter_names : list[str] | None, default=None
        Optional names used in summaries and diagnostics.
    subproblem : str | None, default=None
        Optional trust-region subproblem solver selector.
    initial_radius : float, default=1.0
        Initial trust-region radius.
    max_radius : float, default=1000.0
        Maximum trust-region radius.
    eta : float, default=1e-4
        Acceptance threshold on the ratio of actual to predicted reduction.

    """

    __slots__ = (
        'eta',
        'initial_radius',
        'max_radius',
        'parameter_names',
        'subproblem',
    )

    def __init__(
        self,
        parameter_names: list[str] | None = None,
        subproblem: str | None = None,
        initial_radius: float = 1.0,
        max_radius: float = 1000.0,
        eta: float = 1e-4,
    ) -> None:
        self.parameter_names = parameter_names
        self.subproblem = subproblem
        self.initial_radius = initial_radius
        self.max_radius = max_radius
        self.eta = eta


__all__ = [
    'AIESConfig',
    'AIESInit',
    'AIESStretchMove',
    'AIESWalkMove',
    'AdamConfig',
    'CMAESConfig',
    'CMAESInit',
    'ChainStorageFull',
    'ChainStorageRolling',
    'ChainStorageSampled',
    'ConjugateGradientConfig',
    'CustomSimplex',
    'DifferentialEvolutionConfig',
    'DifferentialEvolutionInit',
    'ESSConfig',
    'ESSDifferentialMove',
    'ESSGaussianMove',
    'ESSGlobalMove',
    'ESSInit',
    'HagerZhangLineSearch',
    'LBFGSBConfig',
    'MoreThuenteLineSearch',
    'NelderMeadConfig',
    'NelderMeadInit',
    'OrthogonalSimplex',
    'PSOConfig',
    'PSOInit',
    'ScaledOrthogonalSimplex',
    'SimulatedAnnealingConfig',
    'TrustRegionConfig',
]
