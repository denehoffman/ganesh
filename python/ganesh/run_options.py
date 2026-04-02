from __future__ import annotations

from sys import float_info

from ._defaults import Default, DefaultType, default_to_factory

SQRT_EPS = float_info.epsilon**0.5
CBRT_EPS = float_info.epsilon ** (1.0 / 3.0)
QUARTER_EPS = float_info.epsilon**0.25


class _GaneshRunOptionsMixin:
    def __ganesh_run_options__(self):
        return self


class _GaneshTerminatorMixin:
    def __ganesh_terminator__(self):
        return self


def _default_or_none_sequence(value, factory):
    if value is Default:
        return [factory()]
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class AutocorrelationTerminator(_GaneshTerminatorMixin):
    """
    Integrated-autocorrelation termination rule for MCMC runs.

    Parameters
    ----------
    n_check : int, default=50
        Check the autocorrelation estimate every ``n_check`` retained steps.
    n_taus_threshold : int, default=50
        Minimum retained-chain length in integrated autocorrelation times before
        termination can trigger.
    dtau_threshold : float, default=0.01
        Relative stability threshold for successive autocorrelation estimates.
    discard : float, default=0.5
        Fraction of the retained chain discarded before estimating
        autocorrelation time.
    terminate : bool, default=True
        Whether meeting the criterion should stop the run.
    sokal_window : float | None, default=None
        Optional fixed Sokal window multiplier.
    verbose : bool, default=False
        Whether to emit diagnostic output while estimating autocorrelation time.

    """

    __slots__ = (
        'discard',
        'dtau_threshold',
        'n_check',
        'n_taus_threshold',
        'sokal_window',
        'terminate',
        'verbose',
    )

    def __init__(
        self,
        *,
        n_check: int = 50,
        n_taus_threshold: int = 50,
        dtau_threshold: float = 0.01,
        discard: float = 0.5,
        terminate: bool = True,
        sokal_window: float | None = None,
        verbose: bool = False,
    ) -> None:
        self.n_check = n_check
        self.n_taus_threshold = n_taus_threshold
        self.dtau_threshold = dtau_threshold
        self.discard = discard
        self.terminate = terminate
        self.sokal_window = sokal_window
        self.verbose = verbose


class LBFGSBFTerminator(_GaneshTerminatorMixin):
    """
    Function-tolerance terminator for L-BFGS-B.

    Parameters
    ----------
    eps_abs : float, default=sqrt(machine epsilon)
        Absolute tolerance on the function-value stopping rule.

    """

    __slots__ = ('eps_abs',)

    def __init__(self, *, eps_abs: float = SQRT_EPS) -> None:
        self.eps_abs = eps_abs


class LBFGSBGTerminator(_GaneshTerminatorMixin):
    """
    Gradient-norm terminator for L-BFGS-B.

    Parameters
    ----------
    eps_abs : float, default=cbrt(machine epsilon)
        Absolute tolerance on the gradient stopping rule.

    """

    __slots__ = ('eps_abs',)

    def __init__(self, *, eps_abs: float = CBRT_EPS) -> None:
        self.eps_abs = eps_abs


class LBFGSBInfNormGTerminator(_GaneshTerminatorMixin):
    """
    Projected-gradient infinity-norm terminator for L-BFGS-B.

    Parameters
    ----------
    eps_abs : float, default=cbrt(machine epsilon)
        Absolute tolerance on the projected-gradient infinity norm.

    """

    __slots__ = ('eps_abs',)

    def __init__(self, *, eps_abs: float = CBRT_EPS) -> None:
        self.eps_abs = eps_abs


class NelderMeadAmoebaFTerminator(_GaneshTerminatorMixin):
    """
    Amoeba-style function tolerance for Nelder-Mead.

    Parameters
    ----------
    eps_rel : float, default=quartic_root(machine epsilon)
        Relative tolerance used by the amoeba stopping rule.

    """

    __slots__ = ('eps_rel', 'kind')

    def __init__(self, *, eps_rel: float = QUARTER_EPS) -> None:
        self.kind = 'amoeba'
        self.eps_rel = eps_rel


class NelderMeadAbsoluteFTerminator(_GaneshTerminatorMixin):
    """
    Absolute function-value spread tolerance for Nelder-Mead.

    Parameters
    ----------
    eps_abs : float, default=quartic_root(machine epsilon)
        Absolute tolerance on the simplex function spread.

    """

    __slots__ = ('eps_abs', 'kind')

    def __init__(self, *, eps_abs: float = QUARTER_EPS) -> None:
        self.kind = 'absolute'
        self.eps_abs = eps_abs


class NelderMeadStdDevFTerminator(_GaneshTerminatorMixin):
    """
    Function standard-deviation tolerance for Nelder-Mead.

    Parameters
    ----------
    eps_abs : float, default=quartic_root(machine epsilon)
        Absolute tolerance on the standard deviation of simplex function
        values.

    """

    __slots__ = ('eps_abs', 'kind')

    def __init__(self, *, eps_abs: float = QUARTER_EPS) -> None:
        self.kind = 'stddev'
        self.eps_abs = eps_abs


class NelderMeadDiameterXTerminator(_GaneshTerminatorMixin):
    """
    Simplex-diameter tolerance for Nelder-Mead.

    Parameters
    ----------
    eps_abs : float, default=quartic_root(machine epsilon)
        Absolute tolerance on the maximum simplex diameter.

    """

    __slots__ = ('eps_abs', 'kind')

    def __init__(self, *, eps_abs: float = QUARTER_EPS) -> None:
        self.kind = 'diameter'
        self.eps_abs = eps_abs


class NelderMeadHighamXTerminator(_GaneshTerminatorMixin):
    """
    Higham simplex-size tolerance for Nelder-Mead.

    Parameters
    ----------
    eps_rel : float, default=quartic_root(machine epsilon)
        Relative tolerance for the Higham simplex-size criterion.

    """

    __slots__ = ('eps_rel', 'kind')

    def __init__(self, *, eps_rel: float = QUARTER_EPS) -> None:
        self.kind = 'higham'
        self.eps_rel = eps_rel


class NelderMeadRowanXTerminator(_GaneshTerminatorMixin):
    """
    Rowan simplex-size tolerance for Nelder-Mead.

    Parameters
    ----------
    eps_rel : float, default=quartic_root(machine epsilon)
        Relative tolerance for the Rowan simplex-size criterion.

    """

    __slots__ = ('eps_rel', 'kind')

    def __init__(self, *, eps_rel: float = QUARTER_EPS) -> None:
        self.kind = 'rowan'
        self.eps_rel = eps_rel


class NelderMeadSingerXTerminator(_GaneshTerminatorMixin):
    """
    Singer simplex-size tolerance for Nelder-Mead.

    Parameters
    ----------
    eps_rel : float, default=quartic_root(machine epsilon)
        Relative tolerance for the Singer simplex-size criterion.

    """

    __slots__ = ('eps_rel', 'kind')

    def __init__(self, *, eps_rel: float = QUARTER_EPS) -> None:
        self.kind = 'singer'
        self.eps_rel = eps_rel


class AdamEMATerminator(_GaneshTerminatorMixin):
    """
    EMA-based convergence test for Adam.

    Parameters
    ----------
    beta_c : float, default=0.9
        Exponential moving-average coefficient for the monitored loss signal.
    eps_loss : float, default=sqrt(machine epsilon)
        Absolute loss-change tolerance.
    patience : int, default=1
        Number of consecutive satisfied checks required before termination.

    """

    __slots__ = ('beta_c', 'eps_loss', 'patience')

    def __init__(
        self,
        *,
        beta_c: float = 0.9,
        eps_loss: float = SQRT_EPS,
        patience: int = 1,
    ) -> None:
        self.beta_c = beta_c
        self.eps_loss = eps_loss
        self.patience = patience


class ConjugateGradientGTerminator(_GaneshTerminatorMixin):
    """
    Gradient-norm terminator for nonlinear conjugate gradient.

    Parameters
    ----------
    eps_abs : float, default=cbrt(machine epsilon)
        Absolute tolerance on the gradient norm.

    """

    __slots__ = ('eps_abs',)

    def __init__(self, *, eps_abs: float = CBRT_EPS) -> None:
        self.eps_abs = eps_abs


class TrustRegionGTerminator(_GaneshTerminatorMixin):
    """
    Gradient-norm terminator for trust-region optimization.

    Parameters
    ----------
    eps_abs : float, default=cbrt(machine epsilon)
        Absolute tolerance on the gradient norm.

    """

    __slots__ = ('eps_abs',)

    def __init__(self, *, eps_abs: float = CBRT_EPS) -> None:
        self.eps_abs = eps_abs


class SimulatedAnnealingTemperatureTerminator(_GaneshTerminatorMixin):
    """
    Minimum-temperature terminator for simulated annealing.

    Parameters
    ----------
    min_temperature : float, default=1e-3
        Stop once the annealing temperature falls below this threshold.

    """

    __slots__ = ('min_temperature',)

    def __init__(self, *, min_temperature: float = 1e-3) -> None:
        self.min_temperature = min_temperature


class CMAESSigmaTerminator(_GaneshTerminatorMixin):
    """
    Global step-size terminator for CMA-ES.

    Parameters
    ----------
    eps_abs : float, default=1e-10
        Stop once the global step size falls below this threshold.

    """

    __slots__ = ('eps_abs',)

    def __init__(self, *, eps_abs: float = 1e-10) -> None:
        self.eps_abs = eps_abs


class CMAESNoEffectAxisTerminator(_GaneshTerminatorMixin):
    """No-effect-axis terminator for CMA-ES."""

    __slots__ = ()


class CMAESNoEffectCoordTerminator(_GaneshTerminatorMixin):
    """No-effect-coordinate terminator for CMA-ES."""

    __slots__ = ()


class CMAESConditionCovTerminator(_GaneshTerminatorMixin):
    """
    Covariance-condition-number terminator for CMA-ES.

    Parameters
    ----------
    max_condition : float, default=1e14
        Maximum allowed covariance-matrix condition number.

    """

    __slots__ = ('max_condition',)

    def __init__(self, *, max_condition: float = 1e14) -> None:
        self.max_condition = max_condition


class CMAESEqualFunValuesTerminator(_GaneshTerminatorMixin):
    """Equal-function-values terminator for CMA-ES."""

    __slots__ = ()


class CMAESStagnationTerminator(_GaneshTerminatorMixin):
    """Stagnation-history terminator for CMA-ES."""

    __slots__ = ()


class CMAESTolXUpTerminator(_GaneshTerminatorMixin):
    """
    Step-size explosion terminator for CMA-ES.

    Parameters
    ----------
    max_growth : float, default=1e4
        Maximum allowed growth factor in the effective search scale.

    """

    __slots__ = ('max_growth',)

    def __init__(self, *, max_growth: float = 1e4) -> None:
        self.max_growth = max_growth


class CMAESTolFunTerminator(_GaneshTerminatorMixin):
    """
    Function-range terminator for CMA-ES.

    Parameters
    ----------
    eps_abs : float, default=1e-12
        Absolute tolerance on the recent best-value range.

    """

    __slots__ = ('eps_abs',)

    def __init__(self, *, eps_abs: float = 1e-12) -> None:
        self.eps_abs = eps_abs


class CMAESTolXTerminator(_GaneshTerminatorMixin):
    """
    Coordinate-scale terminator for CMA-ES.

    Parameters
    ----------
    eps_abs : float, default=0.0
        Absolute lower bound on effective coordinate scales.

    """

    __slots__ = ('eps_abs',)

    def __init__(self, *, eps_abs: float = 0.0) -> None:
        self.eps_abs = eps_abs


class LBFGSBOptions(_GaneshRunOptionsMixin):
    """
    Built-in observer and terminator selection for L-BFGS-B.

    Parameters
    ----------
    max_steps : int | None, default=None
        Optional hard iteration limit.
    debug : bool, default=False
        Add the built-in debug observer.
    progress_every : int | None, default=None
        Add the built-in progress observer every ``progress_every`` steps.
    f_tolerance : LBFGSBFTerminator | DefaultType | None, default=Default
        Function-value terminator. Pass ``Default`` to construct the library
        default terminator, or ``None`` to omit it.
    g_tolerance : LBFGSBGTerminator | DefaultType | None, default=Default
        Gradient-norm terminator. Pass ``Default`` to construct the library
        default terminator, or ``None`` to omit it.
    projected_gradient_tolerance : LBFGSBInfNormGTerminator | DefaultType | None, default=Default
        Projected-gradient terminator. Pass ``Default`` to construct the
        library default terminator, or ``None`` to omit it.

    """

    __slots__ = (
        'debug',
        'f_tolerance',
        'g_tolerance',
        'max_steps',
        'progress_every',
        'projected_gradient_tolerance',
    )

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        debug: bool = False,
        progress_every: int | None = None,
        f_tolerance: LBFGSBFTerminator | DefaultType | None = Default,
        g_tolerance: LBFGSBGTerminator | DefaultType | None = Default,
        projected_gradient_tolerance: LBFGSBInfNormGTerminator | DefaultType | None = Default,
    ) -> None:
        self.max_steps = max_steps
        self.debug = debug
        self.progress_every = progress_every
        self.f_tolerance = default_to_factory(f_tolerance, LBFGSBFTerminator)
        self.g_tolerance = default_to_factory(g_tolerance, LBFGSBGTerminator)
        self.projected_gradient_tolerance = default_to_factory(
            projected_gradient_tolerance,
            LBFGSBInfNormGTerminator,
        )


class NelderMeadOptions(_GaneshRunOptionsMixin):
    """
    Built-in observer and terminator selection for Nelder-Mead.

    Parameters
    ----------
    max_steps : int | None, default=None
        Optional hard iteration limit.
    debug : bool, default=False
        Add the built-in debug observer.
    progress_every : int | None, default=None
        Add the built-in progress observer every ``progress_every`` steps.
    f_terminators : terminator or sequence of terminators or DefaultType or None, default=Default
        Function-space termination rules. ``Default`` constructs the library
        default singleton list. ``None`` omits all function-space terminators.
    x_terminators : terminator or sequence of terminators or DefaultType or None, default=Default
        Parameter-space termination rules. ``Default`` constructs the library
        default singleton list. ``None`` omits all parameter-space terminators.

    """

    __slots__ = ('debug', 'f_terminators', 'max_steps', 'progress_every', 'x_terminators')

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        debug: bool = False,
        progress_every: int | None = None,
        f_terminators: (
            NelderMeadAmoebaFTerminator
            | NelderMeadAbsoluteFTerminator
            | NelderMeadStdDevFTerminator
            | list[
                NelderMeadAmoebaFTerminator
                | NelderMeadAbsoluteFTerminator
                | NelderMeadStdDevFTerminator
            ]
            | DefaultType
            | None
        ) = Default,
        x_terminators: (
            NelderMeadDiameterXTerminator
            | NelderMeadHighamXTerminator
            | NelderMeadRowanXTerminator
            | NelderMeadSingerXTerminator
            | list[
                NelderMeadDiameterXTerminator
                | NelderMeadHighamXTerminator
                | NelderMeadRowanXTerminator
                | NelderMeadSingerXTerminator
            ]
            | DefaultType
            | None
        ) = Default,
    ) -> None:
        self.max_steps = max_steps
        self.debug = debug
        self.progress_every = progress_every
        self.f_terminators = _default_or_none_sequence(
            f_terminators,
            NelderMeadStdDevFTerminator,
        )
        self.x_terminators = _default_or_none_sequence(
            x_terminators,
            NelderMeadSingerXTerminator,
        )


class PSOOptions(_GaneshRunOptionsMixin):
    """
    Built-in observer selection for particle swarm optimization.

    Parameters
    ----------
    max_steps : int | None, default=None
        Optional hard iteration limit.
    debug : bool, default=False
        Add the built-in debug observer.
    progress_every : int | None, default=None
        Add the built-in progress observer every ``progress_every`` steps.

    """

    __slots__ = ('debug', 'max_steps', 'progress_every')

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        debug: bool = False,
        progress_every: int | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.debug = debug
        self.progress_every = progress_every


class DifferentialEvolutionOptions(_GaneshRunOptionsMixin):
    """
    Built-in observer selection for differential evolution.

    Parameters
    ----------
    max_steps : int | None, default=None
        Optional hard iteration limit.
    debug : bool, default=False
        Add the built-in debug observer.
    progress_every : int | None, default=None
        Add the built-in progress observer every ``progress_every`` steps.

    """

    __slots__ = ('debug', 'max_steps', 'progress_every')

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        debug: bool = False,
        progress_every: int | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.debug = debug
        self.progress_every = progress_every


class AIESOptions(_GaneshRunOptionsMixin):
    """
    Built-in observer and terminator selection for AIES.

    Parameters
    ----------
    max_steps : int | None, default=None
        Optional hard iteration limit.
    debug : bool, default=False
        Add the built-in debug observer.
    progress_every : int | None, default=None
        Add the built-in progress observer every ``progress_every`` steps.
    autocorrelation : AutocorrelationTerminator | None, default=None
        Optional autocorrelation-based MCMC terminator.

    """

    __slots__ = ('autocorrelation', 'debug', 'max_steps', 'progress_every')

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        debug: bool = False,
        progress_every: int | None = None,
        autocorrelation: AutocorrelationTerminator | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.debug = debug
        self.progress_every = progress_every
        self.autocorrelation = autocorrelation


class ESSOptions(_GaneshRunOptionsMixin):
    """
    Built-in observer and terminator selection for ESS.

    Parameters
    ----------
    max_steps : int | None, default=None
        Optional hard iteration limit.
    debug : bool, default=False
        Add the built-in debug observer.
    progress_every : int | None, default=None
        Add the built-in progress observer every ``progress_every`` steps.
    autocorrelation : AutocorrelationTerminator | None, default=None
        Optional autocorrelation-based MCMC terminator.

    """

    __slots__ = ('autocorrelation', 'debug', 'max_steps', 'progress_every')

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        debug: bool = False,
        progress_every: int | None = None,
        autocorrelation: AutocorrelationTerminator | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.debug = debug
        self.progress_every = progress_every
        self.autocorrelation = autocorrelation


class CMAESOptions(_GaneshRunOptionsMixin):
    """
    Built-in observer and terminator selection for CMA-ES.

    Parameters
    ----------
    max_steps : int | None, default=None
        Optional hard iteration limit.
    debug : bool, default=False
        Add the built-in debug observer.
    progress_every : int | None, default=None
        Add the built-in progress observer every ``progress_every`` steps.
    sigma : CMAESSigmaTerminator | DefaultType | None, default=Default
        Global step-size terminator. Pass ``Default`` to construct the library
        default terminator, or ``None`` to omit it.
    no_effect_axis : CMAESNoEffectAxisTerminator | DefaultType | None, default=Default
        No-effect-axis terminator. Pass ``Default`` to construct the library
        default terminator, or ``None`` to omit it.
    no_effect_coord : CMAESNoEffectCoordTerminator | DefaultType | None, default=Default
        No-effect-coordinate terminator. Pass ``Default`` to construct the
        library default terminator, or ``None`` to omit it.
    condition_cov : CMAESConditionCovTerminator | DefaultType | None, default=Default
        Covariance-condition-number terminator. Pass ``Default`` to construct
        the library default terminator, or ``None`` to omit it.
    equal_fun_values : CMAESEqualFunValuesTerminator | DefaultType | None, default=Default
        Equal-function-values terminator. Pass ``Default`` to construct the
        library default terminator, or ``None`` to omit it.
    stagnation : CMAESStagnationTerminator | DefaultType | None, default=Default
        Stagnation-history terminator. Pass ``Default`` to construct the
        library default terminator, or ``None`` to omit it.
    tol_x_up : CMAESTolXUpTerminator | DefaultType | None, default=Default
        Step-size explosion terminator. Pass ``Default`` to construct the
        library default terminator, or ``None`` to omit it.
    tol_fun : CMAESTolFunTerminator | DefaultType | None, default=Default
        Function-range terminator. Pass ``Default`` to construct the library
        default terminator, or ``None`` to omit it.
    tol_x : CMAESTolXTerminator | DefaultType | None, default=Default
        Coordinate-scale terminator. Pass ``Default`` to construct the library
        default terminator, or ``None`` to omit it.

    """

    __slots__ = (
        'condition_cov',
        'debug',
        'equal_fun_values',
        'max_steps',
        'no_effect_axis',
        'no_effect_coord',
        'progress_every',
        'sigma',
        'stagnation',
        'tol_fun',
        'tol_x',
        'tol_x_up',
    )

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        debug: bool = False,
        progress_every: int | None = None,
        sigma: CMAESSigmaTerminator | DefaultType | None = Default,
        no_effect_axis: CMAESNoEffectAxisTerminator | DefaultType | None = Default,
        no_effect_coord: CMAESNoEffectCoordTerminator | DefaultType | None = Default,
        condition_cov: CMAESConditionCovTerminator | DefaultType | None = Default,
        equal_fun_values: CMAESEqualFunValuesTerminator | DefaultType | None = Default,
        stagnation: CMAESStagnationTerminator | DefaultType | None = Default,
        tol_x_up: CMAESTolXUpTerminator | DefaultType | None = Default,
        tol_fun: CMAESTolFunTerminator | DefaultType | None = Default,
        tol_x: CMAESTolXTerminator | DefaultType | None = Default,
    ) -> None:
        self.max_steps = max_steps
        self.debug = debug
        self.progress_every = progress_every
        self.sigma = default_to_factory(sigma, CMAESSigmaTerminator)
        self.no_effect_axis = default_to_factory(
            no_effect_axis,
            CMAESNoEffectAxisTerminator,
        )
        self.no_effect_coord = default_to_factory(
            no_effect_coord,
            CMAESNoEffectCoordTerminator,
        )
        self.condition_cov = default_to_factory(
            condition_cov,
            CMAESConditionCovTerminator,
        )
        self.equal_fun_values = default_to_factory(
            equal_fun_values,
            CMAESEqualFunValuesTerminator,
        )
        self.stagnation = default_to_factory(stagnation, CMAESStagnationTerminator)
        self.tol_x_up = default_to_factory(tol_x_up, CMAESTolXUpTerminator)
        self.tol_fun = default_to_factory(tol_fun, CMAESTolFunTerminator)
        self.tol_x = default_to_factory(tol_x, CMAESTolXTerminator)


class AdamOptions(_GaneshRunOptionsMixin):
    """
    Built-in observer and terminator selection for Adam.

    Parameters
    ----------
    max_steps : int | None, default=None
        Optional hard iteration limit.
    debug : bool, default=False
        Add the built-in debug observer.
    progress_every : int | None, default=None
        Add the built-in progress observer every ``progress_every`` steps.
    ema : AdamEMATerminator | DefaultType | None, default=Default
        EMA-based convergence terminator. Pass ``Default`` to construct the
        library default terminator, or ``None`` to omit it.

    """

    __slots__ = ('debug', 'ema', 'max_steps', 'progress_every')

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        debug: bool = False,
        progress_every: int | None = None,
        ema: AdamEMATerminator | DefaultType | None = Default,
    ) -> None:
        self.max_steps = max_steps
        self.debug = debug
        self.progress_every = progress_every
        self.ema = default_to_factory(ema, AdamEMATerminator)


class ConjugateGradientOptions(_GaneshRunOptionsMixin):
    """
    Built-in observer and terminator selection for conjugate gradient.

    Parameters
    ----------
    max_steps : int | None, default=None
        Optional hard iteration limit.
    debug : bool, default=False
        Add the built-in debug observer.
    progress_every : int | None, default=None
        Add the built-in progress observer every ``progress_every`` steps.
    g_tolerance : ConjugateGradientGTerminator | DefaultType | None, default=Default
        Gradient-norm terminator. Pass ``Default`` to construct the library
        default terminator, or ``None`` to omit it.

    """

    __slots__ = ('debug', 'g_tolerance', 'max_steps', 'progress_every')

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        debug: bool = False,
        progress_every: int | None = None,
        g_tolerance: ConjugateGradientGTerminator | DefaultType | None = Default,
    ) -> None:
        self.max_steps = max_steps
        self.debug = debug
        self.progress_every = progress_every
        self.g_tolerance = default_to_factory(g_tolerance, ConjugateGradientGTerminator)


class TrustRegionOptions(_GaneshRunOptionsMixin):
    """
    Built-in observer and terminator selection for trust-region runs.

    Parameters
    ----------
    max_steps : int | None, default=None
        Optional hard iteration limit.
    debug : bool, default=False
        Add the built-in debug observer.
    progress_every : int | None, default=None
        Add the built-in progress observer every ``progress_every`` steps.
    g_tolerance : TrustRegionGTerminator | DefaultType | None, default=Default
        Gradient-norm terminator. Pass ``Default`` to construct the library
        default terminator, or ``None`` to omit it.

    """

    __slots__ = ('debug', 'g_tolerance', 'max_steps', 'progress_every')

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        debug: bool = False,
        progress_every: int | None = None,
        g_tolerance: TrustRegionGTerminator | DefaultType | None = Default,
    ) -> None:
        self.max_steps = max_steps
        self.debug = debug
        self.progress_every = progress_every
        self.g_tolerance = default_to_factory(g_tolerance, TrustRegionGTerminator)


class SimulatedAnnealingOptions(_GaneshRunOptionsMixin):
    """
    Built-in observer and terminator selection for simulated annealing.

    Parameters
    ----------
    max_steps : int | None, default=None
        Optional hard iteration limit.
    debug : bool, default=False
        Add the built-in debug observer.
    progress_every : int | None, default=None
        Add the built-in progress observer every ``progress_every`` steps.
    temperature : SimulatedAnnealingTemperatureTerminator | DefaultType | None, default=Default
        Minimum-temperature terminator. Pass ``Default`` to construct the
        library default terminator, or ``None`` to omit it.

    """

    __slots__ = ('debug', 'max_steps', 'progress_every', 'temperature')

    def __init__(
        self,
        *,
        max_steps: int | None = None,
        debug: bool = False,
        progress_every: int | None = None,
        temperature: SimulatedAnnealingTemperatureTerminator | DefaultType | None = Default,
    ) -> None:
        self.max_steps = max_steps
        self.debug = debug
        self.progress_every = progress_every
        self.temperature = default_to_factory(
            temperature,
            SimulatedAnnealingTemperatureTerminator,
        )


__all__ = [
    'AIESOptions',
    'AdamEMATerminator',
    'AdamOptions',
    'AutocorrelationTerminator',
    'CMAESConditionCovTerminator',
    'CMAESEqualFunValuesTerminator',
    'CMAESNoEffectAxisTerminator',
    'CMAESNoEffectCoordTerminator',
    'CMAESOptions',
    'CMAESSigmaTerminator',
    'CMAESStagnationTerminator',
    'CMAESTolFunTerminator',
    'CMAESTolXTerminator',
    'CMAESTolXUpTerminator',
    'ConjugateGradientGTerminator',
    'ConjugateGradientOptions',
    'DifferentialEvolutionOptions',
    'ESSOptions',
    'LBFGSBFTerminator',
    'LBFGSBGTerminator',
    'LBFGSBInfNormGTerminator',
    'LBFGSBOptions',
    'NelderMeadAbsoluteFTerminator',
    'NelderMeadAmoebaFTerminator',
    'NelderMeadDiameterXTerminator',
    'NelderMeadHighamXTerminator',
    'NelderMeadOptions',
    'NelderMeadRowanXTerminator',
    'NelderMeadSingerXTerminator',
    'NelderMeadStdDevFTerminator',
    'PSOOptions',
    'SimulatedAnnealingOptions',
    'SimulatedAnnealingTemperatureTerminator',
    'TrustRegionGTerminator',
    'TrustRegionOptions',
]
