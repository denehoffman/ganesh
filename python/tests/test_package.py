# ruff: noqa: S101, INP001
from __future__ import annotations

import inspect

import ganesh
import ganesh._ganesh as native  # ty:ignore[unresolved-import]
import ganesh.config as ganesh_config
import ganesh.errors as ganesh_errors
import ganesh.run_options as ganesh_run_options
import numpy as np
import pytest  # ty:ignore[unresolved-import]


def test_package_exports_expected_symbols() -> None:
    expected_public = {
        'Default',
        'GaneshError',
        'GaneshConfigError',
        'GaneshNumericalError',
        'MoreThuenteLineSearch',
        'HagerZhangLineSearch',
        'ScaledOrthogonalSimplex',
        'OrthogonalSimplex',
        'CustomSimplex',
        'ChainStorageFull',
        'ChainStorageRolling',
        'ChainStorageSampled',
        'AIESStretchMove',
        'AIESWalkMove',
        'ESSDifferentialMove',
        'ESSGaussianMove',
        'ESSGlobalMove',
        'LBFGSBConfig',
        'NelderMeadConfig',
        'PSOConfig',
        'AIESConfig',
        'ESSConfig',
        'DifferentialEvolutionConfig',
        'CMAESConfig',
        'SimulatedAnnealingConfig',
        'AdamConfig',
        'ConjugateGradientConfig',
        'TrustRegionConfig',
        'AutocorrelationTerminator',
        'LBFGSBFTerminator',
        'LBFGSBGTerminator',
        'LBFGSBInfNormGTerminator',
        'NelderMeadAmoebaFTerminator',
        'NelderMeadAbsoluteFTerminator',
        'NelderMeadStdDevFTerminator',
        'NelderMeadDiameterXTerminator',
        'NelderMeadHighamXTerminator',
        'NelderMeadRowanXTerminator',
        'NelderMeadSingerXTerminator',
        'AdamEMATerminator',
        'ConjugateGradientGTerminator',
        'TrustRegionGTerminator',
        'SimulatedAnnealingTemperatureTerminator',
        'CMAESSigmaTerminator',
        'CMAESNoEffectAxisTerminator',
        'CMAESNoEffectCoordTerminator',
        'CMAESConditionCovTerminator',
        'CMAESEqualFunValuesTerminator',
        'CMAESStagnationTerminator',
        'CMAESTolXUpTerminator',
        'CMAESTolFunTerminator',
        'CMAESTolXTerminator',
        'LBFGSBOptions',
        'NelderMeadOptions',
        'PSOOptions',
        'DifferentialEvolutionOptions',
        'AIESOptions',
        'ESSOptions',
        'CMAESOptions',
        'AdamOptions',
        'ConjugateGradientOptions',
        'TrustRegionOptions',
        'SimulatedAnnealingOptions',
        'MinimizationSummary',
        'MCMCSummary',
        'SimulatedAnnealingSummary',
        '__version__',
    }
    assert set(ganesh.__all__) == expected_public

    for name in expected_public - {'__version__', 'Default'}:
        exported = getattr(ganesh, name)
        assert isinstance(exported, type), name

    assert repr(ganesh.Default) == 'Default'
    assert isinstance(ganesh.__version__, str)  # ty:ignore[possibly-missing-attribute]


def test_package_reexports_match_submodules() -> None:  # noqa: PLR0915
    assert ganesh.GaneshError is ganesh_errors.GaneshError
    assert ganesh.GaneshConfigError is ganesh_errors.GaneshConfigError
    assert ganesh.GaneshNumericalError is ganesh_errors.GaneshNumericalError

    assert ganesh.Default is not None
    assert ganesh.LBFGSBConfig is ganesh_config.LBFGSBConfig
    assert ganesh.NelderMeadConfig is ganesh_config.NelderMeadConfig
    assert ganesh.PSOConfig is ganesh_config.PSOConfig
    assert ganesh.AIESConfig is ganesh_config.AIESConfig
    assert ganesh.ESSConfig is ganesh_config.ESSConfig
    assert ganesh.DifferentialEvolutionConfig is ganesh_config.DifferentialEvolutionConfig
    assert ganesh.CMAESConfig is ganesh_config.CMAESConfig
    assert ganesh.SimulatedAnnealingConfig is ganesh_config.SimulatedAnnealingConfig
    assert ganesh.AdamConfig is ganesh_config.AdamConfig
    assert ganesh.ConjugateGradientConfig is ganesh_config.ConjugateGradientConfig
    assert ganesh.TrustRegionConfig is ganesh_config.TrustRegionConfig

    assert ganesh.AutocorrelationTerminator is ganesh_run_options.AutocorrelationTerminator
    assert ganesh.LBFGSBFTerminator is ganesh_run_options.LBFGSBFTerminator
    assert ganesh.LBFGSBGTerminator is ganesh_run_options.LBFGSBGTerminator
    assert ganesh.LBFGSBInfNormGTerminator is ganesh_run_options.LBFGSBInfNormGTerminator
    assert ganesh.NelderMeadAmoebaFTerminator is ganesh_run_options.NelderMeadAmoebaFTerminator
    assert ganesh.NelderMeadAbsoluteFTerminator is ganesh_run_options.NelderMeadAbsoluteFTerminator
    assert ganesh.NelderMeadStdDevFTerminator is ganesh_run_options.NelderMeadStdDevFTerminator
    assert ganesh.NelderMeadDiameterXTerminator is ganesh_run_options.NelderMeadDiameterXTerminator
    assert ganesh.NelderMeadHighamXTerminator is ganesh_run_options.NelderMeadHighamXTerminator
    assert ganesh.NelderMeadRowanXTerminator is ganesh_run_options.NelderMeadRowanXTerminator
    assert ganesh.NelderMeadSingerXTerminator is ganesh_run_options.NelderMeadSingerXTerminator
    assert ganesh.AdamEMATerminator is ganesh_run_options.AdamEMATerminator
    assert ganesh.ConjugateGradientGTerminator is ganesh_run_options.ConjugateGradientGTerminator
    assert ganesh.TrustRegionGTerminator is ganesh_run_options.TrustRegionGTerminator
    assert ganesh.SimulatedAnnealingTemperatureTerminator is ganesh_run_options.SimulatedAnnealingTemperatureTerminator
    assert ganesh.CMAESSigmaTerminator is ganesh_run_options.CMAESSigmaTerminator
    assert ganesh.CMAESNoEffectAxisTerminator is ganesh_run_options.CMAESNoEffectAxisTerminator
    assert ganesh.CMAESNoEffectCoordTerminator is ganesh_run_options.CMAESNoEffectCoordTerminator
    assert ganesh.CMAESConditionCovTerminator is ganesh_run_options.CMAESConditionCovTerminator
    assert ganesh.CMAESEqualFunValuesTerminator is ganesh_run_options.CMAESEqualFunValuesTerminator
    assert ganesh.CMAESStagnationTerminator is ganesh_run_options.CMAESStagnationTerminator
    assert ganesh.CMAESTolXUpTerminator is ganesh_run_options.CMAESTolXUpTerminator
    assert ganesh.CMAESTolFunTerminator is ganesh_run_options.CMAESTolFunTerminator
    assert ganesh.CMAESTolXTerminator is ganesh_run_options.CMAESTolXTerminator

    assert ganesh.LBFGSBOptions is ganesh_run_options.LBFGSBOptions
    assert ganesh.NelderMeadOptions is ganesh_run_options.NelderMeadOptions
    assert ganesh.PSOOptions is ganesh_run_options.PSOOptions
    assert ganesh.DifferentialEvolutionOptions is ganesh_run_options.DifferentialEvolutionOptions
    assert ganesh.AIESOptions is ganesh_run_options.AIESOptions
    assert ganesh.ESSOptions is ganesh_run_options.ESSOptions
    assert ganesh.CMAESOptions is ganesh_run_options.CMAESOptions
    assert ganesh.AdamOptions is ganesh_run_options.AdamOptions
    assert ganesh.ConjugateGradientOptions is ganesh_run_options.ConjugateGradientOptions
    assert ganesh.TrustRegionOptions is ganesh_run_options.TrustRegionOptions
    assert ganesh.SimulatedAnnealingOptions is ganesh_run_options.SimulatedAnnealingOptions
    assert ganesh.MinimizationSummary is native.MinimizationSummary  # ty:ignore[possibly-missing-attribute]
    assert ganesh.MCMCSummary is native.MCMCSummary  # ty:ignore[possibly-missing-attribute]
    assert ganesh.SimulatedAnnealingSummary is native.SimulatedAnnealingSummary  # ty:ignore[possibly-missing-attribute]
    assert ganesh.__version__ == native.__version__  # ty:ignore[possibly-missing-attribute]


def test_package_signatures_show_default_sentinel() -> None:
    assert 'line_search' in str(inspect.signature(ganesh.LBFGSBConfig))
    assert '= None' in str(inspect.signature(ganesh.LBFGSBConfig))
    assert 'moves' in str(inspect.signature(ganesh.AIESConfig))
    assert 'chain_storage' in str(inspect.signature(ganesh.AIESConfig))
    assert 'population_size' in str(inspect.signature(ganesh.CMAESConfig))
    assert 'f_tolerance' in str(inspect.signature(ganesh.LBFGSBOptions))
    assert '= Default' in str(inspect.signature(ganesh.LBFGSBOptions))
    assert 'f_terminators' in str(inspect.signature(ganesh.NelderMeadOptions))
    assert 'x_terminators' in str(inspect.signature(ganesh.NelderMeadOptions))
    assert 'g_tolerance' in str(inspect.signature(ganesh.ConjugateGradientOptions))
    assert 'sigma' in str(inspect.signature(ganesh.CMAESOptions))
    assert 'autocorrelation' in str(inspect.signature(ganesh.AIESOptions))
    assert '= None' in str(inspect.signature(ganesh.AIESOptions))


def test_exception_hierarchy_is_exposed() -> None:
    assert issubclass(ganesh.GaneshConfigError, ganesh.GaneshError)
    assert issubclass(ganesh.GaneshNumericalError, ganesh.GaneshError)


def test_minimization_summary_wrapper_uses_numpy_arrays() -> None:
    summary = native._testing_sample_minimization_summary()

    assert isinstance(summary.x0, np.ndarray)
    assert isinstance(summary.x, np.ndarray)
    assert isinstance(summary.std, np.ndarray)
    assert isinstance(summary.covariance, np.ndarray)

    exported = summary.to_dict()
    assert isinstance(exported['x0'], np.ndarray)
    assert isinstance(exported['x'], np.ndarray)
    assert isinstance(exported['std'], np.ndarray)
    assert isinstance(exported['covariance'], np.ndarray)


def test_mcmc_summary_wrapper_uses_numpy_arrays() -> None:
    summary = native._testing_sample_mcmc_summary()

    assert isinstance(summary.chain(), np.ndarray)
    assert isinstance(summary.chain(burn=1, thin=1), np.ndarray)
    assert isinstance(summary.chain(flat=True), np.ndarray)

    diagnostics = summary.diagnostics(burn=0, thin=1)
    assert isinstance(diagnostics['r_hat'], np.ndarray)
    assert isinstance(diagnostics['ess'], np.ndarray)
    assert isinstance(diagnostics['acceptance_rates'], np.ndarray)

    exported = summary.to_dict()
    assert isinstance(exported['chain'], np.ndarray)


def test_mcmc_summary_chain_and_diagnostics_use_keyword_only_options() -> None:
    summary = native._testing_sample_mcmc_summary()

    with pytest.raises(TypeError):
        summary.chain(1)

    with pytest.raises(TypeError):
        summary.chain(None, None, True)  # noqa: FBT003

    with pytest.raises(TypeError):
        summary.diagnostics(1, 1)


def test_simulated_annealing_summary_wrapper_uses_numpy_arrays() -> None:
    summary = native._testing_sample_simulated_annealing_summary()

    assert isinstance(summary.x0, np.ndarray)
    assert isinstance(summary.x, np.ndarray)

    exported = summary.to_dict()
    assert isinstance(exported['x0'], np.ndarray)
    assert isinstance(exported['x'], np.ndarray)
