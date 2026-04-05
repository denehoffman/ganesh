from __future__ import annotations

import inspect
import pickle
from collections.abc import Callable
from typing import Any, Protocol, cast

import ganesh
import ganesh._ganesh as native
import ganesh.config as ganesh_config
import ganesh.errors as ganesh_errors
import ganesh.run_options as ganesh_run_options
import numpy as np
import pytest


class DictExportingSummary(Protocol):
    def to_dict(self) -> dict[object, object]: ...


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
        'AIESInit',
        'ESSConfig',
        'ESSInit',
        'DifferentialEvolutionConfig',
        'DifferentialEvolutionInit',
        'CMAESConfig',
        'CMAESInit',
        'SimulatedAnnealingConfig',
        'AdamConfig',
        'ConjugateGradientConfig',
        'TrustRegionConfig',
        'NelderMeadInit',
        'PSOInit',
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
        'StatusMessage',
        'GradientStatus',
        'GradientFreeStatus',
        'EnsembleStatus',
        'SwarmStatus',
        'SimulatedAnnealingStatus',
        'MinimizationSummary',
        'MCMCSummary',
        'MultiStartSummary',
        'SimulatedAnnealingSummary',
        '__version__',
    }
    assert set(ganesh.__all__) == expected_public

    for name in expected_public - {'__version__', 'Default'}:
        exported = getattr(ganesh, name)
        assert isinstance(exported, type), name

    assert repr(ganesh.Default) == 'Default'
    assert isinstance(ganesh.__version__, str)


def test_package_reexports_match_submodules() -> None:
    assert ganesh.GaneshError is ganesh_errors.GaneshError
    assert ganesh.GaneshConfigError is ganesh_errors.GaneshConfigError
    assert ganesh.GaneshNumericalError is ganesh_errors.GaneshNumericalError

    assert ganesh.Default is not None
    assert ganesh.LBFGSBConfig is ganesh_config.LBFGSBConfig
    assert ganesh.NelderMeadConfig is ganesh_config.NelderMeadConfig
    assert ganesh.NelderMeadInit is ganesh_config.NelderMeadInit
    assert ganesh.PSOConfig is ganesh_config.PSOConfig
    assert ganesh.PSOInit is ganesh_config.PSOInit
    assert ganesh.AIESConfig is ganesh_config.AIESConfig
    assert ganesh.AIESInit is ganesh_config.AIESInit
    assert ganesh.ESSConfig is ganesh_config.ESSConfig
    assert ganesh.ESSInit is ganesh_config.ESSInit
    assert ganesh.DifferentialEvolutionConfig is ganesh_config.DifferentialEvolutionConfig
    assert ganesh.DifferentialEvolutionInit is ganesh_config.DifferentialEvolutionInit
    assert ganesh.CMAESConfig is ganesh_config.CMAESConfig
    assert ganesh.CMAESInit is ganesh_config.CMAESInit
    assert ganesh.SimulatedAnnealingConfig is ganesh_config.SimulatedAnnealingConfig
    assert ganesh.AdamConfig is ganesh_config.AdamConfig
    assert ganesh.ConjugateGradientConfig is ganesh_config.ConjugateGradientConfig
    assert ganesh.TrustRegionConfig is ganesh_config.TrustRegionConfig

    assert (
        ganesh.AutocorrelationTerminator is ganesh_run_options.AutocorrelationTerminator
    )
    assert ganesh.LBFGSBFTerminator is ganesh_run_options.LBFGSBFTerminator
    assert ganesh.LBFGSBGTerminator is ganesh_run_options.LBFGSBGTerminator
    assert ganesh.LBFGSBInfNormGTerminator is ganesh_run_options.LBFGSBInfNormGTerminator
    assert (
        ganesh.NelderMeadAmoebaFTerminator
        is ganesh_run_options.NelderMeadAmoebaFTerminator
    )
    assert (
        ganesh.NelderMeadAbsoluteFTerminator
        is ganesh_run_options.NelderMeadAbsoluteFTerminator
    )
    assert (
        ganesh.NelderMeadStdDevFTerminator
        is ganesh_run_options.NelderMeadStdDevFTerminator
    )
    assert (
        ganesh.NelderMeadDiameterXTerminator
        is ganesh_run_options.NelderMeadDiameterXTerminator
    )
    assert (
        ganesh.NelderMeadHighamXTerminator
        is ganesh_run_options.NelderMeadHighamXTerminator
    )
    assert (
        ganesh.NelderMeadRowanXTerminator is ganesh_run_options.NelderMeadRowanXTerminator
    )
    assert (
        ganesh.NelderMeadSingerXTerminator
        is ganesh_run_options.NelderMeadSingerXTerminator
    )
    assert ganesh.AdamEMATerminator is ganesh_run_options.AdamEMATerminator
    assert (
        ganesh.ConjugateGradientGTerminator
        is ganesh_run_options.ConjugateGradientGTerminator
    )
    assert ganesh.TrustRegionGTerminator is ganesh_run_options.TrustRegionGTerminator
    assert (
        ganesh.SimulatedAnnealingTemperatureTerminator
        is ganesh_run_options.SimulatedAnnealingTemperatureTerminator
    )
    assert ganesh.CMAESSigmaTerminator is ganesh_run_options.CMAESSigmaTerminator
    assert (
        ganesh.CMAESNoEffectAxisTerminator
        is ganesh_run_options.CMAESNoEffectAxisTerminator
    )
    assert (
        ganesh.CMAESNoEffectCoordTerminator
        is ganesh_run_options.CMAESNoEffectCoordTerminator
    )
    assert (
        ganesh.CMAESConditionCovTerminator
        is ganesh_run_options.CMAESConditionCovTerminator
    )
    assert (
        ganesh.CMAESEqualFunValuesTerminator
        is ganesh_run_options.CMAESEqualFunValuesTerminator
    )
    assert (
        ganesh.CMAESStagnationTerminator is ganesh_run_options.CMAESStagnationTerminator
    )
    assert ganesh.CMAESTolXUpTerminator is ganesh_run_options.CMAESTolXUpTerminator
    assert ganesh.CMAESTolFunTerminator is ganesh_run_options.CMAESTolFunTerminator
    assert ganesh.CMAESTolXTerminator is ganesh_run_options.CMAESTolXTerminator

    assert ganesh.LBFGSBOptions is ganesh_run_options.LBFGSBOptions
    assert ganesh.NelderMeadOptions is ganesh_run_options.NelderMeadOptions
    assert ganesh.PSOOptions is ganesh_run_options.PSOOptions
    assert (
        ganesh.DifferentialEvolutionOptions
        is ganesh_run_options.DifferentialEvolutionOptions
    )
    assert ganesh.AIESOptions is ganesh_run_options.AIESOptions
    assert ganesh.ESSOptions is ganesh_run_options.ESSOptions
    assert ganesh.CMAESOptions is ganesh_run_options.CMAESOptions
    assert ganesh.AdamOptions is ganesh_run_options.AdamOptions
    assert ganesh.ConjugateGradientOptions is ganesh_run_options.ConjugateGradientOptions
    assert ganesh.TrustRegionOptions is ganesh_run_options.TrustRegionOptions
    assert (
        ganesh.SimulatedAnnealingOptions is ganesh_run_options.SimulatedAnnealingOptions
    )
    assert ganesh.MinimizationSummary is native.MinimizationSummary
    assert ganesh.MCMCSummary is native.MCMCSummary
    assert ganesh.MultiStartSummary is native.MultiStartSummary
    assert ganesh.SimulatedAnnealingSummary is native.SimulatedAnnealingSummary
    assert ganesh.StatusMessage is native.StatusMessage
    assert ganesh.GradientStatus is native.GradientStatus
    assert ganesh.GradientFreeStatus is native.GradientFreeStatus
    assert ganesh.EnsembleStatus is native.EnsembleStatus
    assert ganesh.SwarmStatus is native.SwarmStatus
    assert ganesh.SimulatedAnnealingStatus is native.SimulatedAnnealingStatus
    assert ganesh.__version__ == native.__version__


def test_package_signatures_show_default_sentinel() -> None:
    assert 'line_search' in str(inspect.signature(ganesh.LBFGSBConfig))
    assert 'x0' not in str(inspect.signature(ganesh.LBFGSBConfig))
    assert '= None' in str(inspect.signature(ganesh.LBFGSBConfig))
    assert 'construction_method' not in str(inspect.signature(ganesh.NelderMeadConfig))
    assert 'x0' in str(inspect.signature(ganesh.NelderMeadInit))
    assert 'moves' in str(inspect.signature(ganesh.AIESConfig))
    assert 'walkers' not in str(inspect.signature(ganesh.AIESConfig))
    assert 'walkers' in str(inspect.signature(ganesh.AIESInit))
    assert 'chain_storage' in str(inspect.signature(ganesh.AIESConfig))
    assert 'population_size' in str(inspect.signature(ganesh.CMAESConfig))
    assert 'sigma' not in str(inspect.signature(ganesh.CMAESConfig))
    assert 'sigma' in str(inspect.signature(ganesh.CMAESInit))
    assert 'f_tolerance' in str(inspect.signature(ganesh.LBFGSBOptions))
    assert '= Default' in str(inspect.signature(ganesh.LBFGSBOptions))
    assert 'f_terminators' in str(inspect.signature(ganesh.NelderMeadOptions))
    assert 'x_terminators' in str(inspect.signature(ganesh.NelderMeadOptions))
    assert 'g_tolerance' in str(inspect.signature(ganesh.ConjugateGradientOptions))
    assert 'sigma' in str(inspect.signature(ganesh.CMAESOptions))
    assert 'autocorrelation' in str(inspect.signature(ganesh.AIESOptions))
    assert '= None' in str(inspect.signature(ganesh.AIESOptions))


def test_nelder_mead_init_rejects_ambiguous_shape() -> None:
    with pytest.raises(ValueError, match='either x0 or construction_method'):
        ganesh.NelderMeadInit(
            x0=[1.0, 1.0],
            construction_method=ganesh.OrthogonalSimplex([1.0, 1.0], simplex_size=0.5),
        )


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
        cast(Any, summary.chain)(1)

    with pytest.raises(TypeError):
        cast(Any, summary.chain)(None, None, True)

    with pytest.raises(TypeError):
        cast(Any, summary.diagnostics)(1, 1)


def test_simulated_annealing_summary_wrapper_uses_numpy_arrays() -> None:
    summary = native._testing_sample_simulated_annealing_summary()

    assert isinstance(summary.x0, np.ndarray)
    assert isinstance(summary.x, np.ndarray)

    exported = summary.to_dict()
    assert isinstance(exported['x0'], np.ndarray)
    assert isinstance(exported['x'], np.ndarray)


def test_multistart_summary_wrapper_exposes_runs_and_best_run() -> None:
    summary = native._testing_sample_multistart_summary()

    assert summary.best_run_index == 1
    assert summary.restart_count == 1
    assert summary.completed_runs == 2

    runs = summary.runs
    assert len(runs) == 2
    assert isinstance(runs[0], ganesh.MinimizationSummary)
    assert isinstance(summary.best_run, ganesh.MinimizationSummary)
    assert summary.best_run.fx == 1.25

    exported = summary.to_dict()
    assert exported['best_run_index'] == 1
    assert exported['restart_count'] == 1
    assert exported['completed_runs'] == 2
    assert len(exported['runs']) == 2
    assert exported['best_run']['fx'] == 1.25


@pytest.mark.parametrize(
    ('factory', 'expected_type'),
    [
        (native._testing_sample_minimization_summary, ganesh.MinimizationSummary),
        (native._testing_sample_mcmc_summary, ganesh.MCMCSummary),
        (native._testing_sample_multistart_summary, ganesh.MultiStartSummary),
        (
            native._testing_sample_simulated_annealing_summary,
            ganesh.SimulatedAnnealingSummary,
        ),
    ],
)
def test_summary_wrappers_support_pickling(
    factory: Callable[[], DictExportingSummary], expected_type: type[object]
) -> None:
    summary = factory()
    restored = pickle.loads(pickle.dumps(summary))

    assert isinstance(restored, expected_type)
    assert restored.to_dict().keys() == summary.to_dict().keys()


def test_status_message_wrapper_exposes_fields() -> None:
    status = native._testing_sample_gradient_status().message

    assert isinstance(status, ganesh.StatusMessage)
    assert status.status_type == 'Step'
    assert status.text == 'iterating'
    assert status.success is False
    assert status.to_dict()['text'] == 'iterating'


def test_gradient_status_wrapper_uses_numpy_arrays() -> None:
    status = native._testing_sample_gradient_status()

    assert isinstance(status.x, np.ndarray)
    assert isinstance(status.hess, np.ndarray)
    assert isinstance(status.cov, np.ndarray)
    assert isinstance(status.err, np.ndarray)
    assert status.n_f_evals == 12
    assert status.n_g_evals == 7
    assert status.n_h_evals == 2

    exported = status.to_dict()
    assert exported['message']['text'] == 'iterating'
    assert isinstance(exported['x'], np.ndarray)
    assert isinstance(exported['hess'], np.ndarray)
    assert isinstance(exported['cov'], np.ndarray)
    assert isinstance(exported['err'], np.ndarray)


def test_gradient_free_status_wrapper_uses_numpy_arrays() -> None:
    status = native._testing_sample_gradient_free_status()

    assert isinstance(status.x, np.ndarray)
    assert isinstance(status.hess, np.ndarray)
    assert isinstance(status.cov, np.ndarray)
    assert isinstance(status.err, np.ndarray)
    assert status.n_f_evals == 18

    exported = status.to_dict()
    assert exported['message']['text'] == 'simplex updated'
    assert isinstance(exported['x'], np.ndarray)


def test_ensemble_status_wrapper_uses_numpy_arrays() -> None:
    status = native._testing_sample_ensemble_status()

    assert isinstance(status.get_chain(), np.ndarray)
    assert isinstance(status.get_flat_chain(), np.ndarray)
    assert status.n_f_evals == 14
    assert status.n_g_evals == 0

    exported = status.to_dict()
    assert isinstance(exported['chain'], np.ndarray)
    assert exported['dimension'] == (2, 2, 2)


def test_ensemble_status_chain_uses_keyword_only_options() -> None:
    status = native._testing_sample_ensemble_status()

    with pytest.raises(TypeError):
        cast(Any, status.get_chain)(1)

    with pytest.raises(TypeError):
        cast(Any, status.get_flat_chain)(1, 1)


def test_swarm_status_wrapper_exposes_swarm() -> None:
    status = native._testing_sample_swarm_status()

    particles = status.swarm['particles']
    assert len(particles) == 2
    assert isinstance(particles[0]['position']['x'], np.ndarray)
    assert isinstance(particles[0]['velocity'], np.ndarray)
    assert status.n_f_evals == 22
    assert status.get_best()['fx'] == 0.125

    exported = status.to_dict()
    assert exported['swarm']['topology'] == 'Ring'
    assert exported['swarm']['update_method'] == 'Asynchronous'
    assert exported['swarm']['boundary_method'] == 'Shr'


def test_simulated_annealing_status_wrapper_uses_numpy_arrays() -> None:
    status = native._testing_sample_simulated_annealing_status()

    assert isinstance(status.initial['x'], np.ndarray)
    assert isinstance(status.best['x'], np.ndarray)
    assert isinstance(status.current['x'], np.ndarray)
    assert status.n_f_evals == 33

    exported = status.to_dict()
    assert exported['temperature'] == 0.75
    assert isinstance(exported['initial']['x'], np.ndarray)
