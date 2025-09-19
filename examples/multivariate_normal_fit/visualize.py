# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "matplotlib",
#     "matplotloom",
#     "numpy",
#     "corner",
#     "joblib",
#     "polars",
# ]
# ///
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from corner import corner, overplot_lines
from joblib import Parallel, delayed
from matplotloom import Loom

if __name__ == '__main__':
    print('Plotting dataset...')
    data = np.array(pickle.load(Path.open('data.pkl', 'rb'))).transpose()
    plt.hist2d(*data, bins=100, cmap='gist_heat_r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sampled Dataset')
    plt.savefig('data.svg')
    plt.close()

    print('Plotting traces (no burn-in)...')
    parameter_labels = [r'$\mu_0$', r'$\mu_1$', r'$\Sigma_{00}$', r'$\Sigma_{01}$', r'$\Sigma_{11}$']
    fit_result_data = pickle.load(Path.open('fit.pkl', 'rb'))
    truths = np.array(fit_result_data[0])
    fit_result = np.array(fit_result_data[1])
    fit_result_err = np.array(fit_result_data[2])
    chain, burn = pickle.load(Path.open('chain.pkl', 'rb'))
    chain = np.array(chain)
    n_walkers, n_steps, n_parameters = chain.shape
    _, ax = plt.subplots(nrows=n_parameters, sharex=True, figsize=(10, 50))
    steps = np.arange(n_steps)
    for i in range(n_parameters):
        for j in range(n_walkers):
            ax[i].plot(steps[burn:], chain[j, burn:, i], color='k', alpha=0.1)
            ax[i].plot(steps[:burn], chain[j, :burn, i], color='k', ls='--', alpha=0.1)
        ax[i].plot(steps[burn:], chain[0, burn:, i], color='m', label='Walker 0')
        ax[i].plot(steps[:burn], chain[0, :burn, i], color='m', ls='--', label='Walker 0 (burn-in)')
        ax[i].axhline(fit_result[i], color='b', label='Best fit')
        ax[i].set_xlabel('Step')
        ax[i].set_ylabel(parameter_labels[i])
        ax[i].legend()
    plt.savefig('traces.svg')
    plt.close()

    print('Plotting traces (with burn-in)...')
    _, ax = plt.subplots(nrows=n_parameters, sharex=True, figsize=(10, 50))
    steps = np.arange(n_steps)
    for i in range(n_parameters):
        for j in range(n_walkers):
            ax[i].plot(steps[burn:], chain[j, burn:, i], color='k', alpha=0.1)
        ax[i].plot(steps[burn:], chain[0, burn:, i], color='m', label='Walker 0')
        ax[i].axhline(fit_result[i], color='b', label='Best fit')
        ax[i].set_xlabel('Step')
        ax[i].set_ylabel(parameter_labels[i])
        ax[i].legend()
    plt.savefig('traces_burned.svg')
    plt.close()

    print('Plotting corner plot...')
    ci = 68.27
    flat_chain = np.array(pickle.load(Path.open('flat_chain.pkl', 'rb')))
    fig = corner(
        flat_chain,
        labels=parameter_labels,
        truths=truths,
        truths_color='r',
        quantiles=[(50 - ci / 2) / 100, 0.5, (50 + ci / 2) / 100],
        show_titles=True,
        title_fmt='.4f',
    )
    overplot_lines(
        fig,
        fit_result,
        color='b',
    )
    plt.savefig('corner.svg')
    plt.close()

    def compute_ranges(chain, pad_frac=0.02):
        mins = chain.min(axis=(0, 1))
        maxs = chain.max(axis=(0, 1))
        widths = np.maximum(maxs - mins, 1e-9)
        mins = mins - pad_frac * widths
        maxs = maxs + pad_frac * widths
        return [(float(a), float(b)) for a, b in zip(mins, maxs)]

    def make_frame(i, chain, labels, ranges, loom):
        j0 = max(0, i - 10)
        window = chain[:, j0 : i + 1, :]
        flat = window.reshape(-1, window.shape[-1])

        fig = corner(
            flat,
            labels=labels,
            range=ranges,
            plot_contours=False,
            show_titles=False,
        )
        loom.save_frame(fig, i)
        plt.close(fig)

    burned_chain = chain[:, burn:, :]
    n_steps = burned_chain.shape[1]
    ranges = compute_ranges(burned_chain)

    print('Making animated corner plot...')
    with Loom('walkers_corner.gif', fps=20, parallel=True, overwrite=True) as loom:
        Parallel(n_jobs=-1, prefer='processes')(
            delayed(make_frame)(i, burned_chain, parameter_labels, ranges, loom) for i in range(n_steps)
        )

    parameter_labels_unicode = ['μ₀', 'μ₁', 'Σ₀₀', 'Σ₀₁', 'Σ₁₁']
    qlo, qmid, qhi = (50 - ci / 2, 50, 50 + ci / 2)
    lo, mid, hi = np.percentile(flat_chain, [qlo, qmid, qhi], axis=0)
    mcmc_err_minus = mid - lo
    mcmc_err_plus = hi - mid
    fit_col = [f'{v:.6g}' for v in fit_result]
    cov_col = [f'±{e:.3g}' for e in fit_result_err]
    mcmc_col = [f'-{em:.3g} / +{ep:.3g}' for em, ep in zip(mcmc_err_minus, mcmc_err_plus)]
    truth_col = [f'{t:.6g}' for t in truths]
    print('Summary')
    print(
        pl.DataFrame(
            {
                'Parameter': parameter_labels_unicode,
                'Truths': truth_col,
                'Fit Result': fit_col,
                'Uncertainty (Covariance)': cov_col,
                'Uncertainty (MCMC)': mcmc_col,
            }
        )
    )
