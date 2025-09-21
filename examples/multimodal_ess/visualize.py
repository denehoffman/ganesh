#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "corner",
#     "matplotlib",
#     "numpy",
# ]
# ///
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    with Path('data.pkl').open('rb') as f:
        (chains, taus) = pickle.load(f)  # noqa: S301
    chains = np.array([[[step[0][i] for i in range(2)] for step in chain] for chain in chains])
    plt.plot(np.arange(len(taus)) * 50, taus)
    plt.title('Integrated Autocorrelation Time')
    plt.xlabel('Step')
    plt.ylabel(r'Mean $\tau$')
    plt.savefig('iat.svg')
    plt.close()
    _, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 50))
    steps = np.arange(chains.shape[1])
    burn_in = int(taus[-1])
    for i in range(chains.shape[2]):
        for j in range(chains.shape[0]):
            ax[i].plot(steps[burn_in:], chains[j, burn_in:, i], color='k', alpha=0.1)
            ax[i].plot(steps[:burn_in], chains[j, :burn_in, i], color='k', ls='--', alpha=0.1)
        ax[i].plot(steps[burn_in:], chains[0, burn_in:, i], color='r', label='Walker 0')
        ax[i].plot(steps[:burn_in], chains[0, :burn_in, i], color='r', ls='--', label='Walker 0 (burn-in)')
        ax[i].set_xlabel('Step')
        ax[i].set_ylabel(f'Parameter {i}')
        ax[i].legend()
    plt.savefig('traces.svg')
    plt.close()
    flat_chain = chains[:, burn_in:, :].reshape(-1, chains.shape[2])

    plt.scatter(
        flat_chain[:, 0],
        flat_chain[:, 1],
        s=1,
        marker='.',
        linewidths=0,
        edgecolors='none',
        color='k',
        label='MCMC Samples',
    )
    plt.plot(
        chains[0, burn_in:, 0],
        chains[0, burn_in:, 1],
        markersize=1.0,
        linewidth=0.1,
        marker='.',
        markeredgecolor='none',
        color='r',
        label='Walker 0',
    )
    g = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(g, g)
    Z = np.power((np.power(X, 2) + Y - 11), 2) + np.power(X + np.power(Y, 2) - 7, 2)
    plt.contour(X, Y, -Z, levels=20)
    plt.legend()
    plt.savefig('scatter.png', dpi=2000)  # high DPI here because the image is very detailed
    plt.close()
