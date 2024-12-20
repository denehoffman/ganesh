#!/usr/bin/env python3
import pickle
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    with Path("data.pkl").open("rb") as f:
        (chains, taus) = pickle.load(f)  # noqa: S301
    chains = np.array([[[step[0][i] for i in range(5)] for step in chain] for chain in chains])
    plt.plot(np.arange(len(taus)) * 50, taus)
    plt.title("Integrated Autocorrelation Time")
    plt.xlabel("Step")
    plt.ylabel(r"Mean $\tau$")
    plt.savefig("iat.svg")
    plt.close()
    _, ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 50))
    steps = np.arange(chains.shape[1])
    burn_in = int(taus[-1] * 50)
    for i in range(chains.shape[2]):
        for j in range(chains.shape[0]):
            ax[i].plot(steps[burn_in:], chains[j, burn_in:, i], color="k", alpha=0.1)
            ax[i].plot(steps[:burn_in], chains[j, :burn_in, i], color="k", ls="--", alpha=0.1)
        ax[i].plot(steps[burn_in:], chains[0, burn_in:, i], color="r", label="Walker 0")
        ax[i].plot(steps[:burn_in], chains[0, :burn_in, i], color="r", ls="--", label="Walker 0 (burn-in)")
        ax[i].set_xlabel("Step")
        ax[i].set_ylabel(f"Parameter {i}")
        ax[i].legend()
    plt.savefig("traces.svg")
    plt.close()
    flat_chain = chains[:, burn_in:, :].reshape(-1, chains.shape[2])
    fig = corner.corner(
        flat_chain, labels=[f"Parameter {i}" for i in range(5)], quantiles=[0.16, 0.5, 0.84], show_titles=True
    )
    plt.savefig("corner_plot.svg")
    plt.close()
