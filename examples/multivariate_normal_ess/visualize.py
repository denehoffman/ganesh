#!/usr/bin/env python3
import pickle
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    with Path("data.pkl").open("rb") as f:
        data = pickle.load(f)  # noqa: S301
    chains = np.array([[[step[0][i] for i in range(5)] for step in chain] for chain in data.get("chains")])
    _, ax = plt.subplots(nrows=5, sharex=True, figsize=(10, 50))
    steps = np.arange(chains.shape[1])
    for i in range(chains.shape[2]):
        for j in range(chains.shape[0]):
            ax[i].plot(steps[50:], chains[j, 50:, i], color="k", alpha=0.1)
            ax[i].plot(steps[:50], chains[j, :50, i], color="k", ls="--", alpha=0.1)
        ax[i].plot(steps[50:], chains[0, 50:, i], color="r", label="Walker 0 (Burn-In)")
        ax[i].plot(steps[:50], chains[0, :50, i], color="r", ls="--", label="Walker 0")
        ax[i].set_xlabel("Step")
        ax[i].set_ylabel(f"Parameter {i}")
        ax[i].legend()
    plt.savefig("traces.svg")
    plt.close()
    flat_chain = chains[:, 50:, :].reshape(-1, chains.shape[2])
    fig = corner.corner(
        flat_chain, labels=[f"Parameter {i}" for i in range(5)], quantiles=[0.16, 0.5, 0.84], show_titles=True
    )
    plt.savefig("corner_plot.svg")
    plt.close()
