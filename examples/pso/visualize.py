#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "joblib",
#     "matplotlib",
#     "matplotloom",
#     "numpy",
# ]
# ///
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotloom import Loom


def func(x, y):
    return 10 + x**2 - 10 * np.cos(2 * np.pi * x) + y**2 - 10 * np.cos(2 * np.pi * y)


def plot_frame(istep, nsteps, loom):
    print(f'Plotting frame {istep} / {nsteps}')
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(Z, extent=(-10, 10, -10, 10), origin='lower', cmap='viridis', alpha=0.5)
    ax.contour(X, Y, Z, 10, colors='black', alpha=0.4)
    ax.scatter(
        0,
        0,
        marker='x',
        color='black',
        alpha=0.5,
    )
    ax.scatter(
        [history[istep][iparticle]['position']['x'][0][0] for iparticle in range(nparticles)],
        [history[istep][iparticle]['position']['x'][0][1] for iparticle in range(nparticles)],
        marker='o',
        color='blue',
        alpha=0.5,
    )
    ax.scatter(
        [history[istep][iparticle]['best']['x'][0][0] for iparticle in range(nparticles)],
        [history[istep][iparticle]['best']['x'][0][1] for iparticle in range(nparticles)],
        marker='o',
        color='green',
        alpha=0.5,
    )
    ax.quiver(
        [history[istep][iparticle]['position']['x'][0][0] for iparticle in range(nparticles)],
        [history[istep][iparticle]['position']['x'][0][1] for iparticle in range(nparticles)],
        [
            history[istep][iparticle]['best']['x'][0][0] - history[istep][iparticle]['position']['x'][0][0]
            for iparticle in range(nparticles)
        ],
        [
            history[istep][iparticle]['best']['x'][0][1] - history[istep][iparticle]['position']['x'][0][1]
            for iparticle in range(nparticles)
        ],
        color='green',
        units='dots',
        width=1,
        angles='xy',
        scale_units='xy',
        scale=1,
    )
    ax.scatter(
        best_history[istep]['x'][0][0],
        best_history[istep]['x'][0][1],
        marker='o',
        color='red',
        alpha=0.5,
    )
    ax.quiver(
        [history[istep][iparticle]['position']['x'][0][0] for iparticle in range(nparticles)],
        [history[istep][iparticle]['position']['x'][0][1] for iparticle in range(nparticles)],
        [
            best_history[istep]['x'][0][0] - history[istep][iparticle]['position']['x'][0][0]
            for iparticle in range(nparticles)
        ],
        [
            best_history[istep]['x'][0][1] - history[istep][iparticle]['position']['x'][0][1]
            for iparticle in range(nparticles)
        ],
        color='red',
        units='dots',
        width=0.2,
        angles='xy',
        scale_units='xy',
        scale=1,
    )
    ax.quiver(
        [history[istep][iparticle]['position']['x'][0][0] for iparticle in range(nparticles)],
        [history[istep][iparticle]['position']['x'][0][1] for iparticle in range(nparticles)],
        [history[istep][iparticle]['velocity'][0][0] for iparticle in range(nparticles)],
        [history[istep][iparticle]['velocity'][0][1] for iparticle in range(nparticles)],
        color='blue',
        units='dots',
        width=1,
        angles='xy',
        scale_units='xy',
        scale=0.1,
    )
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    loom.save_frame(fig, istep)


if __name__ == '__main__':
    with Path('data.pkl').open('rb') as f:
        swarm_history = pickle.load(f)  # noqa: S301
    history = swarm_history['history']
    best_history = swarm_history['best_history']
    nsteps = len(history)
    nparticles = len(history[0])
    g = np.linspace(-10, 10, 1000)
    X, Y = np.meshgrid(g, g)
    Z = func(X, Y)
    with Loom('pso.gif', fps=30, parallel=True, overwrite=True) as loom:
        Parallel(n_jobs=-1)(delayed(plot_frame)(i, nsteps, loom) for i in range(nsteps))
