#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "matplotlib==3.11.0",
#   "numpy==2.5.1",
# ]
# ///
"""Plot deterministic multistart trajectories across Rastrigin basins."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')
ROOT = Path(__file__).parent
payload = json.loads((ROOT / 'data.json').read_text())
starts = np.asarray(payload['starts'])
endpoints = np.asarray(payload['endpoints'])
best = payload['best_run_index']
low, high = payload['bounds']

grid = np.linspace(low, high, 500)
x, y = np.meshgrid(grid, grid)
z = 20 + x * x - 10 * np.cos(2 * np.pi * x) + y * y - 10 * np.cos(2 * np.pi * y)

plt.style.use('seaborn-v0_8-whitegrid')
figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
axis.contourf(x, y, z, levels=36, cmap='YlGnBu', alpha=0.92)
axis.contour(x, y, z, levels=18, colors='#334155', linewidths=0.4, alpha=0.3)
for index, (start, endpoint) in enumerate(zip(starts, endpoints, strict=True)):
    color = '#dc2626' if index == best else '#7c3aed'
    axis.annotate(
        '',
        xy=endpoint,
        xytext=start,
        arrowprops={'arrowstyle': '->', 'color': color, 'alpha': 0.55, 'lw': 1},
    )
axis.scatter(
    starts[:, 0],
    starts[:, 1],
    s=35,
    facecolors='white',
    edgecolors='#0f172a',
    label='starts',
)
axis.scatter(
    endpoints[:, 0],
    endpoints[:, 1],
    s=30,
    color='#7c3aed',
    alpha=0.75,
    label='local endpoints',
)
axis.scatter(
    endpoints[best, 0],
    endpoints[best, 1],
    s=180,
    marker='*',
    color='#dc2626',
    edgecolors='white',
    label='selected best',
)
axis.set(
    xlim=(low, high),
    ylim=(low, high),
    xlabel='x',
    ylabel='y',
    title='Multistart discovers the global Rastrigin basin',
)
axis.legend(frameon=True, loc='upper right')
figure.savefig(ROOT / 'multistart.png', dpi=180)
