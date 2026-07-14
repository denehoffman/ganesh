#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "matplotlib==3.11.0",
#   "numpy==2.5.1",
# ]
# ///
"""Render traces, diagnostics, and the sampled Himmelblau landscape."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')

ROOT = Path(__file__).parent
payload = json.loads((ROOT / 'data.json').read_text())
chains = np.asarray(payload['chains'])
burn, thin = payload['burn'], payload['thin']
samples = chains[:, burn::thin].reshape(-1, 2)
plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
for index, axis in enumerate(axes):
    axis.plot(chains[:, :, index].T, color='#2563eb', alpha=0.08, linewidth=0.5)
    axis.axvspan(0, burn, color='#fdba74', alpha=0.3, label='burn-in')
    axis.set_ylabel(payload['parameter_names'][index])
axes[0].legend(frameon=False)
axes[-1].set_xlabel('ensemble step')
fig.suptitle(payload['title'], fontsize=16)
fig.savefig(ROOT / 'traces.png', dpi=240)
plt.close(fig)

grid = np.linspace(-6, 6, 500)
x, y = np.meshgrid(grid, grid)
energy = (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2
fig, axis = plt.subplots(figsize=(9, 8), constrained_layout=True)
axis.contourf(x, y, np.exp(-energy / 8), levels=32, cmap='YlGnBu')
axis.scatter(samples[::8, 0], samples[::8, 1], s=3, color='#dc2626', alpha=0.2)
axis.set(xlabel='x', ylabel='y', title='Four modes recovered by ESS')
fig.savefig(ROOT / 'scatter.png', dpi=220)
plt.close(fig)

fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
labels = payload['parameter_names']
axis.bar(labels, payload['effective_sample_size'], color='#2563eb')
axis.set(title='Effective sample size', ylabel='samples')
axis.text(
    0.99,
    0.95,
    f'mean acceptance = {payload["acceptance_rate"]:.1%}',
    ha='right',
    va='top',
    transform=axis.transAxes,
)
fig.savefig(ROOT / 'iat.png', dpi=240)
