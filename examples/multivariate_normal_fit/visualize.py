#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "corner==2.3.0",
#   "matplotlib==3.11.0",
#   "numpy==2.5.1",
# ]
# ///
"""Render the generated data, optimizer result, posterior traces, and corner plot."""

import json
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

ROOT = Path(__file__).parent
payload = json.loads((ROOT / 'data.json').read_text())
data = np.asarray(payload['observations'])
chains = np.asarray(payload['chains'])
burn = payload['burn']
labels = payload['parameter_names']
truth = np.asarray(payload['truth'])
fit = np.asarray(payload['fit'])
samples = chains[:, burn:].reshape(-1, chains.shape[-1])

figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
axis.scatter(
    data[:, 0],
    data[:, 1],
    s=18,
    color='#60a5fa',
    alpha=0.35,
    edgecolors='none',
    label='observations',
)
axis.scatter(*truth[:2], marker='*', s=190, color='#f59e0b', edgecolor='white', label='truth')
axis.scatter(*fit[:2], marker='x', s=100, linewidth=2.5, color='#dc2626', label='fit')
axis.set(xlabel='x₀', ylabel='x₁', title='Synthetic correlated Gaussian data')
axis.legend(frameon=True)
figure.savefig(ROOT / 'data.png', dpi=240)
plt.close(figure)

figure, axes = plt.subplots(len(labels), 1, figsize=(12, 11), sharex=True, constrained_layout=True)
for index, axis in enumerate(axes):
    axis.plot(chains[:, :, index].T, color='#2563eb', alpha=0.07, linewidth=0.45)
    axis.axhline(truth[index], color='#f59e0b', linewidth=1)
    axis.axvspan(0, burn, color='#fdba74', alpha=0.3)
    axis.set_ylabel(labels[index])
axes[-1].set_xlabel('ensemble step')
figure.savefig(ROOT / 'traces.png', dpi=240)
plt.close(figure)

figure, axes = plt.subplots(len(labels), 1, figsize=(12, 11), sharex=True, constrained_layout=True)
for index, axis in enumerate(axes):
    axis.plot(chains[:, burn:, index].T, color='#2563eb', alpha=0.07, linewidth=0.45)
    axis.axhline(truth[index], color='#f59e0b', linewidth=1)
    axis.set_ylabel(labels[index])
axes[-1].set_xlabel('post-burn ensemble step')
figure.savefig(ROOT / 'traces_burned.png', dpi=240)
plt.close(figure)

figure = corner.corner(
    samples,
    labels=labels,
    truths=truth,
    color='#2563eb',
    show_titles=True,
    plot_datapoints=False,
    fill_contours=True,
)
corner.overplot_lines(figure, fit, color='#dc2626')
figure.suptitle(payload['title'], fontsize=16)
figure.savefig(ROOT / 'corner.png', dpi=240)
plt.close(figure)
