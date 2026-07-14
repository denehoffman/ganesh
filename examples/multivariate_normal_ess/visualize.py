#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "corner==2.3.0",
#   "matplotlib==3.11.0",
#   "numpy==2.5.1",
# ]
# ///
"""Render a corner plot, traces, and diagnostics for the Gaussian ESS run."""

import json
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

ROOT = Path(__file__).parent
payload = json.loads((ROOT / 'data.json').read_text())
chains = np.asarray(payload['chains'])
burn, thin = payload['burn'], payload['thin']
labels = payload['parameter_names']
samples = chains[:, burn::thin].reshape(-1, chains.shape[-1])

figure = corner.corner(
    samples,
    labels=labels,
    color='#2563eb',
    show_titles=True,
    plot_datapoints=False,
    fill_contours=True,
)
figure.suptitle(payload['title'], fontsize=16)
figure.savefig(ROOT / 'corner_plot.png', dpi=240)
plt.close(figure)

figure, axes = plt.subplots(len(labels), 1, figsize=(12, 10), sharex=True, constrained_layout=True)
for index, axis in enumerate(axes):
    axis.plot(chains[:, :, index].T, color='#2563eb', alpha=0.07, linewidth=0.45)
    axis.axvspan(0, burn, color='#fdba74', alpha=0.3)
    axis.set_ylabel(labels[index])
axes[-1].set_xlabel('ensemble step')
figure.savefig(ROOT / 'traces.png', dpi=240)
plt.close(figure)

figure, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
axis.bar(labels, payload['effective_sample_size'], color='#16a34a')
axis.set(title='Effective sample size after burn-in', ylabel='samples')
figure.savefig(ROOT / 'iat.png', dpi=240)
