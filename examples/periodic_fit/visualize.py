#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "matplotlib==3.11.0",
#   "numpy==2.5.1",
# ]
# ///
"""Visualize a sinusoid fit and the phase seam crossed by the optimizer."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')
ROOT = Path(__file__).parent
payload = json.loads((ROOT / 'data.json').read_text())

time = np.asarray(payload['time'])
observed = np.asarray(payload['observed'])
fitted = np.asarray(payload['fitted'])
truth = np.asarray(payload['truth'])
initial = np.asarray(payload['initial'])
fit = np.asarray(payload['fit'])

plt.style.use('seaborn-v0_8-whitegrid')
figure, (signal, phase) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
signal.scatter(time, observed, s=18, color='#64748b', alpha=0.65, label='observations')
signal.plot(time, fitted, color='#7c3aed', linewidth=2.5, label='fit')
signal.plot(
    time,
    truth[0] + truth[1] * np.cos(time - truth[2]),
    '--',
    color='#0f766e',
    label='truth',
)
signal.set(xlabel='time', ylabel='signal', title='Mixed-domain sinusoid fit')
signal.legend(frameon=True)

angles = np.linspace(-np.pi, np.pi, 600, endpoint=False)
phase.plot(
    np.cos(angles),
    np.sin(angles),
    color='#cbd5e1',
    linewidth=8,
    solid_capstyle='round',
)
phase.scatter(
    [np.cos(initial[2])],
    [np.sin(initial[2])],
    s=110,
    color='#f59e0b',
    label='initial phase',
    zorder=3,
)
phase.scatter(
    [np.cos(truth[2])],
    [np.sin(truth[2])],
    s=110,
    marker='*',
    color='#0f766e',
    label='true phase',
    zorder=3,
)
phase.scatter(
    [np.cos(fit[2])],
    [np.sin(fit[2])],
    s=80,
    marker='x',
    color='#7c3aed',
    label='fitted phase',
    zorder=3,
)
phase.plot(
    [0, -1.12],
    [0, 0],
    color='#ef4444',
    linestyle=':',
    linewidth=1.5,
    label='display seam',
)
phase.set(
    xlim=(-1.25, 1.25),
    ylim=(-1.25, 1.25),
    xticks=[],
    yticks=[],
    title='Adjacent across the periodic seam',
)
phase.set_aspect('equal')
phase.legend(frameon=True, loc='upper center', ncols=2)

figure.savefig(ROOT / 'periodic_fit.png', dpi=180)
