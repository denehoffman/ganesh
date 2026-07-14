#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "matplotlib==3.11.0",
#   "numpy==2.5.1",
#   "pillow==12.3.0",
# ]
# ///
"""Animate PSO positions, memories, velocities, and global best on Rastrigin."""

import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')

ROOT = Path(__file__).parent
payload = json.loads((ROOT / 'data.json').read_text())
history = payload['history']
positions = np.asarray([frame['positions'] for frame in history])
velocities = np.asarray([frame['velocities'] for frame in history])
personal_bests = np.asarray([frame['personal_bests'] for frame in history])
global_bests = np.asarray([frame['global_best'] for frame in history])
low, high = payload['plot_bounds']
grid = np.linspace(low, high, 420)
x, y = np.meshgrid(grid, grid)
z = 10 + x * x - 10 * np.cos(2 * np.pi * x) + y * y - 10 * np.cos(2 * np.pi * y)

plt.style.use('seaborn-v0_8-whitegrid')
figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
axis.contourf(x, y, z, levels=28, cmap='YlGnBu', alpha=0.9)
axis.contour(x, y, z, levels=12, colors='#334155', alpha=0.18, linewidths=0.5)
personal = axis.scatter(
    [],
    [],
    s=38,
    facecolors='none',
    edgecolors='#7c3aed',
    linewidths=0.7,
    alpha=0.55,
    label='personal best',
)
velocity_arrows = axis.quiver(
    positions[0, :, 0],
    positions[0, :, 1],
    np.zeros(len(positions[0])),
    np.zeros(len(positions[0])),
    color='#0f766e',
    alpha=0.65,
    angles='xy',
    scale_units='xy',
    scale=0.1,
    width=0.0018,
    headwidth=3.5,
    label='velocity',
)
personal_arrows = axis.quiver(
    positions[0, :, 0],
    positions[0, :, 1],
    np.zeros(len(positions[0])),
    np.zeros(len(positions[0])),
    color='#7c3aed',
    alpha=0.42,
    angles='xy',
    scale_units='xy',
    scale=1,
    width=0.0014,
    headwidth=3.5,
    label='toward personal best',
)
global_arrows = axis.quiver(
    positions[0, :, 0],
    positions[0, :, 1],
    np.zeros(len(positions[0])),
    np.zeros(len(positions[0])),
    color='#dc2626',
    alpha=0.3,
    angles='xy',
    scale_units='xy',
    scale=1,
    width=0.0012,
    headwidth=3.5,
    label='toward global best',
)
particles = axis.scatter(
    [],
    [],
    s=34,
    color='#2563eb',
    edgecolors='white',
    linewidths=0.45,
    label='particle',
    zorder=5,
)
current_best = axis.scatter(
    [],
    [],
    marker='*',
    s=220,
    color='#ef4444',
    edgecolor='#7f1d1d',
    linewidth=0.9,
    label='current global best',
    zorder=7,
)
axis.scatter(
    [0],
    [0],
    marker='X',
    s=90,
    color='#111827',
    edgecolor='white',
    linewidth=0.6,
    label='true minimum',
    zorder=6,
)
title = axis.set_title('')
axis.set(xlim=(low, high), ylim=(low, high), xlabel='x', ylabel='y')
axis.legend(loc='upper right', frameon=True, fontsize=8)

target = payload['minimum_value']
final_frame = next(
    (index for index, snapshot in enumerate(history) if snapshot['global_best_value'] <= target),
    len(history) - 1,
)
frames = np.arange(final_frame + 1)


def update(frame: int):
    particles.set_offsets(positions[frame])
    personal.set_offsets(personal_bests[frame])
    current_best.set_offsets(global_bests[frame][None, :])

    velocity_arrows.set_offsets(positions[frame])
    velocity_arrows.set_UVC(velocities[frame, :, 0], velocities[frame, :, 1])
    personal_vectors = personal_bests[frame] - positions[frame]
    personal_arrows.set_offsets(positions[frame])
    personal_arrows.set_UVC(personal_vectors[:, 0], personal_vectors[:, 1])
    global_vectors = global_bests[frame] - positions[frame]
    global_arrows.set_offsets(positions[frame])
    global_arrows.set_UVC(global_vectors[:, 0], global_vectors[:, 1])

    snapshot = history[frame]
    title.set_text(
        f'Particle swarm on Rastrigin · step {snapshot["step"]} · '
        f'best = {snapshot["global_best_value"]:.3g}'
    )
    return (
        particles,
        personal,
        current_best,
        velocity_arrows,
        personal_arrows,
        global_arrows,
        title,
    )


movie = animation.FuncAnimation(figure, update, frames=frames, interval=67, blit=False)
movie.save(ROOT / 'pso.gif', writer=animation.PillowWriter(fps=15), dpi=120)
