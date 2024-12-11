#!/usr/bin/env python3
import pickle
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    with Path('data.pkl').open('rb') as f:
        data = pickle.load(f)  # noqa: S301
    flat_chain = data.get('flat chain')
    fig = corner.corner(np.array([np.array([x[0][0], x[0][1], x[0][2], x[0][3], x[0][4]]) for x in flat_chain]))
    plt.show()
