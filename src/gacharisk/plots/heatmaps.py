"""Single-hue heatmaps over a regular (x, y) grid."""

from __future__ import annotations

import numpy as np

from .style import SEQUENTIAL_CMAP


def heatmap(ax, z, x, y, xlabel, ylabel, cbar_label, cmap=SEQUENTIAL_CMAP, vmin=None, vmax=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = (x[1] - x[0]) if len(x) > 1 else 1.0
    dy = (y[1] - y[0]) if len(y) > 1 else 1.0
    im = ax.imshow(
        np.asarray(z, dtype=float),
        origin="lower",
        aspect="auto",
        extent=(x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.grid(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.figure.colorbar(im, ax=ax).set_label(cbar_label)
    return im
