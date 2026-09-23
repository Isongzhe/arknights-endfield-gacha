"""Line plots for PMF, CDF, survival and pity schedules (x axis = paid pulls)."""

from __future__ import annotations

import numpy as np


def plot_pmf(ax, pmf, label=None, color=None):
    pmf = np.asarray(pmf, dtype=float)
    ax.plot(np.arange(len(pmf)), pmf, drawstyle="steps-mid", label=label, color=color)
    ax.set_xlabel("paid pulls")
    ax.set_ylabel("probability")


def plot_cdf(ax, pmf, label=None, color=None):
    pmf = np.asarray(pmf, dtype=float)
    ax.plot(np.arange(len(pmf)), np.cumsum(pmf), drawstyle="steps-post", label=label, color=color)
    ax.set_xlabel("paid pulls b")
    ax.set_ylabel("P(T ≤ b)")


def plot_survival(ax, pmf, label=None, color=None):
    pmf = np.asarray(pmf, dtype=float)
    c = np.cumsum(pmf)
    surv = 1.0 - np.concatenate([[0.0], c[:-1]])  # P(T >= b)
    ax.plot(np.arange(len(pmf)), surv, drawstyle="steps-post", label=label, color=color)
    ax.set_xlabel("paid pulls b")
    ax.set_ylabel("P(T ≥ b)")


def plot_schedule(ax, probs, label=None, color=None):
    probs = np.asarray(probs, dtype=float)
    ax.plot(np.arange(len(probs)), probs, drawstyle="steps-post", label=label, color=color)
    ax.set_xlabel("pity counter t")
    ax.set_ylabel("P(6★ on the next pull)")
