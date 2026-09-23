"""Continuity-corrected normal approximation diagnostics (paper §5.3)."""

from __future__ import annotations

from math import erf, sqrt

import numpy as np

from gacha.kernel.types import HittingTime

from . import metrics as rk


def _phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_approx_cdf(ht: HittingTime, b: int, which: str = "stop") -> float:
    """Phi((b + 0.5 - mean) / sd)."""
    return _phi((b + 0.5 - rk.mean(ht, which)) / rk.sd(ht, which))


def max_abs_cdf_error(ht: HittingTime, which: str = "stop") -> float:
    pmf = rk.pmf_of(ht, which)
    c = np.cumsum(pmf / pmf.sum())
    return max(abs(float(c[j]) - normal_approx_cdf(ht, j, which)) for j in range(len(c)))
