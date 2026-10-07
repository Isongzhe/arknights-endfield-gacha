"""Distributional risk metrics on HittingTime results (spec §5.8).

``which="stop"``: T_stop = paid pulls until absorption of either kind (proper distribution).
``which="success"``: success-only PMF, normalized, i.e. conditional on success.
``completion`` is the only metric that uses the unnormalized success mass.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gacha.kernel.types import HittingTime

RESIDUAL_TOL = 1e-12


def pmf_of(ht: HittingTime, which: str = "stop") -> np.ndarray:
    if which == "stop":
        if ht.residual > RESIDUAL_TOL:
            raise ValueError(
                "stop distribution is truncated (residual > 0); recompute with a larger horizon"
            )
        return ht.pmf_stop
    if which == "success":
        return ht.f_succ
    raise ValueError(f"which must be 'stop' or 'success', got {which!r}")


def _normalized(ht: HittingTime, which: str) -> np.ndarray:
    pmf = pmf_of(ht, which)
    total = float(pmf.sum())
    if total <= 0.0:
        raise ValueError("distribution has zero mass")
    return pmf / total


def from_pmf(pmf: np.ndarray) -> HittingTime:
    """Wrap a plain PMF (e.g. a convolution) so every metric can be applied to it."""
    arr = np.asarray(pmf, dtype=float)
    return HittingTime(arr, np.zeros_like(arr), 0.0)


def convolve(pmf: np.ndarray, m: int) -> np.ndarray:
    """m-fold convolution of a PMF indexed by paid pulls (the paper's i.i.d. stages)."""
    if m < 1:
        raise ValueError("m must be >= 1")
    out = np.array([1.0])
    for _ in range(m):
        out = np.convolve(out, np.asarray(pmf, dtype=float))
    return out


def mean(ht: HittingTime, which: str = "stop") -> float:
    pmf = _normalized(ht, which)
    return float((np.arange(len(pmf)) * pmf).sum())


def var(ht: HittingTime, which: str = "stop") -> float:
    pmf = _normalized(ht, which)
    j = np.arange(len(pmf))
    m = float((j * pmf).sum())
    return float(((j - m) ** 2 * pmf).sum())


def sd(ht: HittingTime, which: str = "stop") -> float:
    return float(np.sqrt(var(ht, which)))


def cv(ht: HittingTime, which: str = "stop") -> float:
    m = mean(ht, which)
    return sd(ht, which) / m if m > 0 else float("nan")


def cdf(ht: HittingTime, b: int, which: str = "stop") -> float:
    """P(T <= b)."""
    if b < 0:
        return 0.0
    pmf = _normalized(ht, which)
    return float(pmf[: min(b, len(pmf) - 1) + 1].sum())


def survival(ht: HittingTime, b: int, which: str = "stop") -> float:
    """P(T >= b), the paper's tail convention."""
    return 1.0 - cdf(ht, b - 1, which)


def completion(ht: HittingTime, b: int) -> float:
    """P(success and T <= b), unnormalized."""
    if b < 0:
        return 0.0
    return float(ht.f_succ[: min(b, ht.horizon) + 1].sum())


def quantile(ht: HittingTime, alpha: float, which: str = "stop") -> int:
    """Smallest j with P(T <= j) >= alpha."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    c = np.cumsum(_normalized(ht, which))
    return int(np.searchsorted(c, alpha - 1e-12, side="left"))


def var_at(ht: HittingTime, alpha: float, which: str = "stop") -> int:
    return quantile(ht, alpha, which)


def cvar(ht: HittingTime, alpha: float, which: str = "stop") -> float:
    """E[T | T >= VaR_alpha] (discrete upper-tail conditional mean, as in the paper)."""
    v = quantile(ht, alpha, which)
    pmf = _normalized(ht, which)
    tail = pmf[v:]
    j = np.arange(v, len(pmf))
    return float((j * tail).sum() / tail.sum())


def expected_excess(ht: HittingTime, b: int, which: str = "stop") -> float:
    """E[(T - b)^+]."""
    pmf = _normalized(ht, which)
    j = np.arange(len(pmf))
    return float(np.clip(j - b, 0, None) @ pmf)


def skewness(ht: HittingTime, which: str = "stop") -> float:
    pmf = _normalized(ht, which)
    j = np.arange(len(pmf))
    m = float((j * pmf).sum())
    s2 = float(((j - m) ** 2 * pmf).sum())
    m3 = float(((j - m) ** 3 * pmf).sum())
    return m3 / s2**1.5 if s2 > 0 else float("nan")


def entropy_nats(ht: HittingTime, which: str = "stop") -> float:
    pmf = _normalized(ht, which)
    p = pmf[pmf > 0]
    return float(-(p * np.log(p)).sum())


def summary(
    ht: HittingTime,
    alphas: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
    budgets: Sequence[int] = (),
    which: str = "stop",
) -> pd.DataFrame:
    rows: list[tuple[str, float]] = [
        ("p_success", ht.p_success),
        ("mean", mean(ht, which)),
        ("sd", sd(ht, which)),
        ("skewness", skewness(ht, which)),
        ("entropy_nats", entropy_nats(ht, which)),
    ]
    for a in alphas:
        tag = f"{round(a * 100):d}"
        rows.append((f"q{tag}", quantile(ht, a, which)))
        rows.append((f"cvar{tag}", cvar(ht, a, which)))
    for b in budgets:
        rows.append((f"completion_at_{b}", completion(ht, b)))
        rows.append((f"expected_excess_at_{b}", expected_excess(ht, b, which)))
    return pd.DataFrame(rows, columns=["metric", "value"])
