"""Exact-vs-Monte-Carlo comparison table with standard errors."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

import pandas as pd

from gacha.kernel.types import HittingTime
from gacha.risk import metrics as rk

from .simulate import Paths


def compare(
    ht: HittingTime, paths: Paths, thresholds: Sequence[int] = (), tol_se: float = 3.0
) -> pd.DataFrame:
    n = len(paths.paid)
    rows: list[dict[str, float | str | bool]] = []

    def add(metric: str, exact: float, mc: float, se: float) -> None:
        z = (mc - exact) / se if se > 0 else 0.0
        rows.append(
            {"metric": metric, "exact": exact, "mc": mc, "se": se, "z": z, "ok": abs(z) <= tol_se}
        )

    p = ht.p_success
    add("p_success", p, float((paths.absorb == 1).mean()), sqrt(max(p * (1 - p), 1e-300) / n))
    add("mean", rk.mean(ht), float(paths.paid.mean()), float(paths.paid.std(ddof=1)) / sqrt(n))
    for b in thresholds:
        s = rk.survival(ht, int(b))
        add(
            f"survival_at_{b}",
            s,
            float((paths.paid >= b).mean()),
            sqrt(max(s * (1 - s), 1e-300) / n),
        )
    return pd.DataFrame(rows)
