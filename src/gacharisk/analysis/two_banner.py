"""Two banners with independent pity that share one stock of pulls.

The player pulls the first banner with at most ``cap`` own pulls (stopping early on success),
then spends whatever is left on the second banner. Because the two pity counters are separate,
the joint law is a product and every quantity below is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gacharisk.kernel.chain import EnumeratedChain
from gacharisk.kernel.forward import hitting_time
from gacharisk.kernel.types import Model


@dataclass(frozen=True)
class CapRow:
    cap: int  # most own pulls spent on the first banner
    p_first: float  # P(first target obtained)
    p_second: float  # P(second target obtained with what is left)
    p_both: float


def first_up_pmf(model: Model) -> np.ndarray:
    """pmf[j] = P(target obtained with exactly j own pulls); the model's cap must reach the
    banner's guarantee so that the distribution is proper."""
    ht = hitting_time(EnumeratedChain.from_model(model))
    if ht.p_fail > 1e-12:
        raise ValueError("the model can fail before its guarantee; raise the banner cap")
    return ht.f_succ


def cap_table(pmf_first: np.ndarray, pmf_second: np.ndarray, stock: int) -> list[CapRow]:
    """One row per cap 0..stock on the first banner."""
    if stock < 0:
        raise ValueError("stock must be >= 0")
    cdf_first = np.cumsum(pmf_first)
    cdf_second = np.cumsum(pmf_second)

    def second_within(pulls: int) -> float:
        return 0.0 if pulls < 0 else float(cdf_second[min(pulls, len(cdf_second) - 1)])

    rows: list[CapRow] = []
    both = 0.0
    for cap in range(stock + 1):
        if 0 < cap < len(pmf_first):
            both += float(pmf_first[cap]) * second_within(stock - cap)
        elif cap == 0:
            both = float(pmf_first[0]) * second_within(stock)
        p_first = float(cdf_first[min(cap, len(cdf_first) - 1)])
        p_second = both + (1.0 - p_first) * second_within(stock - cap)
        rows.append(CapRow(cap, p_first, p_second, both))
    return rows


def recommend(rows: list[CapRow], tolerance: float = 0.01) -> CapRow:
    """Smallest cap whose P(both) is within ``tolerance`` of the best achievable."""
    best = max(r.p_both for r in rows)
    return next(r for r in rows if r.p_both >= best - tolerance)
