"""保障配額 rebate: duplicate 5* and 6* operators return quota that buys more pulls.

This is an expected-value approximation layered on top of the exact engine (see
docs/assumptions.md, R13): it scales a stock of pulls by the long-run rebate rate.
"""

from __future__ import annotations

import numpy as np

from gacha.rules.endfield import EndfieldCharacterRules


def star_rates(rules: EndfieldCharacterRules) -> tuple[float, float]:
    """Long-run 6* and 5* operators per pull, from the stationary law of (t, g) where t is the
    6* pity counter and g the number of pulls since the last 5*-or-better."""
    p = rules.probs
    nt, ng = len(p), rules.five_star_pity
    idx = lambda t, g: t * ng + g  # noqa: E731
    size = nt * ng
    trans = np.zeros((size, size))
    five = np.zeros(size)
    for t in range(nt):
        p6 = p[t]
        for g in range(ng):
            p5 = 1.0 - p6 if g == ng - 1 else min(rules.five_star_rate, 1.0 - p6)
            i = idx(t, g)
            trans[i, idx(0, 0)] += p6
            five[i] = p5
            if p6 < 1.0:
                trans[i, idx(t + 1, 0)] += p5
                rest = 1.0 - p6 - p5
                if rest > 0.0:
                    trans[i, idx(t + 1, g + 1)] += rest
    # stationary distribution: pi (P - I) = 0 with sum(pi) = 1
    a = np.vstack([trans.T - np.eye(size), np.ones(size)])
    b = np.zeros(size + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(a, b, rcond=None)[0]
    six = np.repeat(np.array(p), ng)
    return float(pi @ six), float(pi @ five)


def quota_per_pull(
    rules: EndfieldCharacterRules,
    owned_five_share: float,
    owned_offrate_share: float,
    owns_up: bool = False,
) -> float:
    """Expected 保障配額 per pull. Shares are the fraction of the 5* pool, and of the off-rate
    6* pool, that the player already owns (each operator in a pool is equally likely)."""
    for name, v in (
        ("owned_five_share", owned_five_share),
        ("owned_offrate_share", owned_offrate_share),
    ):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    rate6, rate5 = star_rates(rules)
    six_dupe = (1.0 - rules.up_share) * owned_offrate_share + (rules.up_share if owns_up else 0.0)
    return (
        rules.quota_five_dupe * rate5 * owned_five_share + rules.quota_six_dupe * rate6 * six_dupe
    )


def effective_stock(
    stock: int, quota_now: float, per_pull: float, quota_per_permit: int = 25
) -> int:
    """Pulls available once the rebate is reinvested: N = stock + (quota_now + per_pull N) / 25."""
    if per_pull >= quota_per_permit:
        raise ValueError("rebate per pull must be below the price of a permit")
    return int((stock + quota_now / quota_per_permit) / (1.0 - per_pull / quota_per_permit))


def arsenal_per_pull(rules: EndfieldCharacterRules) -> float:
    """Expected 武庫配額 earned per character pull (every operator pulled grants some)."""
    rate6, rate5 = star_rates(rules)
    rate4 = 1.0 - rate5 - rate6
    return rules.arsenal_four * rate4 + rules.arsenal_five * rate5 + rules.arsenal_six * rate6
